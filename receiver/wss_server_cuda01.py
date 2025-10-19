# wss_asr_server_translate_buffer.py
# WS/WSS ASR server with CUDA: client sends 16 kHz mono float32 PCM (binary).
# Server aggregates ~2.4s sliding windows for robust decoding, translates to English,
# prints "TRANS: ..." on the server and appends to a daily file. Non-blocking queue.

import os, ssl, json, asyncio, logging, time, datetime
import numpy as np
from websockets.asyncio.server import serve
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---- Config (via env) ---------------------------------------------------------
BIND_HOST = os.getenv("BIND_HOST", "192.168.0.110")
BIND_PORT = int(os.getenv("BIND_PORT", "8765"))
MODEL     = os.getenv("WHISPER_MODEL", "small")      # tiny/tiny.en/small/medium/large-v3
LANGUAGE  = os.getenv("LANGUAGE", "auto")            # input language ("auto" to detect)
TASK      = os.getenv("TASK", "translate")           # "translate" -> English (default) | "transcribe"
PING_INT  = int(os.getenv("PING_SECS", "20"))
TLS_CERT  = os.getenv("TLS_CERT")                    # for WSS (optional)
TLS_KEY   = os.getenv("TLS_KEY")
REQUIRED_PATH = os.getenv("WS_PATH", "")             # e.g. "/stream" to require path
DOWNLOAD_ROOT = os.getenv("MODEL_DIR")               # model cache dir (optional)

# Real-time tuning
SILENCE_RMS   = float(os.getenv("SILENCE_RMS", "0.008"))  # 0.006–0.012; raise to ignore noise
TARGET_SEC    = float(os.getenv("TARGET_SEC", "2.4"))     # window length decoded each time
HOP_SEC       = float(os.getenv("HOP_SEC", "0.8"))        # slide hop (match sender chunk)
# --------------------------------------------------------------------------------

def load_model():
    try:
        m = WhisperModel(MODEL, device="cuda", compute_type="float16", download_root=DOWNLOAD_ROOT)
        logging.info("faster-whisper ready on CUDA (float16)")
        return m, "cuda"
    except Exception as e:
        logging.warning(f"CUDA unavailable ({e}); using CPU int8")
        m = WhisperModel(MODEL, device="cpu", compute_type="int8", download_root=DOWNLOAD_ROOT)
        logging.info("faster-whisper ready on CPU (int8)")
        return m, "cpu"

model, device_kind = load_model()
clients = set()

def is_silent(x: np.ndarray, thr: float) -> bool:
    return float(np.sqrt(np.mean(x * x))) < thr

def log_file_path():
    d = datetime.date.today().strftime("%Y%m%d")
    return f"translations-{d}.txt"

async def transcribe_once(x: np.ndarray) -> str:
    lang = None if LANGUAGE.lower() in ("auto", "", "none") else LANGUAGE
    segs, _ = model.transcribe(
        x,
        language=lang,
        task=TASK,                         # translate -> English by default
        beam_size=1, best_of=1, temperature=0.0,
        vad_filter=True, no_speech_threshold=0.5,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    return "".join(s.text for s in segs).strip()

async def _pinger(ws):
    while True:
        try:
            await ws.ping()
        except Exception:
            return
        await asyncio.sleep(PING_INT)

# works on both old/new websockets signatures
async def handler(ws, path=None):
    if REQUIRED_PATH and path != REQUIRED_PATH:
        try: await ws.close(code=4404, reason="Not Found")
        finally: return

    global device_kind, model
    peer = f"{ws.remote_address}"
    logging.info(f"client connected {peer} (device={device_kind})")
    clients.add(ws)

    samplerate = 16000
    q = asyncio.Queue(maxsize=3)         # drop-old buffer to avoid backpressure
    stop = asyncio.Event()
    window = np.zeros(0, dtype=np.float32)
    last_sent = ""                       # last text sent to clients
    target_n = int(TARGET_SEC * 16000)   # target samples per decode window
    hop_n    = int(HOP_SEC * 16000)

    async def worker():
        nonlocal window, last_sent
        while not stop.is_set():
            item = await q.get()
            if item is None:
                return
            x = item
            # append to sliding buffer
            if window.size == 0:
                window = x.copy()
            else:
                window = np.concatenate([window, x], dtype=np.float32)
            # only decode when we have enough audio
            if window.size < target_n:
                continue
            # take last target_n samples (recent window)
            w = window[-target_n:]
            try:
                txt = await transcribe_once(w)
            except Exception as e:
                if device_kind == "cuda":
                    logging.warning(f"CUDA hiccup ({e}); switching to CPU")
                    from faster_whisper import WhisperModel as _WM
                    model = _WM(MODEL, device="cpu", compute_type="int8", download_root=DOWNLOAD_ROOT)
                    device_kind = "cpu"
                    txt = await transcribe_once(w)
                else:
                    logging.error(f"transcribe failed: {e}")
                    txt = ""

            if txt and txt != last_sent:
                logging.info(f"TRANS: {txt}")  # <-- you'll SEE translations here
                # log to file
                try:
                    with open(log_file_path(), "a", encoding="utf-8") as f:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        f.write(f"[{ts}] {peer}: {txt}\n")
                except Exception:
                    pass

                payload = json.dumps({"type":"transcript","text":txt,"ts":time.time()}, ensure_ascii=False)
                # reply to sender
                try: await ws.send(payload)
                except Exception: pass
                # broadcast to others
                if clients:
                    await asyncio.gather(*[
                        c.send(payload) for c in list(clients)
                        if (c is not ws and not c.closed)
                    ], return_exceptions=True)
                last_sent = txt

            # slide the window by hop_n to keep latency low
            if window.size > hop_n:
                window = window[hop_n:]
            else:
                window = np.zeros(0, dtype=np.float32)

    ping_task = asyncio.create_task(_pinger(ws))
    work_task = asyncio.create_task(worker())

    try:
        await ws.send(json.dumps({"type":"hello","device":device_kind,"task":TASK,"lang":LANGUAGE,"ts":time.time()}))
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                if not msg:
                    continue
                x = np.frombuffer(msg, dtype=np.float32)
                # normalize to 16k so worker uses consistent timing
                if samplerate != 16000:
                    ratio = samplerate / 16000.0
                    n = int(x.size / ratio)
                    idx = (np.arange(n) * ratio).astype(np.int32)
                    x = x[np.clip(idx, 0, x.size - 1)]
                # gate out very low energy (noise/music-only)
                if is_silent(x, SILENCE_RMS):
                    continue
                # enqueue (drop oldest if full to stay real-time)
                if q.full():
                    try: _ = q.get_nowait()
                    except asyncio.QueueEmpty: pass
                try:
                    q.put_nowait(x)
                except asyncio.QueueFull:
                    pass
            else:
                # simple control frames
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                t = data.get("type")
                if t == "cfg":
                    sr = int(data.get("rate", samplerate))
                    if 8000 <= sr <= 48000:
                        samplerate = sr
                        await ws.send(json.dumps({"type":"ack","rate":samplerate,"device":device_kind}))
                elif t in ("ping","hello"):
                    await ws.send(json.dumps({"type":"ack","t":t,"device":device_kind}))
    except Exception as e:
        logging.info(f"client {peer} closed: {e}")
    finally:
        stop.set()
        try: await q.put(None)
        except Exception: pass
        work_task.cancel()
        ping_task.cancel()
        clients.discard(ws)
        logging.info(f"client disconnected {peer}")

async def main():
    sslctx = None
    if TLS_CERT and TLS_KEY:
        sslctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        sslctx.load_cert_chain(TLS_CERT, TLS_KEY)
        logging.info("TLS enabled (WSS)")
    async with serve(
        handler,
        BIND_HOST, BIND_PORT,
        ssl=sslctx,
        max_size=2**23, max_queue=64,
        ping_interval=PING_INT, ping_timeout=PING_INT,
        close_timeout=5, compression=None
    ):
        scheme = "wss" if sslctx else "ws"
        logging.info(f"ASR server listening on {scheme}://{BIND_HOST}:{BIND_PORT}{REQUIRED_PATH or ''}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())



