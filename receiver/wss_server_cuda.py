# wss_asr_server_cuda.py
# WSS ASR server: receives 16 kHz mono float32 PCM (binary frames),
# transcribes with faster-whisper on CUDA (float16) or falls back to CPU (int8),
# replies {"type":"transcript","text":...} and optionally broadcasts to other clients.

import os, ssl, json, asyncio, logging, time
import numpy as np
from websockets.server import serve  # modern import path
from faster_whisper import WhisperModel

# --- add NVIDIA DLL folders for this conda env (Windows + Python>=3.8) ---
import os, sys
def _add(p):
    if os.path.isdir(p):
        try:
            os.add_dll_directory(p)   # preferred on Win10/11, Python 3.8+
        except Exception:
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

base = sys.prefix  # this conda env
_add(os.path.join(base, "Lib", "site-packages", "nvidia", "cudnn", "bin"))          # cudnn_ops64_9.dll lives here
_add(os.path.join(base, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"))   # cudart64_*.dll
_add(os.path.join(base, "Lib", "site-packages", "nvidia", "cublas", "bin"))         # cublas64_*.dll (if present)
_add(os.path.join(base, "Lib", "site-packages", "nvidia", "cuda_nvrtc", "bin"))     # nvrtc64_*.dll (if present)
# --------------------------------------------------------------------------


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BIND_HOST = os.getenv("BIND_HOST", "192.168.0.110")
BIND_PORT = int(os.getenv("BIND_PORT", "8765"))
MODEL     = os.getenv("WHISPER_MODEL", "small")
PING_INT  = int(os.getenv("PING_SECS", "20"))
TLS_CERT  = os.getenv("TLS_CERT")   # fullchain.pem
TLS_KEY   = os.getenv("TLS_KEY")    # privkey.pem

# Load model (CUDA float16 if available; else CPU int8)
def load_model():
    try:
        m = WhisperModel(MODEL, device="cuda", compute_type="float16")
        logging.info("faster-whisper ready on CUDA (float16)")
        return m, "cuda"
    except Exception as e:
        logging.warning(f"CUDA unavailable ({e}); using CPU int8")
        m = WhisperModel(MODEL, device="cpu", compute_type="int8")
        logging.info("faster-whisper ready on CPU (int8)")
        return m, "cpu"

model, device_kind = load_model()
clients = set()

def is_silent(x: np.ndarray, thr: float = 0.006) -> bool:
    return float(np.sqrt(np.mean(x * x))) < thr

async def transcribe_chunk(x: np.ndarray) -> str:
    segs, _ = model.transcribe(
        x,
        language=None,
        task="transcribe",
        beam_size=1, best_of=1, temperature=0.0,
        vad_filter=True, no_speech_threshold=0.6,
        condition_on_previous_text=False,
    )
    return "".join(s.text for s in segs).strip()

async def _pinger(ws):
    while True:
        try:
            await ws.ping()
        except Exception:
            return
        await asyncio.sleep(PING_INT)

# ⚠️ NOTE: 'path' is optional to support both old and new websockets versions
async def handler(ws, path=None):
    global device_kind
    peer = f"{ws.remote_address}"
    logging.info(f"client connected {peer} (device={device_kind})")
    clients.add(ws)

    samplerate = 16000
    last_tx = ""
    ping_task = asyncio.create_task(_pinger(ws))

    # one-time hello
    try:
        await ws.send(json.dumps({"type":"hello","device":device_kind,"ts":time.time()}))
    except Exception:
        pass

    try:
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                if not msg:
                    continue
                x = np.frombuffer(msg, dtype=np.float32)

                # quick nearest downsample if sender != 16k
                if samplerate != 16000:
                    ratio = samplerate / 16000.0
                    n = int(x.size / ratio)
                    idx = (np.arange(n) * ratio).astype(np.int32)
                    x = x[np.clip(idx, 0, x.size - 1)]

                if is_silent(x):
                    continue

                try:
                    txt = await transcribe_chunk(x)
                except Exception as e:
                    # runtime CUDA hiccup → switch to CPU once
                    if device_kind == "cuda":
                        logging.warning(f"CUDA error at runtime ({e}); switching to CPU")
                        from faster_whisper import WhisperModel as _WM
                        m = _WM(MODEL, device="cpu", compute_type="int8")
                        # swap
                        globals()["model"] = m
                        device_kind = "cpu"
                        txt = await transcribe_chunk(x)
                    else:
                        logging.error(f"transcribe failed: {e}")
                        txt = ""

                if txt and txt != last_tx:
                    payload = json.dumps({"type":"transcript","text":txt,"ts":time.time()}, ensure_ascii=False)
                    # reply to sender
                    try:
                        await ws.send(payload)
                    except Exception:
                        pass
                    # broadcast to others (optional)
                    if clients:
                        await asyncio.gather(*[
                            c.send(payload) for c in list(clients)
                            if (c is not ws and not c.closed)
                        ], return_exceptions=True)
                    last_tx = txt
            else:
                # Text control frames: {"type":"cfg","rate":16000} or {"type":"ping"/"hello"}
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
                # ignore others
    finally:
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
        BIND_HOST,
        BIND_PORT,
        ssl=sslctx,
        max_size=2**23,
        max_queue=64,
        ping_interval=PING_INT,
        ping_timeout=PING_INT,
        close_timeout=5,
        compression=None,
    ):
        scheme = "wss" if sslctx else "ws"
        logging.info(f"ASR server listening on {scheme}://{BIND_HOST}:{BIND_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
