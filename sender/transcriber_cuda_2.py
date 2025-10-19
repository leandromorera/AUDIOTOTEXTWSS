# sender/transcriber_fwh_gpu.py
# Loopback capture -> faster-whisper (CUDA) -> WS JSON, with keepalive + silence gate
import os, time, json, warnings, argparse, threading, queue, datetime as dt, logging, socket
import numpy as np
import soundcard as sc
from websocket import create_connection
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", message="data discontinuity in recording")

# NumPy>=2 safety shim (if env ever upgrades)
import numpy as _np
try:
    if int(_np.__version__.split('.')[0]) >= 2 and not getattr(_np, "_fromstring_patched", False):
        _np.fromstring = lambda buf, dtype=float, count=-1, sep='': _np.frombuffer(buf, dtype=dtype, count=count) if (sep in ("", None)) else (_np.frombuffer(buf, dtype=dtype, count=count))
        _np._fromstring_patched = True
except Exception:
    pass

def env(k, d=None):
    v = os.getenv(k)
    return d if v in (None, "") else v

ap = argparse.ArgumentParser()
ap.add_argument("--url")
ap.add_argument("--device")
ap.add_argument("--loopback", action="store_true")
ap.add_argument("--chunk", type=float)
ap.add_argument("--rate", type=int)
ap.add_argument("--model")
ap.add_argument("--debug", action="store_true")
ap.add_argument("--silence_rms", type=float)  # 0 disables gate
args = ap.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

WS_URL      = args.url    or env("WS_URL", "ws://127.0.0.1:8765/stream")
MODEL_NAME  = args.model  or env("WHISPER_MODEL", "small")   # good speed/quality on GPU
SAMPLE_RATE = args.rate   or int(env("SAMPLE_RATE", "16000"))
CHUNK_SEC   = args.chunk  or float(env("CHUNK_SEC", "1.5"))  # GPU benefits from a bit larger chunks
DEV_NAME    = args.device or env("LOOPBACK_DEVICE")
USE_LOOP    = args.loopback or (env("LOOPBACK", "true").lower() == "true")
CHANNELS    = int(env("CHANNELS", "1"))
Q_MAX       = int(env("QUEUE_MAX", "6"))
PING_SECS   = int(env("PING_SECS", "15"))
SILENCE_RMS = args.silence_rms if args.silence_rms is not None else float(env("SILENCE_RMS", "0.0"))

# Load faster-whisper on CUDA with float16
logging.info("Loading faster-whisper...")
model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
logging.info("faster-whisper ready (cuda/float16).")

def pick_mic():
    if DEV_NAME:
        return sc.get_microphone(DEV_NAME, include_loopback=USE_LOOP)
    spk = sc.default_speaker()
    logging.info(f"default speaker: {spk.name}")
    return sc.get_microphone(id=str(spk.name), include_loopback=True)

def connect_ws():
    backoff = 1
    while True:
        try:
            ws = create_connection(
                WS_URL,
                timeout=5,
                sockopt=((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                         (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)),
            )
            ws.settimeout(10)
            ws.send(json.dumps({"type":"hello","ts":dt.datetime.utcnow().isoformat()+"Z","note":"fwh gpu sender"}))
            logging.info("ws connected")
            return ws
        except Exception as e:
            logging.warning(f"ws connect failed: {e}; retry {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff*2, 10)

ws = connect_ws()
mic = pick_mic()
logging.info(f"device: {mic.name}")

q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=Q_MAX)
stop = threading.Event()
next_ping = time.time() + PING_SECS

def capture_loop():
    frames = int(SAMPLE_RATE * CHUNK_SEC)
    with mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as rec:
        logging.info("capture started")
        while not stop.is_set():
            try:
                x = rec.record(numframes=frames).astype("float32").reshape(-1)
                if q.full():
                    try: q.get_nowait()
                    except queue.Empty: pass
                q.put_nowait(x)
            except Exception:
                time.sleep(0.02)

def is_silent(x: np.ndarray, thr: float) -> bool:
    if thr <= 0: return False
    return float(np.sqrt(np.mean(x*x))) < thr

t = threading.Thread(target=capture_loop, daemon=True)
t.start()

last_text = ""

def transcribe(x: np.ndarray) -> str:
    # faster-whisper expects 16k mono float32 arrays; that’s what we supply.
    # Use light decoding to maximize speed:
    segs, info = model.transcribe(
        x,
        language=None,                 # autodetect
        task="transcribe",
        beam_size=1,                   # greedy
        best_of=1,
        vad_filter=True,
        no_speech_threshold=0.6,
        temperature=0.0,
        condition_on_previous_text=False,
    )
    out = []
    for s in segs:
        out.append(s.text)
    return "".join(out).strip()

try:
    while True:
        if time.time() >= next_ping:
            try:
                ws.send(json.dumps({"type":"ping","ts":dt.datetime.utcnow().isoformat()+"Z"}))
            except Exception:
                ws = connect_ws()
            next_ping = time.time() + PING_SECS

        try:
            x = q.get(timeout=1)
        except queue.Empty:
            continue

        if is_silent(x, SILENCE_RMS):
            continue

        txt = transcribe(x)
        if txt:
            msg = json.dumps({"type":"transcript","ts":dt.datetime.utcnow().isoformat()+"Z","text":txt}, ensure_ascii=False)
            try:
                ws.send(msg)
            except Exception:
                ws = connect_ws()
                ws.send(msg)
            logging.info(f"tx: {txt[:120]}")
except KeyboardInterrupt:
    pass
finally:
    stop.set()
    logging.info("sender stopped")
