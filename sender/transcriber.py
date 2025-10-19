# Role: AUDIO CAPTURE (Sender) – loopback, threaded, with logs/handshake
import os, time, json, warnings, argparse, threading, queue, datetime as dt, logging
import numpy as np
import soundcard as sc
import whisper
from websocket import create_connection

# Silence noisy (non-fatal) warning from soundcard
warnings.filterwarnings("ignore", message="data discontinuity in recording")

# NumPy>=2 shim (soundcard sometimes calls deprecated fromstring)
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
args = ap.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

WS_URL      = args.url    or env("WS_URL", "ws://192.168.0.110:8765/stream")
MODEL_NAME  = args.model  or env("WHISPER_MODEL", "tiny")        # tiny = faster
SAMPLE_RATE = args.rate   or int(env("SAMPLE_RATE", "16000"))
CHUNK_SEC   = args.chunk  or float(env("CHUNK_SEC", "2"))
DEV_NAME    = args.device or env("LOOPBACK_DEVICE")              # e.g. "Altavoces / Auriculares (Realtek Audio)"
USE_LOOP    = args.loopback or (env("LOOPBACK", "true").lower() == "true")
CHANNELS    = int(env("CHANNELS", "1"))
Q_MAX       = int(env("QUEUE_MAX", "3"))

logging.info(f"Config: ws={WS_URL} model={MODEL_NAME} rate={SAMPLE_RATE}Hz chunk={CHUNK_SEC}s channels={CHANNELS} loopback={USE_LOOP}")
if DEV_NAME:
    logging.info(f"Device (requested): {DEV_NAME}")

def pick_mic():
    try:
        if DEV_NAME:
            return sc.get_microphone(DEV_NAME, include_loopback=USE_LOOP)
        spk = sc.default_speaker()
        logging.info(f"Default speaker for loopback: {spk.name}")
        return sc.get_microphone(id=str(spk.name), include_loopback=True)
    except Exception as e:
        logging.error(f"Failed to open device '{DEV_NAME or 'default loopback'}': {e}")
        logging.info("Available devices (include_loopback=True):")
        for m in sc.all_microphones(include_loopback=True):
            logging.info(f" - {m.name}")
        raise

logging.info("Loading Whisper model (this may take a moment)...")
model = whisper.load_model(MODEL_NAME)
logging.info("Whisper ready.")

def connect_ws():
    while True:
        try:
            logging.info(f"Connecting WS: {WS_URL}")
            ws = create_connection(WS_URL, timeout=5)
            # Send a handshake so the server logs something immediately
            hello = {"type": "hello", "ts": dt.datetime.utcnow().isoformat()+"Z", "note": "sender connected"}
            ws.send(json.dumps(hello))
            logging.info("WebSocket connected and handshake sent.")
            return ws
        except Exception as e:
            logging.warning(f"WS connect failed: {e}; retrying in 1s")
            time.sleep(1)

ws = connect_ws()
mic = pick_mic()
logging.info(f"Using device: {mic.name} (loopback={USE_LOOP})")

q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=Q_MAX)
stop = threading.Event()
drops = 0
captured = 0

def capture_loop():
    global drops, captured
    frames = int(SAMPLE_RATE * CHUNK_SEC)
    with mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as rec:
        logging.info("Capture thread: started.")
        while not stop.is_set():
            try:
                audio = rec.record(numframes=frames).astype("float32").reshape(-1)
                captured += 1
                if q.full():
                    try:
                        q.get_nowait()
                        drops += 1
                    except queue.Empty:
                        pass
                q.put_nowait(audio)
            except Exception as e:
                logging.warning(f"Capture error: {e}")
                time.sleep(0.05)

t = threading.Thread(target=capture_loop, daemon=True)
t.start()

last_text = ""
last_log = time.time()

def transcribe_chunk(x: np.ndarray) -> str:
    try:
        r = model.transcribe(x, task="transcribe", fp16=False)
        return (r.get("text") or "").strip()
    except Exception as e:
        logging.warning(f"Transcribe error: {e}")
        return ""

try:
    while True:
        # Heartbeat every ~5s so you see activity even with silence
        if time.time() - last_log > 5:
            logging.info(f"Heartbeat: chunks captured={captured} queue={q.qsize()} drops={drops}")
            last_log = time.time()

        try:
            audio = q.get(timeout=1)
        except queue.Empty:
            continue

        text = transcribe_chunk(audio)
        if text and text != last_text:
            payload = {"type": "transcript", "ts": dt.datetime.utcnow().isoformat() + "Z", "text": text}
            msg = json.dumps(payload, ensure_ascii=False)
            try:
                ws.send(msg)
            except Exception:
                ws = connect_ws()
                ws.send(msg)
            logging.info(f"TX: {text[:120]}")
            last_text = text
except KeyboardInterrupt:
    pass
finally:
    stop.set()
    logging.info("Shutting down sender.")
