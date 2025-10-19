# sender/transcriber_service.py
# Service-friendly: loopback -> faster-whisper (CPU int8) -> WS
# - Constant 1s reconnect retry (no exponential backoff)
# - Recovers from WS drops, mic errors, and unexpected exceptions
# - Safe ping keepalive, drop-old queue to stay realtime

import os, time, json, warnings, argparse, threading, queue, datetime as dt, logging, socket, ssl, sys
import numpy as np
import soundcard as sc
from websocket import create_connection, WebSocketConnectionClosedException, WebSocketTimeoutException
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", message="data discontinuity in recording")

# NumPy>=2 safety shim
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

ap = argparse.ArgumentParser(description="Service-friendly loopback -> faster-whisper -> WS")
ap.add_argument("--url")
ap.add_argument("--device")
ap.add_argument("--loopback", action="store_true")
ap.add_argument("--chunk", type=float)                 # 0.75–1.5 s
ap.add_argument("--rate", type=int)                    # 16000
ap.add_argument("--model")                             # tiny / tiny.en / small
ap.add_argument("--silence_rms", type=float)           # 0.006–0.012; 0 disables
ap.add_argument("--token")                             # Authorization: Bearer <token>
ap.add_argument("--retry_seconds", type=int, default=1)
ap.add_argument("--debug", action="store_true")
args = ap.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

WS_URL      = args.url    or env("WS_URL", "ws://192.168.0.110:8765/stream")
MODEL_NAME  = args.model  or env("WHISPER_MODEL", "tiny")
SAMPLE_RATE = args.rate   or int(env("SAMPLE_RATE", "16000"))
CHUNK_SEC   = args.chunk  or float(env("CHUNK_SEC", "1.0"))
DEV_NAME    = args.device or env("LOOPBACK_DEVICE")
USE_LOOP    = args.loopback or (env("LOOPBACK", "true").lower() == "true")
CHANNELS    = int(env("CHANNELS", "1"))
Q_MAX       = int(env("QUEUE_MAX", "6"))
PING_SECS   = int(env("PING_SECS", "15"))
SILENCE_RMS = args.silence_rms if args.silence_rms is not None else float(env("SILENCE_RMS", "0.0"))
AUTH_TOKEN  = args.token or env("AUTH_TOKEN", "")
INSECURE    = env("WS_INSECURE", "0") == "1"
RETRY_SECS  = max(1, int(args.retry_seconds))

logging.info(f"cfg ws={WS_URL} model={MODEL_NAME} rate={SAMPLE_RATE}Hz chunk={CHUNK_SEC}s ch={CHANNELS} loopback={USE_LOOP} silence_rms={SILENCE_RMS} retry={RETRY_SECS}s")

# Load model (CPU int8)
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
logging.info("faster-whisper ready (CPU int8)")

def pick_mic():
    if DEV_NAME:
        return sc.get_microphone(DEV_NAME, include_loopback=USE_LOOP)
    spk = sc.default_speaker()
    logging.info(f"default speaker: {spk.name}")
    return sc.get_microphone(id=str(spk.name), include_loopback=True)

def connect_ws():
    """Always retry after RETRY_SECS; no exponential backoff."""
    hdr = []
    if AUTH_TOKEN:
        hdr.append(f"Authorization: Bearer {AUTH_TOKEN}")

    sslopt = None
    if WS_URL.startswith("wss://"):
        sslopt = {"cert_reqs": ssl.CERT_NONE, "check_hostname": False} if INSECURE else {"cert_reqs": ssl.CERT_REQUIRED}

    while True:
        try:
            ws = create_connection(
                WS_URL,
                timeout=5,
                header=hdr if hdr else None,
                sslopt=sslopt,
                sockopt=((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                         (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)),
            )
            ws.settimeout(30)
            ws.send(json.dumps({"type":"hello","ts":dt.datetime.utcnow().isoformat()+"Z","note":"cpu-int8 sender"}))
            logging.info("ws connected")
            return ws
        except Exception as e:
            logging.warning(f"ws connect failed: {e}; retrying in {RETRY_SECS}s")
            time.sleep(RETRY_SECS)

def is_silent(x: np.ndarray, thr: float) -> bool:
    if thr <= 0: return False
    return float(np.sqrt(np.mean(x*x))) < thr

def transcribe(x: np.ndarray) -> str:
    segs, _ = model.transcribe(
        x,
        language=None,                  # autodetect
        task="transcribe",
        beam_size=1, best_of=1,
        vad_filter=True, no_speech_threshold=0.6,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    return "".join(s.text for s in segs).strip()

def run_once():
    ws = connect_ws()
    mic = pick_mic()
    logging.info(f"device: {mic.name}")

    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=Q_MAX)
    stop = threading.Event()
    next_ping = time.time() + PING_SECS

    def capture_loop():
        frames = int(SAMPLE_RATE * CHUNK_SEC)
        try:
            with mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as rec:
                logging.info("capture started")
                while not stop.is_set():
                    try:
                        x = rec.record(numframes=frames).astype("float32").reshape(-1)
                        if q.full():
                            try: q.get_nowait()  # drop oldest to stay real-time
                            except queue.Empty: pass
                        q.put_nowait(x)
                    except Exception:
                        time.sleep(0.02)
        except Exception as e:
            logging.warning(f"capture init/loop error: {e}")

    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    last_text = ""
    try:
        while True:
            if time.time() >= next_ping:
                try:
                    ws.ping()
                    ws.send(json.dumps({"type":"ping","ts":dt.datetime.utcnow().isoformat()+"Z"}))
                except Exception:
                    # force reconnect outer loop
                    raise
                next_ping = time.time() + PING_SECS

            try:
                x = q.get(timeout=1)
            except queue.Empty:
                continue

            if is_silent(x, SILENCE_RMS):
                continue

            txt = transcribe(x)
            if txt and txt != last_text:
                msg = json.dumps({"type":"transcript","ts":dt.datetime.utcnow().isoformat()+"Z","text":txt}, ensure_ascii=False)
                try:
                    ws.send(msg)
                except (WebSocketConnectionClosedException, WebSocketTimeoutException, OSError):
                    raise
                last_text = txt
                logging.info(f"tx: {txt[:120]}")
    finally:
        # ensure we exit cleanly so outer supervisor can restart
        stop.set()
        try:
            ws.close()
        except Exception:
            pass

# --- supervisor loop: always retry every RETRY_SECS ---
while True:
    try:
        run_once()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logging.warning(f"run_once crashed: {e}; restarting in {RETRY_SECS}s")
        time.sleep(RETRY_SECS)
