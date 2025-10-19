# sender_forwarder.py
# Forward loopback audio (16 kHz mono float32) to your WS/WSS ASR server.
# - Single WS connection (no duplicate connect)
# - Auto-reconnect on timeouts / drops
# - Optional /stream path
# - Works with Realtek loopback names containing accents

import os, argparse, warnings, json, time, socket, ssl
import numpy as np
import soundcard as sc
from websocket import create_connection, WebSocketTimeoutException, WebSocketConnectionClosedException

warnings.filterwarnings("ignore", message="data discontinuity in recording")

# NumPy>=2 safety shim for some soundcard builds
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

ap = argparse.ArgumentParser(description="Forward loopback audio to ASR server")
ap.add_argument("--url", help="ws://host:port or ws://host:port/stream")
ap.add_argument("--device", help='e.g. "Altavoces / Auriculares (Realtek Audio)"')
ap.add_argument("--loopback", action="store_true", help="use loopback on the selected device")
ap.add_argument("--chunk", type=float, help="chunk seconds (0.8–1.2 good)")
ap.add_argument("--rate", type=int, help="sample rate (16000 recommended)")
ap.add_argument("--insecure", action="store_true", help="allow self-signed when using wss://")
args = ap.parse_args()

URL   = args.url    or env("WS_URL", "ws://192.168.0.110:8765")       # add /stream if server enforces it
DEV   = args.device or env("LOOPBACK_DEVICE")
LOOP  = args.loopback or (env("LOOPBACK", "true").lower() == "true")
RATE  = args.rate   or int(env("SAMPLE_RATE", "16000"))
CHUNK = args.chunk  or float(env("CHUNK_SEC", "1.0"))

def pick_mic():
    if DEV:
        return sc.get_microphone(DEV, include_loopback=LOOP)
    spk = sc.default_speaker()
    return sc.get_microphone(id=str(spk.name), include_loopback=True)

def make_ws():
    sslopt = None
    if URL.startswith("wss://"):
        sslopt = {"cert_reqs": ssl.CERT_NONE, "check_hostname": False} if args.insecure else {"cert_reqs": ssl.CERT_REQUIRED}
    ws = create_connection(
        URL,
        timeout=5,
        sslopt=sslopt,
        sockopt=((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                 (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1))
    )
    ws.settimeout(30)  # give the server time to process GPU work
    # Tell server our rate
    try:
        ws.send(json.dumps({"type":"cfg","rate": RATE}))
    except Exception:
        pass
    return ws

mic = pick_mic()
frames = int(RATE * CHUNK)

print(f"Device: {mic.name}")
print(f"Forwarding loopback to {URL} at {RATE} Hz, chunk {CHUNK}s")

while True:
    try:
        ws = make_ws()
        with mic.recorder(samplerate=RATE, channels=1) as rec:
            while True:
                x = rec.record(numframes=frames).astype("float32").reshape(-1)
                try:
                    ws.send_binary(x.tobytes())
                except (WebSocketTimeoutException, WebSocketConnectionClosedException, OSError) as e:
                    # connection issue → break to reconnect
                    # print minimal info; server logs will show details
                    print("socket issue, reconnecting:", type(e).__name__)
                    break
    except Exception as e:
        print("connect error:", e)
    # small backoff before reconnect
    time.sleep(1)

