ANTOJA Audio → Whisper → WebSocket (with explicit IP roles)

Roles
- AUDIO CAPTURE (Sender, Machine A): your computer with headphones. Runs sender/transcriber.py.
- MESSAGE RECEIVER (WebSocket, Machine B): server that receives transcripts. Runs receiver/ws_server.py.

IP Settings
- Put the MESSAGE RECEIVER IP (Machine B) into sender/.env as WS_URL.
  Example: WS_URL=ws://192.168.0.109:8765  ← 192.168.0.109 is the MESSAGE RECEIVER IP
- You do NOT need the AUDIO CAPTURE IP in configs; it's shown only in logs.

Quick Start
1) Install deps on both machines: Python 3.10+, ffmpeg, pip install -r requirements.txt (+ torch per OS).
2) Machine B (MESSAGE RECEIVER): python receiver/ws_server.py
3) Machine A (AUDIO CAPTURE):
   - Copy sender/.env.sample to sender/.env
   - Edit WS_URL to point to the MESSAGE RECEIVER IP (Machine B)
   - Optional: set LOOPBACK_DEVICE if needed
   - Run: python sender/transcriber.py
