# receiver_fixed.py
import os, asyncio, json
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

BIND_HOST = os.getenv("BIND_HOST", "192.168.0.110")
BIND_PORT = int(os.getenv("BIND_PORT", "8765"))

clients = set()

async def handler(ws, path=None):   # path optional => compatible with both APIs
    peer = ws.remote_address
    print(f"Client connected from {peer}")
    clients.add(ws)
    try:
        async for msg in ws:
            try:
                data = json.loads(msg)
            except Exception:
                data = {"raw": msg}
            payload = json.dumps(data, ensure_ascii=False)
            print(payload)
            # Optional broadcast (skip closed sockets)
            if clients:
                tasks = [c.send(payload) for c in list(clients) if (c is not ws and not c.closed)]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
    except (ConnectionClosedOK, ConnectionClosedError) as e:
        # Expected on network hiccups / ping timeouts / abrupt client exits
        print(f"Client {peer} closed: {e}")
    except Exception as e:
        # Unexpected error
        print(f"Handler error for {peer}: {e}")
    finally:
        clients.discard(ws)
        print(f"Client disconnected: {peer}")

async def main():
    async with serve(
        handler,
        BIND_HOST,
        BIND_PORT,
        max_size=2**23,
        compression=None,     # lower latency
        ping_interval=15,     # send pings every 15s
        ping_timeout=30,      # consider dead if no pong by 30s
        close_timeout=1,      # don't hang trying to close gracefully
        max_queue=64,
    ):
        print(f"WebSocket MESSAGE RECEIVER on ws://{BIND_HOST}:{BIND_PORT}")
        print("Point your AUDIO CAPTURE sender WS_URL to this address.")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())


