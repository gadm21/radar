"""Wait for chen to reappear online at Brain, then run a 30-min
node-side capture (radar-a316 + csi-c50e) via the /v1/nodes relay.

The capture is written locally on the Pi (~/.thoth/captures/<id>/),
so no LAN path to this machine is needed while it records.

Usage: python poll_and_capture.py [poll_s] [capture_s]
"""
import base64
import json
import sys
import time
import urllib.request

BRAIN = "https://web-production-d7d37.up.railway.app"
HUB = "https://hub.thothcraft.com"
DEVICE = "fa81cdda-58b2-5caa-b31b-e6ff3dce19ec"
POLL_S = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
CAPTURE_S = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0

_token = {"v": None, "t": 0.0}


def token():
    if _token["v"] and time.time() - _token["t"] < 3000:
        return _token["v"]
    req = urllib.request.Request(
        f"{HUB}/api/proxy/token",
        data=json.dumps({"username": "gadgad", "password": "password"})
        .encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        _token["v"] = json.loads(r.read())["access_token"]
    _token["t"] = time.time()
    return _token["v"]


def relay(method, path, body=None, timeout=45):
    req = urllib.request.Request(
        f"{BRAIN}/v1/nodes/{DEVICE}/api",
        data=json.dumps({"method": method, "path": path, "body": body})
        .encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token()}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        try:
            return None, json.loads(e.read())
        except Exception:
            return None, {"detail": f"HTTP {e.code}"}
    except Exception as e:
        return None, {"detail": str(e)}


def main():
    print(f"watching {DEVICE} for ws tunnel ...", flush=True)
    while True:
        body, err = relay("GET", "/api/status")
        if body is not None:
            print(f"node online: {json.dumps(body)[:200]}", flush=True)
            break
        print(f"  {time.strftime('%H:%M:%S')} offline "
              f"({(err or {}).get('detail')})", flush=True)
        time.sleep(POLL_S)

    body, err = relay("POST", "/api/captures/start",
                      {"sensors": ["radar-a316", "csi-c50e"]})
    print("start:", body or err, flush=True)
    cap_id = (body or {}).get("capture_id") or (body or {}).get("id")
    if not cap_id:
        print("could not start capture", flush=True)
        sys.exit(1)
    relay("POST", f"/api/captures/{cap_id}/label",
          {"label": "unlabeled", "source": "manual"})
    print(f"capture {cap_id} running {CAPTURE_S:.0f}s", flush=True)
    time.sleep(CAPTURE_S)
    body, err = relay("POST", "/api/captures/stop", {"capture_id": cap_id})
    print("stop:", body or err, flush=True)


if __name__ == "__main__":
    main()
