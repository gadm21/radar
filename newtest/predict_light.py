"""Live occupancy -> trigger light.

Streams csi-c50e + radar-a316 from chen via whispy.lan, runs the
built-in multimodal model (occ-rf-hier) on rolling 15-s windows every
5 s, and drives the room's trigger light:

    occupied -> green
    empty    -> purple
    unknown/stale -> low-brightness warm white

Usage: python predict_light.py [duration_s]
"""
import sys
import threading
import time
from collections import deque
from pathlib import Path

import whispy
from whispy.contracts import SensorSample, SensorWindow
from whispy_model_occ.hier import OccHierModel

ROOT = Path(__file__).resolve().parent
TOK = (ROOT / ".chen_token").read_text().strip()
HOST = "10.0.0.24"
LIGHT = "light-c483"          # light.gad_room_hue (color)
WIN = 15.0
STEP = 5.0
STALE_S = 10.0

GREEN = [0, 255, 0]
PURPLE = [128, 0, 128]
DIM = [40, 32, 24]

bufs = {"csi-c50e": deque(), "radar-a316": deque()}
stop = threading.Event()
latest = {"t": 0.0}


def pump(dev, sid):
    handle = dev.sensor(sid)
    try:
        for s in handle.stream():
            if stop.is_set():
                break
            q = bufs[sid]
            q.append(s)
            latest["t"] = max(latest["t"], s.timestamp)
            while q and q[0].timestamp < s.timestamp - WIN - 5:
                q.popleft()
    except Exception as exc:                    # noqa: BLE001
        print(f"  stream {sid} died: {exc}", flush=True)


def set_state(light, state):
    if state == "occupied":
        light.execute({"operation": "set_color",
                       "params": {"rgb_color": GREEN, "brightness": 200}})
    elif state == "empty":
        light.execute({"operation": "set_color",
                       "params": {"rgb_color": PURPLE, "brightness": 200}})
    else:
        light.execute({"operation": "set_color",
                       "params": {"rgb_color": DIM, "brightness": 20}})


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 2400.0
    dev = whispy.lan(HOST, token=TOK, timeout=15)
    light = dev.actuator(LIGHT)
    if light is None:
        raise SystemExit(f"no actuator {LIGHT}")
    model = OccHierModel()
    threads = [threading.Thread(target=pump, args=(dev, sid), daemon=True)
               for sid in bufs]
    for t in threads:
        t.start()
    print(f"predict->light for {dur:.0f}s on {LIGHT}", flush=True)
    t0 = time.time()
    last_state = None
    while time.time() - t0 < dur:
        time.sleep(STEP)
        now = time.time()
        if now - latest["t"] > STALE_S:
            state = "unknown"
            pred_label, conf = "unknown", 0.0
        else:
            lo, hi = latest["t"] - WIN, latest["t"]
            w = SensorWindow(start_timestamp=lo, end_timestamp=hi)
            for sid, q in bufs.items():
                w.samples[sid] = [s for s in q
                                  if lo <= s.timestamp < hi]
            pred = model.predict(w)
            pred_label, conf = pred.label, pred.confidence
            state = pred_label if pred_label in ("occupied", "empty") \
                else "unknown"
        if state != last_state:
            try:
                set_state(light, state)
                print(f"  {now-t0:6.0f}s {pred_label:9s} "
                      f"conf={conf:.3f} -> {state}", flush=True)
            except Exception as exc:            # noqa: BLE001
                print(f"  light err: {exc}", flush=True)
            last_state = state
        else:
            print(f"  {now-t0:6.0f}s {pred_label:9s} conf={conf:.3f}",
                  flush=True)
    stop.set()
    try:
        light.execute({"operation": "turn_on",
                       "params": {"brightness": 255,
                                  "rgb_color": [255, 244, 229]}})
        print("collection ended -> light on (warm white)", flush=True)
    except Exception as exc:                    # noqa: BLE001
        print(f"  light-on err: {exc}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
