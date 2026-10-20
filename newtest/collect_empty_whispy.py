"""Collect a 1-hour capture labeled 'empty' from chen via the whispy
library (whispy.lan -> node tail API), writing the standard capture
layout: <sensor>.jsonl + manifest.json.

Usage: python collect_empty_whispy.py <out_dir> [seconds] [label]
"""
import json
import sys
import threading
import time
from pathlib import Path

import whispy

ROOT = Path(__file__).resolve().parent
TOK = (ROOT / ".chen_token").read_text().strip()
HOST = "10.0.0.24"
DEVICE_ID = "fa81cdda-58b2-5caa-b31b-e6ff3dce19ec"
SENSORS = ["radar-a316", "csi-c50e"]

counts = {s: 0 for s in SENSORS}


def pump(dev, sid, fh, stop):
    handle = dev.sensor(sid)
    n = 0
    for s in handle.stream():
        if stop.is_set():
            break
        rec = {
            "device_id": DEVICE_ID,
            "sensor_id": sid,
            "sensor_type": s.payload.get("encoding")
                          if isinstance(s.payload, dict) else None
                          or sid.split("-")[0],
            "timestamp": s.timestamp,
            "sequence": s.sequence,
            "payload_type": s.payload_type,
            "payload": s.payload,
        }
        fh.write(json.dumps(rec) + "\n")
        n += 1
        if n % 500 == 0:
            fh.flush()
        counts[sid] = n
    fh.flush()


def main():
    out_dir = Path(sys.argv[1])
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
    label = sys.argv[3] if len(sys.argv) > 3 else "empty"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = whispy.lan(HOST, token=TOK, timeout=15)
    t0 = time.time()
    stop = threading.Event()
    threads = []
    files = {}
    for sid in SENSORS:
        fh = open(out_dir / f"{sid}.jsonl", "w", encoding="utf-8")
        files[sid] = fh
        t = threading.Thread(target=pump, args=(dev, sid, fh, stop),
                             daemon=True)
        t.start()
        threads.append(t)
    print(f"collecting -> {out_dir} for {dur:.0f}s", flush=True)
    while time.time() - t0 < dur:
        time.sleep(30)
        print(f"  {time.time()-t0:6.0f}s  " +
              "  ".join(f"{s}:{counts[s]}" for s in SENSORS), flush=True)
    stop.set()
    for t in threads:
        t.join(timeout=10)
    for fh in files.values():
        fh.close()
    manifest = {
        "id": out_dir.name,
        "device_id": DEVICE_ID,
        "started_at": t0,
        "stopped_at": time.time(),
        "duration_s": time.time() - t0,
        "state": "stopped",
        "sensors": SENSORS,
        "sample_counts": dict(counts),
        "labels": [{"label": label, "source": "manual",
                    "start": t0, "end": time.time()}],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("done:", counts, flush=True)


if __name__ == "__main__":
    main()
