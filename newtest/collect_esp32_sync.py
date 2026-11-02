"""Synchronized multi-ESP32 whispy collection.

Streams every whispy CSI sensor visible on this host (COM-attached
thoth_csi boards) in parallel for `duration_s`, writes one JSONL per
sensor, then reports per-board sample counts, rates and stream timing.
"""

import json
import sys
import threading
import time
from pathlib import Path

import whispy

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
OUT = Path(__file__).parent / "esp32_sync"


def collect(handle, sensor_id, stop_at, stats):
    path = OUT / f"{sensor_id}.jsonl"
    n = 0
    first = last = None
    intervals = []
    prev = None
    with path.open("w", encoding="utf-8") as f:
        for sample in handle.stream():
            now = time.time()
            if now >= stop_at:
                break
            rec = {"t": now, "sensor": sensor_id}
            try:
                rec["ts"] = getattr(sample, "timestamp", None)
                rec["payload"] = getattr(sample, "payload", sample)
            except Exception:
                rec["payload"] = str(sample)
            f.write(json.dumps(rec, default=str) + "\n")
            if prev is not None:
                intervals.append(now - prev)
            prev = now
            if first is None:
                first = now
            last = now
            n += 1
    stats[sensor_id] = {
        "samples": n,
        "first": first,
        "last": last,
        "span_s": (last - first) if first and last else 0,
        "hz": n / (last - first) if first and last and last > first else 0,
        "max_gap_s": max(intervals) if intervals else 0,
        "file": str(path),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = whispy.local()
    csi = [s for s in dev.sensors() if s.type == "wifi_csi"]
    print(f"csi sensors: {[s.id for s in csi]}", flush=True)

    handles = {}
    for s in csi:
        try:
            handles[s.id] = dev.sensor(s.id)
        except Exception as e:
            print(f"open {s.id} failed: {e}", flush=True)

    stop_at = time.time() + DURATION
    stats = {}
    threads = [
        threading.Thread(target=collect,
                         args=(h, sid, stop_at, stats), daemon=True)
        for sid, h in handles.items()
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    # progress line every 60 s
    while time.time() < stop_at:
        time.sleep(60)
        elapsed = time.time() - t0
        print(f"[{elapsed:5.0f}s] " + " ".join(
            f"{k}={v.get('samples', 0)}" for k, v in stats.items()),
            flush=True)
    for t in threads:
        t.join(timeout=10)

    report = {"duration_s": DURATION,
              "started": t0, "ended": time.time(),
              "sensors": stats}
    (OUT / "sync_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
