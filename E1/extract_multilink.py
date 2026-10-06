"""Pass-A extraction for the multilink captures (JSONL streams).

For every capture dir under multilink_train/ this produces
E1/cache/multilink/<cap>.npz with

  meta           json: label, t0, duration, per-stream counts
  csi per link   dots[ns,104] f32 (mean52|std52 amplitude per second),
                 ok[ns] bool, a90[ns] f32 (per-second q90 amplitude),
                 rv[ns,3] f32 (0.2 s rolling-var mean/q90/max),
                 tv[ns] f32 (per-second total-variation sum of mean amp),
                 dopf[ns] f32 (per-second high-band PSD fraction)
  radar          vsec[ns,4,24,24] f16 (per-second mean log1p maps),
                 rsec[ns] bool, rts[nf] f64, snr_f[nf] f32,
                 w5_*/w10_*  per-window aggregates (sum, sumsq, diffsum,
                 count) of the 4 views, for window descriptors
                 (suffix: _s _q _d _n)

Lines are parsed with regexes, not json.loads: each CSI line embeds a
128-int iq list and each radar line a 576-float xy_map we don't need,
so regex + base64 decode of the compact 'data'/'views_b64' fields is
~10x faster.
"""
import base64
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

_TS = re.compile(r'"ts":\s*"?([0-9.eE+-]+)')
_IQ = re.compile(r'"iq":\s*\[([^\]]+)\]')
_VIEWS = re.compile(r'"views_b64":\s*"([A-Za-z0-9+/=]+)"')
_SNR = re.compile(r'"snr_f":\s*([0-9.eE+-]+)')

SECS_PER_MIN = 60.0
RV_W_FRAC = 0.2          # 0.2 s rolling-variance window


# ---------------------------------------------------------------------------
# CSI link -> per-second products
# ---------------------------------------------------------------------------
def _flush_second(rows):
    """rows: list of complex64[52] for one second -> per-second dict."""
    if len(rows) < 8:
        return None
    a = np.abs(np.asarray(rows, np.complex64)).astype(np.float64)
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    dot = np.r_[mean, std]
    a90 = float(np.quantile(a, 0.9))
    # 0.2 s causal rolling variance of log1p amplitude (E2 convention)
    w = max(4, int(round(len(rows) * RV_W_FRAC)))
    rv = np.log1p(C.rollvar(np.log1p(a), w))
    # total variation of the mean-amp trace within the second
    tv = float(np.abs(np.diff(a.mean(axis=1))).sum())
    # high-band PSD fraction of the mean-amp trace (micro-Doppler proxy)
    tr = a.mean(axis=1)
    tr = tr - tr.mean()
    ps = np.abs(np.fft.rfft(tr)) ** 2
    hf = ps[len(ps) // 3:].sum() / (ps[1:].sum() + 1e-12) if len(ps) > 3 else 0.
    return {"dot": dot, "a90": a90,
            "rv": np.array([rv.mean(), np.quantile(rv, 0.9), rv.max()],
                           np.float32),
            "tv": tv, "dopf": float(hf), "n": len(rows)}


def extract_csi_link(path, t0):
    """Parse one CSI JSONL -> per-second arrays."""
    dots, ok, a90, rv, tv, dopf = [], [], [], [], [], []
    cur_sec, rows = -1, []
    for line in open(path, "r", encoding="utf-8", errors="replace"):
        m = _IQ.search(line)
        if m is None:
            continue
        t = _TS.search(line)
        if t is None:
            continue
        sec = int(float(t.group(1)) - t0)
        if sec < 0:
            continue
        if sec != cur_sec and rows:
            r = _flush_second(rows)
            if r is not None:
                dots.append(r["dot"]); a90.append(r["a90"])
                rv.append(r["rv"]); tv.append(r["tv"])
                dopf.append(r["dopf"]); ok.append(cur_sec)
            rows = []
        cur_sec = sec
        b = np.fromstring(m.group(1), sep=",").astype(np.float32)
        if b.size != 128:
            continue
        imag = b[0::2][C.CSI_SUBCARRIER_MASK]
        real = b[1::2][C.CSI_SUBCARRIER_MASK]
        rows.append((real + 1j * imag).astype(np.complex64))
    if rows:
        r = _flush_second(rows)
        if r is not None:
            dots.append(r["dot"]); a90.append(r["a90"])
            rv.append(r["rv"]); tv.append(r["tv"])
            dopf.append(r["dopf"]); ok.append(cur_sec)
    ok = np.asarray(ok, np.int64)
    nsec = int(ok.max()) + 1 if len(ok) else 0
    D = np.zeros((nsec, 104), np.float32)
    A90 = np.zeros(nsec, np.float32)
    RV = np.zeros((nsec, 3), np.float32)
    TV = np.zeros(nsec, np.float32)
    DP = np.zeros(nsec, np.float32)
    OK = np.zeros(nsec, bool)
    for i, s in enumerate(ok):
        D[s] = dots[i]; A90[s] = a90[i]; RV[s] = rv[i]
        TV[s] = tv[i]; DP[s] = dopf[i]; OK[s] = True
    return {"dots": D, "a90": A90, "rv": RV, "tv": TV, "dopf": DP,
            "ok": OK}


# ---------------------------------------------------------------------------
# radar -> per-second + per-window products
# ---------------------------------------------------------------------------
def extract_radar(path, t0):
    n = 0
    # first pass for duration not needed: second/window counts grow lazily
    vsec_s, vsec_n = {}, {}
    wag = {W: {"s": {}, "q": {}, "d": {}, "n": {}} for W in C.WIN_LIST}
    rts, snrs = [], []
    last = None
    for line in open(path, "r", encoding="utf-8", errors="replace"):
        m = _VIEWS.search(line)
        if m is None:
            continue
        t = _TS.search(line)
        if t is None:
            continue
        rel = float(t.group(1)) - t0
        if rel < 0:
            continue
        v = np.frombuffer(base64.b64decode(m.group(1)),
                          np.float16).reshape(4, 24, 24).astype(np.float32)
        s = _SNR.search(line)
        snrs.append(float(s.group(1)) if s else np.nan)
        rts.append(rel)
        sec = int(rel)
        if sec not in vsec_s:
            vsec_s[sec] = np.zeros((4, 24, 24), np.float64)
            vsec_n[sec] = 0
        vsec_s[sec] += v
        vsec_n[sec] += 1
        if last is not None:
            dv = np.abs(v - last)
        for W in C.WIN_LIST:
            w = int(rel // W)
            d = wag[W]
            if w not in d["n"]:
                d["s"][w] = np.zeros((4, 24, 24), np.float64)
                d["q"][w] = np.zeros((4, 24, 24), np.float64)
                d["d"][w] = np.zeros((4, 24, 24), np.float64)
                d["n"][w] = 0
            d["s"][w] += v
            d["q"][w] += v.astype(np.float64) ** 2
            d["n"][w] += 1
            if last is not None:
                d["d"][w] += dv
        last = v
        n += 1
    nsec = (max(vsec_s) + 1) if vsec_s else 0
    vsec = np.zeros((nsec, 4, 24, 24), np.float16)
    rsec = np.zeros(nsec, bool)
    for s in vsec_s:
        vsec[s] = (vsec_s[s] / vsec_n[s]).astype(np.float16)
        rsec[s] = True
    out = {"vsec": vsec, "rsec": rsec,
           "rts": np.asarray(rts), "snr_f": np.asarray(snrs, np.float32)}
    for W in C.WIN_LIST:
        d = wag[W]
        nw = (max(d["n"]) + 1) if d["n"] else 0
        for key, suf in (("s", "s"), ("q", "q"), ("d", "d")):
            arr = np.zeros((nw, 4, 24, 24), np.float32)
            for w, m_ in d[key].items():
                arr[w] = m_.astype(np.float32)
            out[f"w{W}_{suf}"] = arr
        cn = np.zeros(nw, np.int32)
        for w, c in d["n"].items():
            cn[w] = c
        out[f"w{W}_n"] = cn
    return out


# ---------------------------------------------------------------------------
def extract_capture(cap_dir):
    cap_dir = Path(cap_dir)
    out_path = C.ML_CACHE / f"{cap_dir.name}.npz"
    if out_path.exists():
        return cap_dir.name, "cached"
    man = C.load_manifest(cap_dir)
    t0 = float(man["started_at"])
    label = man["label"]
    files = {k: v["file"] for k, v in man["sensors"].items()}
    csi_files = [f for k, f in files.items() if "csi" in k]
    csi_files.sort()                                  # chen link first
    radar_file = files.get("radar")
    blobs = {}
    for i, f in enumerate(csi_files):
        p = cap_dir / f
        if p.exists():
            blobs[f"csi{i + 1}"] = extract_csi_link(p, t0)
    if radar_file and (cap_dir / radar_file).exists():
        blobs["radar"] = extract_radar(cap_dir / radar_file, t0)
    flat = {"meta": json.dumps({"cap": cap_dir.name, "label": label,
                                "t0": t0,
                                "duration_s": man.get("duration_s"),
                                "counts": man.get("counts", {})})}
    for link, b in blobs.items():
        for k, v in b.items():
            flat[f"{link}_{k}"] = v
    C.ML_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **flat)
    return cap_dir.name, "ok"


def main():
    caps = sorted(p for p in C.ML_DIR.iterdir() if p.is_dir())
    print(f"{len(caps)} captures", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(4, len(caps))) as pool:
        for name, res in pool.map(extract_capture, caps):
            print(f"{name}: {res} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
