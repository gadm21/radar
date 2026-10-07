"""JEV — LLM-judge occupancy predictor (descriptor-level, not raw data).

For each W-second window the judge receives a compact JSON bundle of
physical descriptors (PCA-window stats, cluster distances, FFT bands)
plus TRAIN class statistics (mean/std per descriptor per class) and is
asked to score P(occupied).  Raw CSI/radar never leaves the machine.

Runs ONLY when env config is present — model_compare.py skips the jev
class otherwise:

    JEV_API_KEY   required
    JEV_BASE_URL  default https://api.openai.com/v1
    JEV_MODEL     default gpt-4o-mini
    JEV_PROVIDER  'openai' (chat/completions) or 'anthropic' (messages)

Per-window token cost is kept low by rounding to 3 decimals and by
sending only the top-K descriptors (ranked by |cohen d| on train).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Iterable

import numpy as np
import pandas as pd

TOPK = 14          # descriptors per prompt
TIMEOUT = 45
RETRIES = 3

SYS = """You are a strict binary occupancy judge for a radar+CSI room
sensor.  You receive per-window physical descriptors for the CURRENT
window and reference statistics (mean, std) for the two classes
'empty' and 'occupied' measured on the training site.  Reply with a
JSON object only: {"occupied": <0..1>, "reason": <short string>}.
Score = probability the room is occupied during this window.
Descriptors with 'std' are window variances of PCA-compressed streams;
'traj' stats summarise the window's trajectory in PCA space; 'fft_'
columns are spectral band powers; 'clu_' are distances to unsupervised
clusters.  Decide by comparing the window's descriptors against BOTH
class reference distributions."""


def class_stats(df_train: pd.DataFrame, cols: list[str]) -> dict:
    s = {}
    for lab, name in ((0, "empty"), (1, "occupied")):
        d = df_train[df_train.label == lab]
        s[name] = {c: {"mean": round(float(d[c].mean()), 4),
                       "std": round(float(d[c].std()), 4)}
                   for c in cols}
    return s


def pick_cols(df_train: pd.DataFrame, k: int = TOPK) -> list[str]:
    """top-k descriptors by |cohen d| on train."""
    y = df_train.label.values
    cols = [c for c in df_train.columns
            if c not in ("id", "s0", "label", "session", "cov",
                         "clu_kmlabel") and df_train[c].std() > 1e-9]
    ds = []
    for c in cols:
        v = df_train[c].values
        a, b = v[y == 0], v[y == 1]
        ds.append((c, abs((a.mean() - b.mean()) /
                          np.sqrt((a.std() ** 2 + b.std() ** 2) / 2
                                  + 1e-12))))
    ds.sort(key=lambda t: -t[1])
    return [c for c, _ in ds[:k]]


def _post(url: str, headers: dict, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers,
        method="POST")
    for i in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** i)
                continue
            raise
        except Exception:
            if i == RETRIES - 1:
                raise
            time.sleep(2 ** i)
    return {}


def _call_openai(prompt: str, cfg: dict) -> str:
    out = _post(
        f"{cfg['base']}/chat/completions",
        {"Authorization": f"Bearer {cfg['key']}",
         "Content-Type": "application/json"},
        {"model": cfg["model"], "temperature": 0,
         "messages": [{"role": "system", "content": SYS},
                      {"role": "user", "content": prompt}],
         "response_format": {"type": "json_object"}})
    return out["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str, cfg: dict) -> str:
    out = _post(
        f"{cfg['base']}/messages",
        {"x-api-key": cfg["key"], "anthropic-version": "2023-06-01",
         "Content-Type": "application/json"},
        {"model": cfg["model"], "max_tokens": 200, "temperature": 0,
         "system": SYS,
         "messages": [{"role": "user", "content": prompt}]})
    return "".join(b.get("text", "") for b in out.get("content", []))


def _parse(txt: str) -> float:
    try:
        j = json.loads(txt)
        return float(np.clip(float(j["occupied"]), 0, 1))
    except Exception:
        import re
        m = re.search(r'"occupied"\s*:\s*([0-9.]+)', txt)
        return float(np.clip(float(m.group(1)), 0, 1)) if m else 0.5


def make_prompt(row: pd.Series, cols: list[str],
                stats: dict) -> str:
    win = {c: round(float(row[c]), 4) for c in cols}
    return ("Window descriptors: " + json.dumps(win)
            + "\nClass reference stats (train): "
            + json.dumps(stats)
            + "\nReturn JSON only.")


def jev_predict(rows: Iterable[pd.Series],
                df_train: pd.DataFrame,
                cols: list[str] | None = None,
                delay: float = 0.0) -> np.ndarray:
    """P(occupied) per window.  Raises RuntimeError if unconfigured."""
    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise RuntimeError("JEV_API_KEY not set — jev disabled")
    cfg = {
        "key": key,
        "base": os.environ.get("JEV_BASE_URL",
                               "https://api.openai.com/v1"),
        "model": os.environ.get("JEV_MODEL", "gpt-4o-mini"),
        "provider": os.environ.get("JEV_PROVIDER", "openai"),
    }
    cols = cols or pick_cols(df_train)
    stats = class_stats(df_train, cols)
    call = _call_anthropic if cfg["provider"] == "anthropic" \
        else _call_openai
    out = np.empty(len(list(rows)), np.float32)
    rows = list(rows)
    for i, row in enumerate(rows):
        p = make_prompt(row, cols, stats)
        out[i] = _parse(call(p, cfg))
        if delay:
            time.sleep(delay)
        if (i + 1) % 25 == 0:
            print(f"    jev {i + 1}/{len(rows)}", flush=True)
    return out


def configured() -> bool:
    return bool(os.environ.get("JEV_API_KEY"))
