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
sensor.  All descriptor values are PERCENTILE RANKS (0-1) computed
within their own capture session: 0 = lowest window in that session,
0.5 = session median, 1 = highest.  You receive the current window's
percentile profile and the mean percentile profiles of the classes
'empty' and 'occupied' measured on the training site.  Reply with a
JSON object only: {"occupied": <0..1>, "reason": <short string>}.
Score = probability the room is occupied during this window.
Physical priors: occupied rooms show HIGHER percentiles on activity /
range-spread / variance descriptors (r_, ra_, rd_, csi_rv, pcv, traj)
and lower percentiles on quiet-baseline descriptors (fft low band,
c_min).  Decide by comparing the window's percentile profile against
BOTH class percentile profiles."""


def _pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """within-session percentile rank per column (0-1)."""
    return df[cols].rank(pct=True)


def class_stats(df_train: pd.DataFrame, cols: list[str]) -> dict:
    """mean/std of within-train percentiles per class."""
    pt = _pct(df_train, cols)
    s = {}
    for lab, name in ((0, "empty"), (1, "occupied")):
        d = pt[df_train.label.values == lab]
        s[name] = {c: {"pct_mean": round(float(d[c].mean()), 3),
                       "pct_std": round(float(d[c].std()), 3)}
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


def make_prompt(row_pct: pd.Series, cols: list[str],
                stats: dict) -> str:
    win = {c: round(float(row_pct[c]), 2) for c in cols}
    return ("Window descriptor percentiles (rank within this "
            "session): " + json.dumps(win)
            + "\nClass percentile profiles (train): "
            + json.dumps(stats)
            + "\nReturn JSON only.")


def pick_stable_cols(df_train: pd.DataFrame, df_ref: pd.DataFrame,
                     k: int = 10) -> list[str]:
    """top-k descriptors by |cohen d| on train, restricted to those
    whose class-separation sign is consistent between train and a
    reference session (cal_sup) — direction-stable across domains."""
    def signs(df):
        y = df.label.values
        d0 = df[y == 0].mean(numeric_only=True)
        d1 = df[y == 1].mean(numeric_only=True)
        return np.sign(d1 - d0)
    s_tr, s_rf = signs(df_train), signs(df_ref)
    cand = pick_cols(df_train, k * 3)
    stable = [c for c in cand
              if c in s_rf.index and s_tr[c] == s_rf[c]
              and s_tr[c] != 0][:k]
    if len(stable) < 5:                      # fallback: plain pick
        stable = pick_cols(df_train, k)
    return stable


def jev_predict(df_eval: pd.DataFrame,
                df_train: pd.DataFrame,
                cols: list[str] | None = None,
                delay: float = 0.0) -> np.ndarray:
    """P(occupied) per eval window.  Raises RuntimeError if
    unconfigured."""
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
    pe = _pct(df_eval, cols)
    call = _call_anthropic if cfg["provider"] == "anthropic" \
        else _call_openai
    out = np.empty(len(pe), np.float32)
    for i, (_, row) in enumerate(pe.iterrows()):
        p = make_prompt(row, cols, stats)
        out[i] = _parse(call(p, cfg))
        if delay:
            time.sleep(delay)
        if (i + 1) % 25 == 0:
            print(f"    jev {i + 1}/{len(pe)}", flush=True)
    return out


def configured() -> bool:
    return bool(os.environ.get("JEV_API_KEY"))


# ---------------- embedding-model judge (text-embedding + kNN) ----------------

def _embed(texts: list[str], cfg: dict,
           batch: int = 512) -> np.ndarray:
    """OpenAI /embeddings; returns (n, d) float32."""
    model = os.environ.get("JEV_EMB_MODEL", "text-embedding-3-small")
    vecs = []
    for i in range(0, len(texts), batch):
        out = _post(
            f"{cfg['base']}/embeddings",
            {"Authorization": f"Bearer {cfg['key']}",
             "Content-Type": "application/json"},
            {"model": model, "input": texts[i:i + batch]})
        vecs.extend(d["embedding"] for d in out["data"])
    return np.asarray(vecs, np.float32)


def emb_predict(df_eval: pd.DataFrame, df_train: pd.DataFrame,
                cols: list[str] | None = None, n_train: int = 400,
                k: int = 15) -> np.ndarray:
    """Per-descriptor band prototypes: each eval percentile is banded
    into a phrase 'descriptor is <band>'; band phrases are embedded,
    per-class per-descriptor prototype embeddings are built from
    labeled train windows, and P(occupied) = softmax over the
    |cohen-d|-weighted similarity margin."""
    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise RuntimeError("JEV_API_KEY not set — jev disabled")
    cfg = {"key": key,
           "base": os.environ.get("JEV_BASE_URL",
                                  "https://api.openai.com/v1")}
    cols = cols or pick_cols(df_train)
    tr = df_train.sample(min(n_train, len(df_train)), random_state=0)
    pt = _pct(df_train, cols).loc[tr.index]
    pe = _pct(df_eval, cols)
    ytr = tr.label.values.astype(int)

    bands = ("very low", "low", "mid", "high", "very high")
    phrases = [f"sensor descriptor {c} is {b}"
               for c in cols for b in bands]
    E = _embed(phrases, cfg)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    B = E.reshape(len(cols), len(bands), -1)           # (d, 5, emb)

    # per-descriptor class prototypes = mean band embedding per class
    proto = np.zeros((len(cols), 2, B.shape[2]), np.float32)
    for j, c in enumerate(cols):
        bidx = np.clip((pt[c].values * 5).astype(int), 0, 4)
        for lab in (0, 1):
            sel = bidx[ytr == lab]
            proto[j, lab] = (B[j, sel].mean(0)
                             if len(sel) else np.zeros(B.shape[2]))
        proto[j] /= np.linalg.norm(proto[j], axis=1,
                                   keepdims=True) + 1e-12

    # cohen-d weights from train percentiles
    w = np.array([abs((pt.loc[tr.index[ytr == 1], c].mean()
                       - pt.loc[tr.index[ytr == 0], c].mean()) + 1e-3)
                  for c in cols], np.float32)
    w /= w.sum()

    bidx_e = np.clip((pe[cols].values * 5).astype(int), 0, 4)
    score = np.zeros((len(pe), 2), np.float32)
    for j in range(len(cols)):
        ve = B[j][bidx_e[:, j]]                      # (n, emb)
        score[:, 0] += w[j] * (ve * proto[j, 0]).sum(1)
        score[:, 1] += w[j] * (ve * proto[j, 1]).sum(1)
    m = score - score.mean(0)
    return (1 / (1 + np.exp(-8 * (m[:, 1] - m[:, 0])))).astype(np.float32)
