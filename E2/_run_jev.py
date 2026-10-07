"""Run ONLY the JEV (LLM-judge) + embedding-judge evaluation and merge
into the existing model_compare outputs (results.csv/json, probs.pkl).

While the prediction pipeline is being refined we evaluate on 10% of
each set only.  Requires JEV_API_KEY (see E2/jev.py for JEV_* env vars).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jev
import model_compare as mc

OUT = mc.OUT
FRAC = 0.10          # 10% of each set during pipeline refinement
CAP = 300            # absolute cap per set


def sampled_sets(dfs: dict) -> dict:
    """same eval frames as the in-script jev block, 10% sampled."""
    sets = {"val": dfs["val"], "testml": dfs["testml"]}
    cal = dfs.get("calib")
    if cal is not None:
        ids = list(dict.fromkeys(cal.id))[:10]
        sets["cal_sup"] = cal[cal.id.isin(ids)]
    val = dfs.get("val")
    if val is not None:
        e = val[val.label == 0].id.unique()[:5]
        o = val[val.label == 1].id.unique()[:5]
        sets["cal_unsup"] = val[val.id.isin(list(e) + list(o))]
    return {s: df.sample(min(CAP, max(1, int(len(df) * FRAC))),
                         random_state=0)
            for s, df in sets.items() if df is not None and len(df)}


def main() -> None:
    if not jev.configured():
        raise SystemExit("JEV_API_KEY not set")
    t0 = time.time()
    dfs = mc.load()
    tr = dfs["train"]
    cols = (jev.pick_stable_cols(tr, dfs["calib"])
            if dfs.get("calib") is not None and len(dfs["calib"])
            else jev.pick_cols(tr))
    print(f"jev cols ({len(cols)}): {cols}", flush=True)

    probs_p = OUT / "probs.pkl"
    probs = pickle.loads(probs_p.read_bytes()) if probs_p.exists() else {}
    ssets = sampled_sets(dfs)
    jres, eres = {}, {}
    for s, df in ssets.items():
        y = df.label.values.astype(int)
        pr = jev.jev_predict(df, tr, cols=cols, delay=0.05)
        jres[s] = {k: v for k, v in
                   mc.metrics(y, (pr > .5).astype(int), pr).items()}
        probs.setdefault("jev", {})[s] = pr
        print(f"jev/{s}: bal={jres[s]['bal']:.3f} acc={jres[s]['acc']:.3f} "
              f"auc={jres[s]['auc']:.3f} n={jres[s]['n']}", flush=True)
        # embedding-model judge: kNN in text-embedding space
        pe = jev.emb_predict(df, tr, cols=cols)
        eres[s] = {k: v for k, v in
                   mc.metrics(y, (pe > .5).astype(int), pe).items()}
        probs.setdefault("jev_emb", {})[s] = pe
        print(f"jev_emb/{s}: bal={eres[s]['bal']:.3f} "
              f"acc={eres[s]['acc']:.3f} auc={eres[s]['auc']:.3f}",
              flush=True)

    # merge into results.csv / results.json / probs.pkl
    res_csv = OUT / "results.csv"
    if res_csv.exists():
        flat = pd.read_csv(res_csv)
        # archive previous iteration as _vN before overwriting
        if flat.model.isin(["jev"]).any():
            i = 1
            while flat.model.isin([f"jev_v{i}"]).any():
                i += 1
            flat.loc[flat.model == "jev", "model"] = f"jev_v{i}"
            flat.loc[flat.model == "jev_emb", "model"] = f"jev_emb_v{i}"
        flat = flat[~flat.model.isin(["jev", "jev_emb"])]
        flat = pd.concat([flat, pd.DataFrame(
            [{"model": "jev", "set": s, **m} for s, m in jres.items()]
            + [{"model": "jev_emb", "set": s, **m}
               for s, m in eres.items()])], ignore_index=True)
        flat.to_csv(res_csv, index=False)

    res_json = OUT / "results.json"
    res = json.loads(res_json.read_text()) if res_json.exists() else {}
    if "jev" in res:
        i = 1
        while f"jev_v{i}" in res:
            i += 1
        res[f"jev_v{i}"] = res["jev"]
        res[f"jev_emb_v{i}"] = res.get("jev_emb", {})
    res["jev"] = jres
    res["jev"]["note"] = (
        f"model={jev.os.environ.get('JEV_MODEL', 'gpt-4o-mini')} "
        f"cols={len(cols)} sampled {FRAC:.0%}/set")
    res["jev_emb"] = eres
    res["jev_emb"]["note"] = (
        f"emb={jev.os.environ.get('JEV_EMB_MODEL', 'text-embedding-3-small')} "
        f"kNN k=15 n_train=400 cols={len(cols)}")
    res_json.write_text(json.dumps(res, indent=1, default=str))

    probs_p.write_bytes(pickle.dumps(probs))
    print(f"done {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
