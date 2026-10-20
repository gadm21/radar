"""Deep-learning branch of the occupancy pipeline.

Encoder/decoder design (per user request): one encoder per modality
(radar sequence over frames, CSI sequence over the resampled amplitude),
producing a single embedded vector; single-modality models attach the
decoder classifier directly, the dual-modality model concatenates both
embeddings. Encoders come in 1-D CNN and LSTM flavors.

Inputs per window:
  * CSI   -> log1p amplitude mean-pooled to 20 Hz, shape (52, 20*W)
  * radar -> per-frame 12x12 log-power RD maps flattened (144), linearly
    embedded per frame, resampled to 5 fps, shape (144, 5*W)
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import occupancy_pipeline as P

SEED = P.SEED
CSI_DS = 20          # CSI DL input rate (Hz) after mean-pooling
RAD_DS = 5           # radar DL input rate (frames/s) after resampling
EMB = 64
BATCH = 256
PATIENCE = 8


# ---------------------------------------------------------------------------
# window tensor assembly
# ---------------------------------------------------------------------------
def _window_tensors(z, W, npz_key_idx=None):
    """Return (csi[nw_valid,52,20W], rad[nw_valid,144,5W], cok[nw],
    rok[nw]) for one minute dict z."""
    nw = 60 // W
    Tc, Tr = CSI_DS * W, RAD_DS * W
    amp = np.asarray(z['amp60'], np.float32)          # (60,100,52)
    sec_valid = np.asarray(z['sec_valid'])
    rd12 = np.asarray(z['rd12'], np.float32)          # (n,12,12)
    rts = np.asarray(z['rts'])

    cok = np.array([sec_valid[w * W:(w + 1) * W].mean() >= P.MIN_COV
                    for w in range(nw)])
    csi = np.zeros((nw, Tc, P.N_SUB), np.float32)
    if cok.any():
        a = P._ffill_rows(amp.reshape(60 * P.SEC_HZ, P.N_SUB))
        if a is None:
            a = np.zeros((60 * P.SEC_HZ, P.N_SUB), np.float32)
        a = np.log1p(a)
        a = a.reshape(60 * CSI_DS, P.SEC_HZ // CSI_DS, P.N_SUB).mean(axis=1)
        a = a.reshape(nw, Tc, P.N_SUB)
        csi[cok] = a[cok]

    rok = np.zeros(nw, bool)
    rad = np.zeros((nw, Tr, 144), np.float32)
    if len(rts):
        flat = rd12.reshape(len(rd12), -1)
        for w in range(nw):
            sel = np.flatnonzero((rts >= w * W) & (rts < (w + 1) * W))
            if len(sel) >= P.MIN_RADAR_FRAMES[W]:
                rok[w] = True
                idx = np.linspace(0, len(sel) - 1, Tr).round().astype(int)
                rad[w] = flat[sel[idx]]
    return csi.transpose(0, 2, 1), rad.transpose(0, 2, 1), cok, rok


def build_split(W, which, feat, index, specs=None):
    """Build tensors + window mask for 'fit' (train+aug), 'val', 'test'.

    which == 'fit' returns augmented minutes as extra windows (aligned
    with feat rows via rec_id order)."""
    items = []
    if which == 'test':
        mins = feat[(feat.dataset == 'test')]
        ids = mins.groupby('rec_id').first().reset_index()[
            ['rec_id', 'label']]
        id_to_npz = index.set_index('rec_id').npz.to_dict()
        for _, r in ids.iterrows():
            items.append((r.rec_id, np.load(id_to_npz[r.rec_id]),
                          int(r.label)))
    else:
        is_val = which == 'val'
        sub = index[(index.dataset == 'train') & (index.is_val == is_val)
                    & index.usable]
        for _, r in sub.iterrows():
            items.append((r.rec_id, np.load(r.npz), int(r.label)))
    C_, R_, Y, Wmask, REC = [], [], [], [], []
    nw = 60 // W
    for rid, z, y in items:
        c, r_, cok, rok = _window_tensors(z, W)
        C_.append(c)
        R_.append(r_)
        Y.append(np.full(nw, y, np.float32))
        Wmask.append(np.stack([cok, rok], axis=1))
        REC.extend([rid] * nw)
    if which == 'fit' and specs:
        for sp in specs:
            z2 = P.realize_aug(sp)
            c, r_, cok, rok = _window_tensors(z2, W)
            C_.append(c)
            R_.append(r_)
            Y.append(np.zeros(nw, np.float32))
            Wmask.append(np.stack([cok, rok], axis=1))
            REC.extend([f"{sp['rec_id']}#aug{sp['aug_id'] - 1}"] * nw)
    return (np.concatenate(C_), np.concatenate(R_),
            np.concatenate(Y), np.concatenate(Wmask), np.array(REC))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, ch_in, emb=EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch_in, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(128, emb)

    def forward(self, x):
        return self.head(self.net(x).squeeze(-1))


class LSTMEncoder(nn.Module):
    def __init__(self, ch_in, emb=EMB):
        super().__init__()
        self.lstm = nn.LSTM(ch_in, 64, batch_first=True,
                            bidirectional=True)
        self.head = nn.Linear(128, emb)

    def forward(self, x):
        h, _ = self.lstm(x.transpose(1, 2))
        return self.head(h[:, -1])


class Decoder(nn.Module):
    def __init__(self, in_dim, emb=EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, emb), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(emb, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


class UniModel(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.dec = Decoder(EMB)

    def forward(self, x):
        return self.dec(self.enc(x))


class DualModel(nn.Module):
    def __init__(self, enc_csi, enc_rad):
        super().__init__()
        self.enc_csi, self.enc_rad = enc_csi, enc_rad
        self.dec = Decoder(2 * EMB)

    def forward(self, c, r):
        return self.dec(torch.cat([self.enc_csi(c), self.enc_rad(r)], 1))


def make_enc(kind, ch_in):
    return (CNNEncoder if kind == 'cnn' else LSTMEncoder)(ch_in)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _train(model, xtr, ytr, xva, yva, epochs, log):
    dev = 'cpu'
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    n = len(ytr)
    best = (0, -1, None)      # (epoch, val_f1, state)
    bad = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            ix = perm[i:i + BATCH]
            xs = [x[ix].to(dev) for x in xtr]
            opt.zero_grad()
            out = model(*xs)
            loss = lossf(out, ytr[ix].to(dev))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(*[x.to(dev) for x in xva])).cpu().numpy()
        from sklearn.metrics import f1_score
        f1 = f1_score(yva.numpy(), pv >= 0.5)
        if f1 > best[1]:
            best = (ep, f1, {k: v.clone() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    if best[2] is not None:
        model.load_state_dict(best[2])
    log['best_epoch'] = int(best[0])
    return model


def _prob(model, xs):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xs[0]), 512):
            out.append(torch.sigmoid(
                model(*[x[i:i + 512] for x in xs])).numpy())
    return np.concatenate(out)


def run(W, feat, out_dir, epochs=40, seed=SEED, probs=None):
    """Train/evaluate DL models for window size W. Returns metric rows.
    If `probs` is a dict, store minute-level test (y, p) per model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    index = pd.read_csv(out_dir / 'minutes_index.csv')
    index['is_val'] = index.is_val.astype(bool)
    index['usable'] = index.usable.astype(bool)
    clean = index[index.usable]
    specs = P.plan_augment(clean)

    # ---- assemble tensors -------------------------------------------------
    t0 = time.time()
    Ctr, Rtr, Ytr, Mtr, RECtr = build_split(W, 'fit', feat, clean, specs)
    Cva, Rva, Yva, Mva, RECva = build_split(W, 'val', feat, clean)
    Cte, Rte, Yte, Mte, RECte = build_split(W, 'test', feat, clean)
    print(f'dl W={W}s tensors built in {time.time() - t0:.0f}s: '
          f'fit={len(Ytr)} val={len(Yva)} test={len(Yte)}', flush=True)

    # balanced test minutes: same downsample as run_models
    rng = np.random.default_rng(SEED)
    lab_of = {}
    for r, y in zip(RECte, Yte):
        lab_of[r] = int(y)
    ser = pd.Series(lab_of)
    cnt = ser.value_counts()
    keep_min = cnt.min()
    keep = []
    for lab, g in ser.groupby(ser):
        keep.extend(rng.choice(g.index, size=min(keep_min, len(g)),
                               replace=False))
    keep = set(keep)
    tb = np.array([r in keep for r in RECte])

    # ---- standardization from fit statistics ------------------------------
    def _stats(X, mask):
        sel = X[mask]
        flat = sel.transpose(0, 2, 1).reshape(-1, sel.shape[1])
        mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        return mu.astype(np.float32), np.clip(sd, 1e-3, None).astype(np.float32)

    cmu, csd = _stats(Ctr, Mtr[:, 0])
    rmu, rsd = _stats(Rtr, Mtr[:, 1])
    Ctr = (Ctr - cmu[None, :, None]) / csd[None, :, None]
    Cva = (Cva - cmu[None, :, None]) / csd[None, :, None]
    Cte = (Cte - cmu[None, :, None]) / csd[None, :, None]
    Rtr = (Rtr - rmu[None, :, None]) / rsd[None, :, None]
    Rva = (Rva - rmu[None, :, None]) / rsd[None, :, None]
    Rte = (Rte - rmu[None, :, None]) / rsd[None, :, None]

    Ct_ = torch.from_numpy(Ctr); Rt_ = torch.from_numpy(Rtr)
    Cv_ = torch.from_numpy(Cva); Rv_ = torch.from_numpy(Rva)
    Ce_ = torch.from_numpy(Cte); Re_ = torch.from_numpy(Rte)
    yt_ = torch.from_numpy(Ytr); yv_ = torch.from_numpy(Yva)

    rows = []
    for enc in ('cnn', 'lstm'):
        for mode in ('csi', 'radar', 'dual'):
            name = f'{mode}-{enc}'
            mtr = (Mtr[:, 0] if mode == 'csi'
                   else Mtr[:, 1] if mode == 'radar'
                   else Mtr[:, 0] & Mtr[:, 1])
            mva = (Mva[:, 0] if mode == 'csi'
                   else Mva[:, 1] if mode == 'radar'
                   else Mva[:, 0] & Mva[:, 1])
            mte = (Mte[:, 0] if mode == 'csi'
                   else Mte[:, 1] if mode == 'radar'
                   else Mte[:, 0] & Mte[:, 1])
            if mode == 'csi':
                model = UniModel(make_enc(enc, P.N_SUB))
                xtr, xva, xte = [Ct_], [Cv_], [Ce_]
            elif mode == 'radar':
                model = UniModel(make_enc(enc, 144))
                xtr, xva, xte = [Rt_], [Rv_], [Re_]
            else:
                model = DualModel(make_enc(enc, P.N_SUB), make_enc(enc, 144))
                xtr, xva, xte = [Ct_, Rt_], [Cv_, Rv_], [Ce_, Re_]
            tr_args = [x[mtr] for x in xtr]
            va_args = [x[mva] for x in xva]
            te_args = [x[mte & tb] for x in xte]
            log = {}
            model = _train(model, tr_args, yt_[mtr], va_args, yv_[mva],
                           epochs, log)
            pv = _prob(model, va_args)
            thr = P._best_thr(yv_[mva].numpy(), pv)
            pt = _prob(model, te_args)
            y_true = Yte[mte & tb].astype(int)
            rec_true = RECte[mte & tb]
            met = P._metrics(y_true, pt, thr)
            met.update(window=W, model=name, features='dl',
                       thr=round(thr, 3), level='window')
            rows.append(met)
            dva = pd.DataFrame({'r': RECva[mva], 'y': yv_[mva].numpy(),
                                'p': pv})
            gva = dva.groupby('r').agg(y=('y', 'first'), p=('p', 'mean'))
            thr_m = P._best_thr(gva.y.values, gva.p.values)
            d = pd.DataFrame({'r': rec_true, 'y': y_true, 'p': pt})
            g = d.groupby('r').agg(y=('y', 'first'), p=('p', 'mean'))
            metm = P._metrics(g.y.values, g.p.values, thr_m)
            metm.update(window=W, model=name, features='dl',
                        thr=round(thr_m, 3), level='minute')
            rows.append(metm)
            if probs is not None:
                probs[(W, name, 'dl', seed)] = (
                    g.y.values.astype(np.int8),
                    g.p.values.astype(np.float32))
            print(f'  dl {name}: acc={met["acc"]:.3f} auc={met["auc"]:.3f} '
                  f'(ep {log.get("best_epoch")})', flush=True)
    return rows
