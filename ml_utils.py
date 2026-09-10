import csv
import datetime
import functools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import stft
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from radar_utils import fit_temporal_length, load_radar_chunk, load_radar_tensor

CSI_SUBCARRIER_MASK = np.array(
    [False] * 6 + [True] * 26 + [False] + [True] * 26 + [False] * 5,
    dtype=bool,
)


class CSILoader:
    def __init__(self, guaranteed_sr):
        if guaranteed_sr <= 0:
            raise ValueError("guaranteed_sr must be positive")
        self.guaranteed_sr = float(guaranteed_sr)

    def load(self, filepath):
        filepath = Path(filepath)
        if filepath.name == "wifi_csi_raw.csv":
            frame = pd.read_csv(filepath, on_bad_lines="skip", low_memory=False)
            rows = frame[frame["type"].astype(str).str.startswith("CSI_DATA", na=False)]
            payloads = rows["data"].astype(str).tolist()
            timestamps = pd.to_numeric(rows["local_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
            timestamp_scale = 1_000_000.0
        else:
            frame = pd.read_csv(filepath, on_bad_lines="skip", low_memory=False)
            rows = frame[frame["raw_csi_line"].astype(str).str.startswith("CSI_DATA", na=False)]
            payloads = rows["raw_csi_line"].astype(str).tolist()
            timestamps = pd.to_numeric(rows["monotonic_ns"], errors="coerce").to_numpy(dtype=np.float64)
            timestamp_scale = 1_000_000_000.0

        real_values = []
        imag_values = []
        valid_timestamps = []
        for payload, timestamp in zip(payloads, timestamps):
            match = re.search(r"\[([^\]]+)\]", payload)
            if match is None or not np.isfinite(timestamp):
                continue
            token_strings = [t for t in match.group(1).split(",") if t.strip()]
            if len(token_strings) != 128:
                continue
            try:
                values = np.array([float(t) for t in token_strings], dtype=np.float64)
            except ValueError:
                continue
            imag_values.append(values[0::2])
            real_values.append(values[1::2])
            valid_timestamps.append(timestamp)
        if not real_values:
            raise ValueError(f"No valid 128-value CSI rows in {filepath}")

        real = np.asarray(real_values)[:, CSI_SUBCARRIER_MASK]
        imag = np.asarray(imag_values)[:, CSI_SUBCARRIER_MASK]
        timestamps = np.asarray(valid_timestamps, dtype=np.float64)
        timestamp_seconds = (timestamps - timestamps[0]) / timestamp_scale
        complex_csi = real + 1j * imag
        return self._resample(complex_csi, timestamp_seconds)

    def _resample(self, values, timestamps):
        order = np.argsort(timestamps, kind="stable")
        timestamps = timestamps[order]
        values = values[order]
        unique_times, unique_indices = np.unique(timestamps, return_index=True)
        values = values[unique_indices]
        duration = float(unique_times[-1] - unique_times[0])
        if len(unique_times) < 2 or duration <= 0:
            return values.astype(np.complex64)
        target_count = max(2, int(math.ceil(duration * self.guaranteed_sr)) + 1)
        target_times = np.arange(target_count, dtype=np.float64) / self.guaranteed_sr
        real = np.column_stack([np.interp(target_times, unique_times, values[:, index].real) for index in range(values.shape[1])])
        imag = np.column_stack([np.interp(target_times, unique_times, values[:, index].imag) for index in range(values.shape[1])])
        return (real + 1j * imag).astype(np.complex64)


class AmplitudeExtractor:
    def __call__(self, complex_csi):
        return np.abs(complex_csi).astype(np.float32)


class WindowTransformer:
    def __init__(self, window_length, stride=None):
        self.window_length = int(window_length)
        self.stride = int(stride or window_length)

    def __call__(self, values):
        count = (len(values) - self.window_length) // self.stride + 1
        if count <= 0:
            return np.empty((0, self.window_length, values.shape[1]), dtype=values.dtype)
        return np.stack([values[index:index + self.window_length] for index in range(0, count * self.stride, self.stride)])


class RollingVarianceTransformer:
    def __init__(self, windows=(10, 200, "full")):
        self.windows = windows

    def __call__(self, amplitude):
        frame = pd.DataFrame(amplitude)
        results = []
        for window in self.windows:
            size = len(amplitude) if window == "full" else min(int(window), len(amplitude))
            variance = frame.rolling(size, min_periods=1).var(ddof=0).fillna(0.0).to_numpy(dtype=np.float32)
            results.append(variance)
        return results


class STFTTransformer:
    def __init__(self, sample_rate, nperseg=64, noverlap=None):
        self.sample_rate = float(sample_rate)
        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap if noverlap is not None else nperseg // 2)

    def __call__(self, amplitude):
        segment = min(self.nperseg, len(amplitude))
        overlap = min(self.noverlap, max(0, segment - 1))
        spectra = []
        for channel in range(amplitude.shape[1]):
            _, _, transformed = stft(
                amplitude[:, channel],
                fs=self.sample_rate,
                window="hann",
                nperseg=segment,
                noverlap=overlap,
                boundary="zeros",
                padded=True,
            )
            spectra.append(np.abs(transformed).T)
        return np.stack(spectra, axis=-1).astype(np.float32)


def _resample_feature_rows(values, count):
    if len(values) == count:
        return values
    if len(values) == 1:
        return np.repeat(values, count, axis=0)
    positions = np.linspace(0, len(values) - 1, count)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    weights = (positions - lower).reshape((-1,) + (1,) * (values.ndim - 1))
    return (values[lower] * (1.0 - weights) + values[upper] * weights).astype(np.float32)


def extract_csi_features(filepath, sample_rate, stft_nperseg=64):
    complex_csi = CSILoader(sample_rate).load(filepath)
    amplitude = AmplitudeExtractor()(complex_csi)
    rolling = RollingVarianceTransformer()(amplitude)
    spectra = STFTTransformer(sample_rate, stft_nperseg)(amplitude)
    spectra = _resample_feature_rows(spectra, len(amplitude)).reshape(len(amplitude), -1)
    features = np.concatenate([amplitude, *rolling, spectra], axis=1)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features = (features - mean) / (std + 1e-6)
    return features.astype(np.float32)


def build_csi_features(filepath, sample_rate, sample_seconds, stft_nperseg=64):
    features = extract_csi_features(filepath, sample_rate, stft_nperseg)
    length = int(round(sample_rate * sample_seconds))
    windows = WindowTransformer(length)(features)
    return windows


def find_csi_file(minute_dir):
    minute_dir = Path(minute_dir)
    for name in ("wifi_csi_timestamped.csv", "wifi_csi_raw.csv"):
        candidate = minute_dir / name
        if candidate.exists() and candidate.stat().st_size > 1000:
            return candidate
    return None


def discover_minutes(dataset_root):
    records = []
    for class_dir in sorted(Path(dataset_root).iterdir()):
        if not class_dir.is_dir():
            continue
        label = 0 if class_dir.name.lower() == "empty" else 1
        for minute_dir in sorted(class_dir.iterdir()):
            if minute_dir.is_dir():
                records.append((minute_dir, label, class_dir.name))
    return records


@dataclass
class PreprocessingConfig:
    sample_seconds: float
    radar_rate: float
    csi_rate: float
    stft_nperseg: int = 64
    radar_size: int = 64
    csi_n_channels: int = 52

    def to_dict(self):
        return self.__dict__.copy()

    def csi_feature_dim(self):
        n_freq = self.stft_nperseg // 2 + 1
        return int(52 * (4 + n_freq))

    @property
    def csi_frames(self):
        return int(round(self.csi_rate * self.sample_seconds))

    @property
    def radar_frames(self):
        return int(round(self.radar_rate * self.sample_seconds))


def extract_single_csi_features(filepath, config):
    features = extract_csi_features(filepath, config.csi_rate, config.stft_nperseg)
    return fit_temporal_length(features, config.csi_frames)


def extract_radar_sample(minute_dir, config):
    radar_tensor = load_radar_tensor(minute_dir, config.radar_rate, (config.radar_size, config.radar_size))
    radar_tensor = fit_temporal_length(radar_tensor, config.radar_frames)
    return np.transpose(radar_tensor, (1, 0, 2, 3)).astype(np.float32)


def _window_radar_tensor(radar_tensor, sample_frames):
    T, C, H, W = radar_tensor.shape
    flat = radar_tensor.reshape(T, -1)
    windows = WindowTransformer(sample_frames)(flat)
    N = windows.shape[0]
    return windows.reshape(N, sample_frames, C, H, W).transpose(0, 2, 1, 3, 4)


class MinuteCache:
    def __init__(self, cfg):
        self.cfg = cfg

    @functools.lru_cache(maxsize=8)
    def _load_csi(self, minute_dir, label):
        minute_dir = Path(minute_dir)
        csi_file = find_csi_file(minute_dir)
        csi_windows = None
        if csi_file:
            csi_windows = build_csi_features(
                csi_file, self.cfg.csi_rate, self.cfg.sample_seconds, self.cfg.stft_nperseg
            )
        return csi_windows, label

    @functools.lru_cache(maxsize=4)
    def _load_both(self, minute_dir, label):
        minute_dir = Path(minute_dir)
        radar_files = sorted(minute_dir.glob("mmw_radar_raw_*.bin"))
        csi_file = find_csi_file(minute_dir)

        radar_windows = None
        csi_windows = None

        if radar_files:
            radar_tensor = load_radar_tensor(
                minute_dir, self.cfg.radar_rate, (self.cfg.radar_size, self.cfg.radar_size)
            )
            radar_windows = _window_radar_tensor(radar_tensor, self.cfg.radar_frames)

        if csi_file:
            csi_windows = build_csi_features(
                csi_file, self.cfg.csi_rate, self.cfg.sample_seconds, self.cfg.stft_nperseg
            )

        return radar_windows, csi_windows, label

    def get_csi(self, minute_dir, label):
        return self._load_csi(str(minute_dir), label)

    def get_both(self, minute_dir, label):
        return self._load_both(str(minute_dir), label)


class CSIDataset(Dataset):
    def __init__(self, root_dir, cfg, cache=None):
        self.cfg = cfg
        self.cache = cache if cache is not None else MinuteCache(cfg)
        self.records = []
        for minute_dir, label, _ in discover_minutes(root_dir):
            if find_csi_file(minute_dir) is not None:
                self.records.append((minute_dir, label))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        minute_dir, label = self.records[idx]
        csi_windows, _ = self.cache.get_csi(str(minute_dir), label)
        if csi_windows is None or len(csi_windows) == 0:
            raise RuntimeError(f"No CSI windows for {minute_dir}")
        return csi_windows, label


class RadarCSIDataset(Dataset):
    def __init__(self, root_dir, cfg, cache=None):
        self.cfg = cfg
        self.cache = cache if cache is not None else MinuteCache(cfg)
        self.records = []
        for minute_dir, label, _ in discover_minutes(root_dir):
            if find_csi_file(minute_dir) is None:
                continue
            if not sorted(minute_dir.glob("mmw_radar_raw_*.bin")):
                continue
            self.records.append((minute_dir, label))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        minute_dir, label = self.records[idx]
        radar_windows, csi_windows, _ = self.cache.get_both(str(minute_dir), label)
        if radar_windows is None or csi_windows is None:
            raise RuntimeError(f"No radar/CSI data for {minute_dir}")
        if len(radar_windows) == 0 or len(csi_windows) == 0:
            raise RuntimeError(f"No radar/CSI windows for {minute_dir}")
        common = min(len(radar_windows), len(csi_windows))
        return radar_windows[:common], csi_windows[:common], label


class CSIEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class RadarEncoder(nn.Module):
    def __init__(self, in_channels=4, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(2, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class PresenceModel(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.encoder = CSIEncoder(input_dim, embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        emb = self.encoder(x)
        return self.head(emb)


class DualLocalizationModel(nn.Module):
    def __init__(self, csi_input_dim, radar_channels=4, embed_dim=128, max_targets=8):
        super().__init__()
        self.radar_encoder = RadarEncoder(radar_channels, embed_dim)
        self.csi_encoder = CSIEncoder(csi_input_dim, embed_dim)
        self.fusion = nn.Linear(2 * embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, max_targets * 6)
        self.max_targets = max_targets

    def forward(self, radar, csi):
        r = self.radar_encoder(radar)
        c = self.csi_encoder(csi)
        x = torch.cat([r, c], dim=1)
        x = self.fusion(x)
        x = self.head(x)
        return x.view(-1, self.max_targets, 6)


class RadarOnlyLocalizationModel(nn.Module):
    def __init__(self, radar_channels=4, embed_dim=128, max_targets=8):
        super().__init__()
        self.radar_encoder = RadarEncoder(radar_channels, embed_dim)
        self.head = nn.Linear(embed_dim, max_targets * 6)
        self.max_targets = max_targets

    def forward(self, radar):
        x = self.radar_encoder(radar)
        x = self.head(x)
        return x.view(-1, self.max_targets, 6)


MODEL_REGISTRY = {
    "PresenceModel": PresenceModel,
    "DualLocalizationModel": DualLocalizationModel,
    "RadarOnlyLocalizationModel": RadarOnlyLocalizationModel,
}


def save_model(path, model, model_config, preprocessing_config):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "preprocessing_config": preprocessing_config,
        },
        path,
    )


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    cls = MODEL_REGISTRY[ckpt["model_config"]["class_name"]]
    model = cls(**ckpt["model_config"]["init_args"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["preprocessing_config"]


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    y_label = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_true, y_label)
    f1 = f1_score(y_true, y_label, zero_division=0)
    precision = precision_score(y_true, y_label, zero_division=0)
    recall = recall_score(y_true, y_label, zero_division=0)
    auc = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            pass
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
    }


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    all_y = []
    all_p = []
    for batch_x, batch_y in loader:
        if batch_x is None:
            continue
        batch_x = batch_x.squeeze(0).to(device)
        batch_y = batch_y.squeeze(0).to(device)
        if batch_x.dim() == 0 or batch_x.size(0) == 0:
            continue
        windows = batch_x
        labels = batch_y.expand(windows.size(0)).float()
        optimizer.zero_grad()
        out = model(windows).squeeze(1)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * windows.size(0)
        total += windows.size(0)
        all_y.extend(labels.cpu().numpy().astype(int))
        all_p.extend(torch.sigmoid(out.detach()).cpu().numpy())
    return total_loss / max(total, 1), all_y, all_p


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    all_y = []
    all_p = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            if batch_x is None:
                continue
            batch_x = batch_x.squeeze(0).to(device)
            batch_y = batch_y.squeeze(0).to(device)
            if batch_x.dim() == 0 or batch_x.size(0) == 0:
                continue
            windows = batch_x
            labels = batch_y.expand(windows.size(0)).float()
            out = model(windows).squeeze(1)
            loss = criterion(out, labels)
            total_loss += loss.item() * windows.size(0)
            total += windows.size(0)
            all_y.extend(labels.cpu().numpy().astype(int))
            all_p.extend(torch.sigmoid(out).cpu().numpy())
    return total_loss / max(total, 1), all_y, all_p


def fit_presence_model(model, train_loader, test_loader, epochs, lr, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows = []
    best_f1 = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss, train_y, train_p = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_y, test_p = eval_epoch(model, test_loader, criterion, device)
        train_metrics = compute_metrics(train_y, train_p)
        test_metrics = compute_metrics(test_y, test_p)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_auc": train_metrics["auc"],
            "test_loss": test_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_auc": test_metrics["auc"],
        }
        rows.append(row)
        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rows


# ============ Chunk-based dual-head activity helpers ============


def _activity_from_label(label):
    if not isinstance(label, str):
        return None, None
    if label.startswith("t1_"):
        return "train", label[3:].lower()
    if label.startswith("t2_"):
        return "test", label[3:].lower()
    return None, None


def _parse_iso_to_epoch(ts):
    if ts is None:
        return float("nan")
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return float(pd.to_datetime(ts, utc=True).timestamp())
    except Exception:
        try:
            return float(datetime.datetime.fromisoformat(str(ts)).timestamp())
        except Exception:
            return float("nan")


def discover_chunk_records(root_dir, cfg, min_chunk_frames=10):
    root = Path(root_dir)
    records = []
    activity_set = set()
    for minute_dir in sorted(root.iterdir()):
        if not minute_dir.is_dir():
            continue
        manifest_path = minute_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        labels = manifest.get("labels", []) or []
        if not labels:
            continue
        split, activity = _activity_from_label(labels[0])
        if split is None:
            continue
        activity_set.add(activity)
        csi_files = [
            str(p)
            for p in sorted(minute_dir.glob("wifi_csi*.csv"))
            if p.is_file() and p.stat().st_size > 1000
        ]
        if not csi_files:
            continue
        csi_file = csi_files[0]
        radar_meta = manifest.get("outputs", {}).get("radar", {})
        radar_chunks = radar_meta.get("chunks", [])
        for chunk in radar_chunks:
            bin_name = chunk.get("bin_path")
            if not bin_name:
                continue
            bin_path = minute_dir / bin_name if not Path(bin_name).is_absolute() else Path(bin_name)
            if not bin_path.exists():
                continue
            started = _parse_iso_to_epoch(chunk.get("started"))
            finished = _parse_iso_to_epoch(chunk.get("finished_capture"))
            if not (np.isfinite(started) and np.isfinite(finished)):
                continue
            chunk_seconds = chunk.get("chunk_seconds", finished - started)
            if chunk_seconds is None or not np.isfinite(chunk_seconds) or chunk_seconds <= 0:
                chunk_seconds = cfg.radar_frames / cfg.radar_rate
            chunk_frames = chunk.get("chunk_frames", cfg.radar_frames)
            if chunk_frames < min_chunk_frames:
                continue
            record = {
                "minute": minute_dir.name,
                "minute_dir": str(minute_dir),
                "chunk_index": chunk.get("chunk_index", 0),
                "bin_path": str(bin_path),
                "started": float(started),
                "finished": float(finished),
                "chunk_seconds": float(chunk_seconds),
                "csi_file": csi_file,
                "label": labels[0],
                "activity": activity,
                "split": split,
            }
            records.append(record)
    label_to_idx = {a: i for i, a in enumerate(sorted(activity_set))}
    for rec in records:
        rec["activity_idx"] = label_to_idx[rec["activity"]]
    return records, label_to_idx


def compute_chunk_stats(records):
    by_split = {}
    activity_set = set()
    for rec in records:
        by_split.setdefault(rec["split"], {})
        by_split[rec["split"]][rec["activity"]] = by_split[rec["split"]].get(rec["activity"], 0) + 1
        activity_set.add(rec["activity"])
    return {
        "total_chunks": len(records),
        "by_split": by_split,
        "activities": sorted(activity_set),
    }


def _parse_csi_payload(payload):
    match = re.search(r"\[([^\]]+)\]", payload)
    if not match:
        return None
    numbers = match.group(1).strip().rstrip(",")
    if not numbers:
        return None
    tokens = numbers.split(",")
    if len(tokens) != 128:
        return None
    try:
        values = np.array(tokens, dtype=np.float64)
    except ValueError:
        return None
    imag = values[0::2][CSI_SUBCARRIER_MASK]
    real = values[1::2][CSI_SUBCARRIER_MASK]
    return real + 1j * imag


def load_csi_file(csi_path):
    csi_path = Path(csi_path)
    frame = pd.read_csv(csi_path, on_bad_lines="skip", low_memory=False)
    if "raw_csi_line" in frame.columns:
        rows = frame[frame["raw_csi_line"].astype(str).str.startswith("CSI_DATA", na=False)]
        payloads = rows["raw_csi_line"].astype(str).tolist()
        ts_col = rows.get("host_timestamp")
    elif "data" in frame.columns:
        rows = frame[frame["data"].astype(str).str.startswith("CSI_DATA", na=False)]
        payloads = rows["data"].astype(str).tolist()
        ts_col = rows.get("local_timestamp")
    else:
        raise ValueError(f"Unknown CSI CSV format in {csi_path}")

    if ts_col is None:
        raise ValueError(f"No timestamp column in {csi_path}")

    if pd.api.types.is_numeric_dtype(ts_col):
        times = pd.to_numeric(ts_col, errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(times)
        payloads = [p for p, ok in zip(payloads, valid) if ok]
        times = times[valid]
        if len(times) == 0:
            raise ValueError(f"No valid timestamps in {csi_path}")
        times = (times - times[0]) / 1_000_000.0
    else:
        times = pd.to_datetime(ts_col, errors="coerce", utc=True, format="ISO8601")
        valid = times.notna().to_numpy(dtype=bool)
        payloads = [p for p, ok in zip(payloads, valid) if ok]
        times = times[valid]
        if len(times) == 0:
            raise ValueError(f"No valid timestamps in {csi_path}")
        times = times.astype("int64") / 1e9
        times = times.to_numpy(dtype=np.float64)

    csi = []
    valid_ts = []
    for payload, t in zip(payloads, times):
        comp = _parse_csi_payload(payload)
        if comp is None or not np.isfinite(t):
            continue
        csi.append(comp)
        valid_ts.append(t)
    if not csi:
        raise ValueError(f"No valid CSI rows in {csi_path}")
    csi = np.asarray(csi, dtype=np.complex64)
    valid_ts = np.asarray(valid_ts, dtype=np.float64)
    order = np.argsort(valid_ts, kind="stable")
    return csi[order], valid_ts[order]


def extract_csi_window(csi, times, start, end, csi_frames, margin=0.5):
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        end = start + csi_frames / 100.0
    mask = (times >= (start - margin)) & (times <= (end + margin))
    sel = csi[mask]
    t = times[mask]
    if len(sel) < 2:
        return None
    target_t = np.linspace(start, end, csi_frames)
    real = np.column_stack([np.interp(target_t, t, sel[:, i].real) for i in range(sel.shape[1])])
    imag = np.column_stack([np.interp(target_t, t, sel[:, i].imag) for i in range(sel.shape[1])])
    amplitude = np.abs(real + 1j * imag).astype(np.float32)
    mean = amplitude.mean(axis=0, keepdims=True)
    std = amplitude.std(axis=0, keepdims=True)
    normalized = (amplitude - mean) / (std + 1e-6)
    return np.clip(normalized, -10.0, 10.0).astype(np.float32)


def build_balanced_chunk_arrays(records, cfg, split=None, max_per_class=None, balance=False, random_state=42):
    rng = np.random.RandomState(random_state)
    if split:
        records = [r for r in records if r["split"] == split]
    by_activity = {}
    for r in records:
        by_activity.setdefault(r["activity"], []).append(r)
    selected = []
    if balance and max_per_class:
        target = max_per_class
        for activity, recs in by_activity.items():
            if len(recs) >= target:
                idx = rng.choice(len(recs), target, replace=False)
            else:
                idx = rng.choice(len(recs), target, replace=True)
            selected.extend([recs[i] for i in idx])
    elif max_per_class:
        for activity, recs in by_activity.items():
            idx = rng.choice(len(recs), min(len(recs), max_per_class), replace=False)
            selected.extend([recs[i] for i in idx])
    else:
        for activity, recs in by_activity.items():
            selected.extend(recs)

    selected_by_csi = {}
    for r in selected:
        selected_by_csi.setdefault(r["csi_file"], []).append(r)

    radar_list, csi_list, label_list, used_records = [], [], [], []
    csi_cache = {}
    for csi_file, recs in selected_by_csi.items():
        try:
            csi_cache[csi_file] = load_csi_file(csi_file)
        except Exception as exc:
            print(f"Warning: could not load CSI {csi_file}: {exc}")

    for csi_file, recs in selected_by_csi.items():
        if csi_file not in csi_cache:
            continue
        csi_complex, csi_times = csi_cache[csi_file]
        for r in recs:
            try:
                radar = load_radar_chunk(r["bin_path"], (cfg.radar_size, cfg.radar_size))
                radar = fit_temporal_length(radar, cfg.radar_frames)
                start = r["started"]
                end = start + max(r["chunk_seconds"], cfg.radar_frames / cfg.radar_rate)
                csi = extract_csi_window(csi_complex, csi_times, start, end, cfg.csi_frames)
                if csi is None:
                    continue
                radar_list.append(radar)
                csi_list.append(csi)
                label_list.append(r["activity_idx"])
                used_records.append(r)
            except Exception as exc:
                print(f"Warning: failed to load chunk {r['bin_path']}: {exc}")
                continue
    if not radar_list:
        raise RuntimeError("No valid chunks loaded")
    radar = np.stack(radar_list, axis=0).astype(np.float32)
    csi = np.stack(csi_list, axis=0).astype(np.float32)
    labels = np.asarray(label_list, dtype=np.int64)
    return radar, csi, labels, used_records


# ============ Dual-head activity model ============


class RadarActivityEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        # x: (B, T, H, W, C)
        b, t, h, w, c = x.size()
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(b * t, c, h, w)
        x = self.conv(x)
        x = x.view(b * t, -1)
        x = x.view(b, t, -1)
        x = x.mean(dim=1)
        return self.fc(x)


class CSIActivityEncoder(nn.Module):
    def __init__(self, n_channels, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        # x: (B, L, F)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DualHeadActivityModel(nn.Module):
    def __init__(self, n_channels, n_classes, embed_dim=128, num_heads=4, in_channels=3):
        super().__init__()
        self.radar_encoder = RadarActivityEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.csi_encoder = CSIActivityEncoder(n_channels, embed_dim=embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, radar, csi):
        r, c, fused, attn_weights = self.get_embeddings(radar, csi)
        return self.classifier(fused), attn_weights

    def get_embeddings(self, radar, csi):
        r = self.radar_encoder(radar)
        c = self.csi_encoder(csi)
        tokens = torch.stack([r, c], dim=1)  # (B, 2, E)
        attn_out, attn_weights = self.attention(tokens, tokens, tokens)
        fused = attn_out.mean(dim=1)
        fused = self.norm(fused)
        return r, c, fused, attn_weights


MODEL_REGISTRY["DualHeadActivityModel"] = DualHeadActivityModel


# ============ Multiclass training helpers ============


def compute_multiclass_metrics(y_true, y_pred, n_classes, labels):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    y_label = y_pred.argmax(axis=1)
    cm = confusion_matrix(y_true, y_label, labels=list(range(n_classes)))
    acc = accuracy_score(y_true, y_label)
    f1 = f1_score(y_true, y_label, average="macro", labels=list(range(n_classes)), zero_division=0)
    precision = precision_score(y_true, y_label, average="macro", labels=list(range(n_classes)), zero_division=0)
    recall = recall_score(y_true, y_label, average="macro", labels=list(range(n_classes)), zero_division=0)
    per_class = {}
    for i, name in enumerate(labels):
        yt = (y_true == i).astype(int)
        yp = (y_label == i).astype(int)
        conf = y_pred[:, i]
        per_class[name] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
        }
        try:
            if len(np.unique(yt)) > 1:
                per_class[name]["auc"] = float(roc_auc_score(yt, conf))
            else:
                per_class[name]["auc"] = float("nan")
        except ValueError:
            per_class[name]["auc"] = float("nan")
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_label": y_label.tolist(),
    }


def train_epoch_activity(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    all_y = []
    all_p = []
    for radar, csi, labels in loader:
        radar = radar.to(device)
        csi = csi.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(radar, csi)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * radar.size(0)
        total += radar.size(0)
        all_y.extend(labels.cpu().numpy())
        all_p.extend(F.softmax(logits.detach(), dim=1).cpu().numpy())
    return total_loss / max(total, 1), all_y, all_p


def eval_epoch_activity(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    all_y = []
    all_p = []
    with torch.no_grad():
        for radar, csi, labels in loader:
            radar = radar.to(device)
            csi = csi.to(device)
            labels = labels.to(device)
            logits, _ = model(radar, csi)
            loss = criterion(logits, labels)
            total_loss += loss.item() * radar.size(0)
            total += radar.size(0)
            all_y.extend(labels.cpu().numpy())
            all_p.extend(F.softmax(logits, dim=1).cpu().numpy())
    return total_loss / max(total, 1), all_y, all_p


def fit_activity_model(model, train_loader, test_loader, epochs, lr, device, n_classes, labels, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    rows = []
    best_f1 = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss, train_y, train_p = train_epoch_activity(model, train_loader, criterion, optimizer, device)
        test_loss, test_y, test_p = eval_epoch_activity(model, test_loader, criterion, device)
        train_metrics = compute_multiclass_metrics(train_y, train_p, n_classes, labels)
        test_metrics = compute_multiclass_metrics(test_y, test_p, n_classes, labels)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_macro_recall": train_metrics["macro_recall"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
        }
        rows.append(row)
        if test_metrics["macro_f1"] > best_f1:
            best_f1 = test_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step(test_metrics["macro_f1"])
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rows
