#!/usr/bin/env python3
"""Train a dual-head radar+CSI activity classifier.

Usage example:
    python train.py minutes --output_dir output --epochs 20 --max_per_class 500 --balance_train
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ml_utils import (
    DualHeadActivityModel,
    PreprocessingConfig,
    build_balanced_chunk_arrays,
    compute_chunk_stats,
    discover_chunk_records,
    fit_activity_model,
    save_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", default="minutes", nargs="?", help="Root folder containing per-minute subdirectories")
    parser.add_argument("--output_dir", default="output", help="Where to write model, stats, and config")
    parser.add_argument("--sample_seconds", type=float, default=1.0, help="Chunk duration in seconds")
    parser.add_argument("--radar_rate", type=float, default=10.0, help="Radar frame rate (Hz)")
    parser.add_argument("--csi_rate", type=float, default=100.0, help="CSI resampling rate (Hz)")
    parser.add_argument("--radar_size", type=int, default=64, help="Spatial size to resize radar maps")
    parser.add_argument("--csi_n_channels", type=int, default=52, help="Number of CSI subcarrier channels")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument("--max_per_class", type=int, default=500, help="Max chunks per class for train/test (0 = all)")
    parser.add_argument("--balance_train", action="store_true", help="Balance training classes by oversampling")
    parser.add_argument("--balance_test", action="store_true", help="Balance test classes by capping")
    parser.add_argument("--stats_only", action="store_true", help="Discover chunks, print/save stats, then exit")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_training_stats(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreprocessingConfig(
        sample_seconds=args.sample_seconds,
        radar_rate=args.radar_rate,
        csi_rate=args.csi_rate,
        radar_size=args.radar_size,
        csi_n_channels=args.csi_n_channels,
    )

    with open(output_dir / "preprocessing_config.json", "w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Discovering chunk records...")
    records, label_to_idx = discover_chunk_records(args.root, cfg)
    stats = compute_chunk_stats(records)
    print(json.dumps(stats, indent=2))
    with open(output_dir / "chunk_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    with open(output_dir / "label_to_idx.json", "w", encoding="utf-8") as handle:
        json.dump(label_to_idx, handle, indent=2)

    if args.stats_only:
        print("Stats only mode; exiting.")
        return

    n_classes = len(label_to_idx)
    if n_classes == 0:
        raise RuntimeError("No t1/t2 labels found")
    labels = [name for name, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]

    max_train = args.max_per_class if args.max_per_class > 0 else None
    max_test = args.max_per_class if args.max_per_class > 0 else None

    print("Building train arrays...")
    train_radar, train_csi, train_labels, _ = build_balanced_chunk_arrays(
        records, cfg, split="train", max_per_class=max_train, balance=args.balance_train, random_state=args.seed
    )
    print("Building test arrays...")
    test_radar, test_csi, test_labels, _ = build_balanced_chunk_arrays(
        records, cfg, split="test", max_per_class=max_test, balance=args.balance_test, random_state=args.seed
    )

    print(f"Train: {train_radar.shape}, test: {test_radar.shape}, classes: {n_classes}")

    train_dataset = TensorDataset(
        torch.from_numpy(train_radar), torch.from_numpy(train_csi), torch.from_numpy(train_labels)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_radar), torch.from_numpy(test_csi), torch.from_numpy(test_labels)
    )

    train_counts = np.bincount(train_labels, minlength=n_classes).astype(np.float64)
    train_counts[train_counts == 0] = 1
    sample_weights = 1.0 / train_counts[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_weights = torch.from_numpy((1.0 / train_counts) / (1.0 / train_counts).sum()).float()

    model = DualHeadActivityModel(
        n_channels=cfg.csi_n_channels,
        n_classes=n_classes,
        embed_dim=128,
        num_heads=4,
        in_channels=3,
    ).to(device)

    model, rows = fit_activity_model(
        model, train_loader, test_loader, args.epochs, args.lr, device, n_classes, labels, class_weights=class_weights
    )

    model_path = output_dir / "activity_model.pth"
    save_model(
        model_path,
        model,
        model_config={
            "class_name": "DualHeadActivityModel",
            "init_args": {
                "n_channels": cfg.csi_n_channels,
                "n_classes": n_classes,
                "embed_dim": 128,
                "num_heads": 4,
                "in_channels": 3,
            },
        },
        preprocessing_config=cfg.to_dict(),
    )
    print(f"Saved activity model to {model_path}")

    stats_path = output_dir / "activity_training_stats.csv"
    write_training_stats(stats_path, rows)
    print(f"Saved training stats to {stats_path}")
    print("Done.")


if __name__ == "__main__":
    main()
