"""Model definitions for the E4 hierarchical pipeline.

Three stages are composed:

  * OccupancyModel: binary (empty/occupied). This is E2's exported
    `RadarOnlyModel` — the class below reproduces its module names
    (`radar_encoder.fc.*`, `head.net.*`) so `E2/outputs/best_model.pt`
    loads directly. Trained on t1, validated on t2, tested on t by E2.

  * SleepPresentModel: binary (sleep/present), run only on windows the
    occupancy stage calls occupied. CNN over temporal mean+max+std maps
    (keeps spatial structure): the model is trained on the deployment
    placement t (optionally augmented with t1+t2), so absolute spatial
    structure is informative rather than a placement shortcut.

  * PositionModel: binary (left/right), run only on windows called
    present. Reuses the E3 `LeftRightEncoder` (temporal mean+max maps ->
    2D CNN), which keeps absolute spatial structure — right for the
    single fixed placement t. Trained/evaluated on t only.
"""
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Activity model (E2-style encoder, 3-class head)
# ---------------------------------------------------------------------------
class ActivityEncoder(nn.Module):
    """Same architecture as E2.models.RadarEncoder (temporal-std pooling)."""

    def __init__(self, in_channels=2, embed_dim=64, dropout=0.3):
        super().__init__()
        cin = in_channels * 3
        self.fc = nn.Sequential(
            nn.Linear(cin, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, embed_dim), nn.ReLU())

    def forward(self, x):
        # x: (B, T, C, H, W)
        sd = x.std(1)
        f = torch.cat([sd.mean((2, 3)), sd.std((2, 3)), sd.amax((2, 3))], dim=1)
        return self.fc(f)


class _E2Head(nn.Module):
    """Matches E2.models.OccupancyHead (submodule name `net`)."""

    def __init__(self, embed_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    def forward(self, z):
        return self.net(z).squeeze(1)


class OccupancyModel(nn.Module):
    """Matches E2.models.RadarOnlyModel state-dict keys
    (`radar_encoder.fc.*`, `head.net.*`) so the exported
    `E2/outputs/best_model.pt` loads directly."""

    def __init__(self, radar_channels=2, embed_dim=64, dropout=0.3):
        super().__init__()
        self.radar_encoder = ActivityEncoder(radar_channels, embed_dim, dropout)
        self.head = _E2Head(embed_dim, dropout)

    def forward(self, radar):
        return self.head(self.radar_encoder(radar))


class MapStatsEncoder(nn.Module):
    """Temporal mean+max+std maps -> 2D CNN.

    Pools each window along time into three per-channel maps (mean, max,
    std) and feeds the resulting 3*C-channel image to a small CNN. Unlike
    `ActivityEncoder` (which collapses each map to 3 scalars and so only
    keeps relative temporal variability), this preserves WHERE in the
    range/angle space the motion happens — the right inductive bias when
    the model is trained on the deployment placement.
    """

    def __init__(self, in_channels=2, embed_dim=64, dropout=0.3):
        super().__init__()
        cin = in_channels * 3
        self.conv = nn.Sequential(
            nn.Conv2d(cin, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, embed_dim), nn.ReLU())

    def forward(self, x):
        # x: (B, T, C, H, W)
        f = torch.cat([x.mean(1), x.amax(1), x.std(1)], dim=1)
        return self.fc(self.conv(f))


class SleepPresentModel(nn.Module):
    """Binary sleep(0)/present(1) head on the map-stats CNN encoder."""

    def __init__(self, radar_channels=2, embed_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = MapStatsEncoder(radar_channels, embed_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    def forward(self, radar):
        return self.head(self.encoder(radar)).squeeze(1)


# ---------------------------------------------------------------------------
# Position model (E3-style encoder, binary head)
# ---------------------------------------------------------------------------
class PositionEncoder(nn.Module):
    """Same architecture as E3.models.LeftRightEncoder (mean+max maps -> CNN)."""

    def __init__(self, in_channels=2, embed_dim=64, dropout=0.3):
        super().__init__()
        cin = in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(cin, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, embed_dim), nn.ReLU())

    def forward(self, x):
        mean_t = x.mean(1)
        max_t = x.amax(1)
        f = torch.cat([mean_t, max_t], dim=1)
        return self.fc(self.conv(f))


class PositionModel(nn.Module):
    def __init__(self, radar_channels=2, embed_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = PositionEncoder(radar_channels, embed_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    def forward(self, radar):
        return self.head(self.encoder(radar)).squeeze(1)


def build_occupancy_model(embed_dim=64, dropout=0.3):
    return OccupancyModel(embed_dim=embed_dim, dropout=dropout)


def build_sleep_present_model(embed_dim=64, dropout=0.3):
    return SleepPresentModel(embed_dim=embed_dim, dropout=dropout)


def build_position_model(embed_dim=64, dropout=0.3):
    return PositionModel(embed_dim=embed_dim, dropout=dropout)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
