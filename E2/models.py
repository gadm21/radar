"""Model definitions for the E2 occupancy pipeline.

All models are small (edge-oriented). Embedding dimension is shared so the
gated fusion can weight modalities per-feature:

  radar window (B,5,2,S,S) -> RadarEncoder  -> r_emb (B,E)
  csi window   (B,T,52)    -> CSIEncoder    -> c_emb (B,E)
  g = sigmoid(W[r_emb ; c_emb])             (per-feature gate)
  z = g * r_emb + (1-g) * c_emb -> shared MLP -> logit(occupied)
"""
import torch
from torch import nn


class RadarEncoder(nn.Module):
    """Pooled-statistics encoder — built for cross-placement transfer.

    The T-frame window (B,T,C,H,W) is pooled along time into a per-pixel
    std map (motion/breathing energy), then each channel's map is pooled
    spatially to mean/std/max -> (B, 3C) -> MLP. Absolute levels and
    spatial layout are placement-dependent and destroy transfer; temporal
    variation pooled over space is the robust cue (a logistic regression
    on these features alone reaches test AUC ~0.98)."""

    def __init__(self, in_channels=2, n_frames=50, embed_dim=64, dropout=0.3):
        super().__init__()
        cin = in_channels * 3
        self.fc = nn.Sequential(
            nn.Linear(cin, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, embed_dim), nn.ReLU())

    def forward(self, x):
        # x: (B, T, C, H, W) -> temporal-std map -> spatial mean/std/max
        sd = x.std(1)                                    # (B, C, H, W)
        f = torch.cat([sd.mean((2, 3)), sd.std((2, 3)),
                       sd.amax((2, 3))], dim=1)          # (B, 3C)
        return self.fc(f)


class CSIEncoder(nn.Module):
    """Temporal Conv1d over causal rolling variance of CSI amplitude.

    Follows the 'rolling_variance' pipeline from WifiSensingESP32HAR
    (src/train/utils.py::_rolling_variance): per subcarrier, the variance
    of the trailing `var_window` amplitude samples, computed with
    cumulative sums. Amplitude only — phase is not used. Rolling
    variance removes the receiver-dependent DC/gain and keeps the
    motion-induced fluctuation, the cross-placement cue."""

    def __init__(self, n_features=52, embed_dim=64, dropout=0.3,
                 var_window=20):
        super().__init__()
        self.var_window = var_window
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, embed_dim), nn.ReLU())

    def _rolling_variance(self, x):
        # x: (B, T, F) -> causal rolling variance along T (trailing window)
        B, T, F = x.shape
        w = self.var_window
        if w <= 1:
            return torch.zeros_like(x)
        z = x.new_zeros(B, 1, F)
        cs = torch.cat([z, torch.cumsum(x, dim=1)], dim=1)      # (B,T+1,F)
        cs2 = torch.cat([z, torch.cumsum(x * x, dim=1)], dim=1)
        hi = torch.arange(1, T + 1, device=x.device)
        lo = (hi - w).clamp(min=0)
        cnt = (hi - lo).float().view(1, T, 1)
        mean = (cs[:, hi] - cs[:, lo]) / cnt
        msq = (cs2[:, hi] - cs2[:, lo]) / cnt
        return (msq - mean * mean).clamp(min=0)

    def forward(self, x):
        x = self._rolling_variance(x)
        return self.fc(self.conv(x.permute(0, 2, 1)))


class CSIStatsEncoder(nn.Module):
    """MLP over per-window CSI amplitude statistics.

    The conv CSIEncoder learns receiver/day-specific patterns that do not
    transfer (train_minutes has no empty minutes with CSI at all, and the
    test-day gain scale is ~2x). The amplitude *variance* is the robust
    occupancy cue: occupied minutes show clearly higher variance on every
    split. Inputs: [log1p(var), mean, log1p(mean temporal-std)] of the
    (normalized) amplitude window."""

    def __init__(self, embed_dim=64, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, embed_dim), nn.ReLU())

    def forward(self, csi):
        # csi: (B, T, S) amplitude window (zeros when no CSI)
        stats = torch.stack(
            [torch.log1p(csi.var(dim=(1, 2)).clamp(min=0)),
             csi.mean(dim=(1, 2)),
             torch.log1p(csi.std(dim=1).mean(dim=1).clamp(min=0))], dim=1)
        return self.fc(stats)


class GatedFusion(nn.Module):
    """Per-feature sigmoid gate: z = g*r + (1-g)*c."""

    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, r, c):
        g = torch.sigmoid(self.gate(torch.cat([r, c], dim=1)))
        return g * r + (1.0 - g) * c, g


class OccupancyHead(nn.Module):
    def __init__(self, embed_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, 1))

    def forward(self, z):
        return self.net(z).squeeze(1)


class FusionOccupancyModel(nn.Module):
    def __init__(self, radar_channels=2, csi_features=52, embed_dim=64,
                 dropout=0.3, n_frames=50):
        super().__init__()
        self.radar_encoder = RadarEncoder(
            radar_channels, n_frames=n_frames, embed_dim=embed_dim,
            dropout=dropout)
        # stats encoder (not the conv CSIEncoder): conv features are
        # receiver/day-specific and hijack the gate on the test day
        self.csi_encoder = CSIStatsEncoder(embed_dim, dropout)
        self.fusion = GatedFusion(embed_dim)
        self.head = OccupancyHead(embed_dim, dropout)
        self.last_gate = None

    def forward(self, radar, csi):
        r = self.radar_encoder(radar)
        c = self.csi_encoder(csi)
        # windows with no CSI samples arrive as zeros -> constant stats
        # embedding that can hijack the gate; mask it out
        has_csi = (csi.abs().sum(dim=(1, 2)) > 0).float().unsqueeze(1)
        c = c * has_csi
        z, g = self.fusion(r, c)
        self.last_gate = g.detach()
        return self.head(z)


class RadarOnlyModel(nn.Module):
    def __init__(self, radar_channels=2, embed_dim=64, dropout=0.3,
                 n_frames=50):
        super().__init__()
        self.radar_encoder = RadarEncoder(
            radar_channels, n_frames=n_frames, embed_dim=embed_dim,
            dropout=dropout)
        self.head = OccupancyHead(embed_dim, dropout)

    def forward(self, radar, csi=None):
        return self.head(self.radar_encoder(radar))


class CSIOnlyModel(nn.Module):
    def __init__(self, csi_features=52, embed_dim=64, dropout=0.3):
        super().__init__()
        self.csi_encoder = CSIEncoder(csi_features, embed_dim, dropout)
        self.head = OccupancyHead(embed_dim, dropout)

    def forward(self, csi, _unused=None):
        return self.head(self.csi_encoder(csi))


def build_model(modality, embed_dim=64, dropout=0.3, n_frames=50):
    if modality == "fusion":
        return FusionOccupancyModel(embed_dim=embed_dim, dropout=dropout,
                                    n_frames=n_frames)
    if modality == "radar":
        return RadarOnlyModel(embed_dim=embed_dim, dropout=dropout,
                              n_frames=n_frames)
    if modality == "csi":
        return CSIOnlyModel(embed_dim=embed_dim, dropout=dropout)
    raise ValueError(modality)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
