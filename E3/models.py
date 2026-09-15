"""Model definitions for the E3 left/right localization pipeline.

Unlike E2 (cross-placement occupancy, where absolute levels must be
discarded), here there is only one room / one radar placement, so the
model is free to learn from absolute spatial structure. The encoder keeps
BOTH a temporal-mean map (average reflector position/strength over the
window) and a temporal-max map (strongest instantaneous return, robust to
occasional dropouts/motion) per channel, then a small 2-D CNN learns
whatever spatial pattern (line-of-sight azimuth, multipath signature,
etc.) discriminates left vs right.

  radar window (B,T,2,S,S) -> [mean_t ; max_t] -> (B,4,S,S) -> CNN -> logit
"""
import torch
from torch import nn


class LeftRightEncoder(nn.Module):
    def __init__(self, in_channels=2, embed_dim=64, dropout=0.3):
        super().__init__()
        cin = in_channels * 2  # mean + max per channel
        self.conv = nn.Sequential(
            nn.Conv2d(cin, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, embed_dim), nn.ReLU()
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        mean_t = x.mean(1)
        max_t = x.amax(1)
        f = torch.cat([mean_t, max_t], dim=1)  # (B, 2C, H, W)
        return self.fc(self.conv(f))


class LeftRightHead(nn.Module):
    def __init__(self, embed_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(1)


class LeftRightModel(nn.Module):
    def __init__(self, radar_channels=2, embed_dim=64, dropout=0.3):
        super().__init__()
        self.radar_encoder = LeftRightEncoder(radar_channels, embed_dim, dropout)
        self.head = LeftRightHead(embed_dim, dropout)

    def forward(self, radar):
        return self.head(self.radar_encoder(radar))


def build_model(embed_dim=64, dropout=0.3):
    return LeftRightModel(embed_dim=embed_dim, dropout=dropout)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
