"""CNN for spectrogram-based RF fingerprinting.

Same four-block design as the original notebook CNN; ``width`` scales the channel counts so
the model can be trained on a CPU (width=8 gives 8/16/32/64 channels, about 170k parameters,
instead of the original 32-256).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int, pool: bool) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers += [nn.MaxPool2d(2), nn.Dropout2d(0.1)]
    return nn.Sequential(*layers)


class RFCNN(nn.Module):
    def __init__(self, n_classes: int, width: int = 8, emb_dim: int = 128):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(1, w, True), _block(w, 2 * w, True), _block(2 * w, 4 * w, True),
            nn.Conv2d(4 * w, 8 * w, 3, padding=1), nn.BatchNorm2d(8 * w), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.embed = nn.Sequential(nn.Linear(8 * w * 16, emb_dim), nn.BatchNorm1d(emb_dim),
                                   nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.head = nn.Linear(emb_dim, n_classes)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(x))
