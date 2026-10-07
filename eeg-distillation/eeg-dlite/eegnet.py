"""EEGNet-8,2 (Lawhern et al., 2018, J. Neural Eng.) in PyTorch.

Faithful to the reference Keras implementation (arl-eegmodels/EEGModels.py, `EEGNet`):
  block1: Conv2D(F1, (1, kernLength), same, no bias) -> BN
          DepthwiseConv2D((C, 1), depth_multiplier=D, max_norm=1, no bias) -> BN -> ELU
          AvgPool(1, 4) -> Dropout
  block2: SeparableConv2D(F2, (1, 16), same, no bias) -> BN -> ELU -> AvgPool(1, 8) -> Dropout
  Flatten -> Dense(num_classes, max_norm=0.25)
Max-norm constraints are applied after every optimizer step via `apply_max_norm()`.

The constructor signature also accepts torcheeg's `EEGNet(chunk_size, num_electrodes, F1, F2, D,
num_classes, kernel_1, kernel_2, dropout)` keyword names so it can stand in for
`torcheeg.models.cnn.EEGNet` when the official EEG-DLite module is imported (the k-center path
never instantiates it).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, chunk_size: int = 800, num_electrodes: int = 22, F1: int = 8, F2: int = 16, D: int = 2,
                 num_classes: int = 4, kernel_1: int = 100, kernel_2: int = 16, dropout: float = 0.5,
                 **kwargs):
        super().__init__()
        self.chunk_size, self.num_electrodes = chunk_size, num_electrodes
        self.F1, self.F2, self.D = F1, F2, D
        self.kernel_1, self.kernel_2 = kernel_1, kernel_2

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), padding="same", bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3),
            nn.Conv2d(F1, F1 * D, (num_electrodes, 1), groups=F1, bias=False),   # depthwise, max_norm=1
            nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, kernel_2), padding="same", groups=F1 * D, bias=False),  # separable: depthwise
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),                                            # separable: pointwise
            nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_feat = self.block2(self.block1(torch.zeros(1, 1, num_electrodes, chunk_size))).numel()
        self.classifier = nn.Linear(n_feat, num_classes)                                          # max_norm=0.25

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)          # (B, 1, C, T)
        x = self.block2(self.block1(x))
        return self.classifier(x.flatten(1))

    @torch.no_grad()
    def apply_max_norm(self, depthwise_max: float = 1.0, dense_max: float = 0.25):
        """Keras-style `max_norm` kernel constraints.
        Depthwise: one L2 norm per output filter (F1*D of them), as in braindecode's EEGNetv4.  NOTE the Keras
        reference uses `max_norm(1., axis=(0,1,2))` on a (C,1,F1,D) kernel, i.e. one norm per depth-multiplier
        index d over all F1 filters, which is a slightly *looser* constraint; the per-filter version is the
        common PyTorch port (documented in README row 12).  Dense: axis=0 -> per class column, same as here."""
        w = self.block1[2].weight                 # (F1*D, 1, C, 1)
        norms = w.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
        w.mul_((norms.clamp(max=depthwise_max) / norms).view(-1, 1, 1, 1))
        w = self.classifier.weight                # (classes, n_feat); Keras kernel is (n_feat, classes), axis=0 -> per class
        norms = w.norm(dim=1, keepdim=True).clamp_min(1e-12)
        w.mul_(norms.clamp(max=dense_max) / norms)
