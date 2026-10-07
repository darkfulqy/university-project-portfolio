"""EEGNet v4 (Lawhern et al., 2018), PyTorch."""
import torch
import torch.nn as nn


class _MaxNormConv2d(nn.Conv2d):
    """Depthwise conv with the max-norm weight constraint the paper specifies."""

    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            norm = self.weight.norm(dim=(2, 3), keepdim=True).clamp(min=self.max_norm / 2)
            self.weight.mul_(torch.clamp(norm, max=self.max_norm) / norm)
        return super().forward(x)


class EEGNet(nn.Module):
    def __init__(self, n_classes=4, n_chans=22, n_samples=1000,
                 F1=8, D=2, F2=16, kern_length=125, dropout=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            _MaxNormConv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_feat = self.block2(self.block1(torch.zeros(1, 1, n_chans, n_samples))).numel()
        self.head = nn.Linear(n_feat, n_classes)

    def forward(self, x):            # x: (N, C, T)
        x = x.unsqueeze(1)
        x = self.block2(self.block1(x))
        return self.head(x.flatten(1))
