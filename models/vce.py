#Author: Billy
#Visual Confidence Estimator, scores each frame's reliability as alpha in [0,1].

import torch
import torch.nn as nn


class VCE(nn.Module):
    """[B, T, D] -> [B, T, 1], per-frame confidence score."""

    def __init__(self, input_dim=512, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 64]

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, visual_feat):
        return self.net(visual_feat)


class VCEWithTemporalSmoothing(VCE):
    """VCE + causal Conv1d smoothing to reduce frame-to-frame jitter."""

    def __init__(self, input_dim=512, hidden_dims=None, smooth_kernel=5):
        super().__init__(input_dim, hidden_dims)

        self.smooth_kernel = smooth_kernel
        self.smoother = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=smooth_kernel,
            padding=0,
            bias=False,
        )
        nn.init.constant_(self.smoother.weight, 1.0 / smooth_kernel)

    def forward(self, visual_feat):
        alpha = super().forward(visual_feat)  #[B, T, 1]

        alpha = alpha.permute(0, 2, 1)  #[B, 1, T]
        alpha = torch.nn.functional.pad(alpha, (self.smooth_kernel - 1, 0))  #causal left-pad
        alpha = self.smoother(alpha)
        alpha = alpha.permute(0, 2, 1)  #[B, T, 1]
        alpha = torch.clamp(alpha, 0.0, 1.0)
        return alpha
