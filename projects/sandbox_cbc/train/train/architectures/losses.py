"""Module storing loss functions"""
import torch
import torch.nn.functional as F
from torch import nn


# implemented from https://github.com/violatingcp/codec
class VICRegLoss(nn.Module):
    def forward(self, x, y, wt_repr=1.0, wt_cov=1.0, wt_std=1.0):
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        N = x.size(0)
        D = x.size(1)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = (
            torch.mean(F.relu(1 - std_x)) / 2
            + torch.mean(F.relu(1 - std_y)) / 2
        )

        x = (x - x.mean(dim=0)) / x.std(dim=0)
        y = (y - y.mean(dim=0)) / y.std(dim=0)

        # transpose dims 1 and 2; keep batch dim i.e. 0, unchanged
        cov_x = (x.transpose(1, 2) @ x) / (N - 1)
        cov_y = (y.transpose(1, 2) @ y) / (N - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(D)
        cov_loss += self.off_diagonal(cov_y).pow_(2).sum().div(D)

        s = wt_repr * repr_loss + wt_cov * cov_loss + wt_std * std_loss
        return s, repr_loss, cov_loss, std_loss

    def off_diagonal(self, x):
        num_batch, n, m = x.shape
        assert n == m
        # All off diagonal elements from complete batch flattened
        return (
            x.flatten(start_dim=1)[..., :-1]
            .view(num_batch, n - 1, n + 1)[..., 1:]
            .flatten()
        )
