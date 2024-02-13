import torch
import torch.nn as nn


def check_loss(input, target, tau, reduction: str = "mean"):
    error = target - input

    loss = torch.where(error < 0, tau - 1, tau) * error

    return torch.mean(loss) if reduction == "mean" else torch.sum(loss)


def smooth_check_loss(input, target, tau, alpha, reduction: str = "mean"):
    error = target - input

    loss = tau * error + alpha * torch.log(1 + torch.exp(-error / alpha))

    return torch.mean(loss) if reduction == "mean" else torch.sum(loss)


class CheckLoss(nn.Module):
    def __init__(self, tau, reduction="mean"):
        super().__init__()
        self.tau = tau
        self.reduction = reduction

    def forward(self, input, target):
        return check_loss(input, target, self.tau, self.reduction)


class SmoothCheckLoss(nn.Module):
    def __init__(self, tau, alpha, reduction="mean"):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        return smooth_check_loss(input, target, self.tau, self.alpha, self.reduction)
