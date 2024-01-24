import torch

def check_loss(input, target, tau, reduction: str = "mean"):
    error = target - input

    loss = torch.where(error < 0, tau - 1, tau) * error

    return torch.mean(loss) if reduction == "mean" else torch.sum(loss)

# def smooth_check_loss(input, target, tau, reduction: str = "mean"):
#     error = target - input

#     return torch.mean(error) if reduction == "mean" else torch.sum(error)