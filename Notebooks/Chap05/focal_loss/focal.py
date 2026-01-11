import torch

def focal_loss_binary_probs(p, y, alpha=0.25, gamma=2.0, eps=1e-8, reduction="none"):
    """
    p: probabilities for class 1, shape [N]
    y: labels in {0,1}, shape [N]
    """
    p = torch.clamp(p, eps, 1 - eps)
    y = y.float()

    pt = torch.where(y == 1, p, 1 - p)
    at = torch.where(y == 1, torch.tensor(alpha, device=p.device), torch.tensor(1 - alpha, device=p.device))

    loss = -at * (1 - pt) ** gamma * torch.log(pt)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

# "лёгкий" позитивный: p=0.99 при y=1
# "трудный" позитивный: p=0.10 при y=1
p = torch.tensor([0.99, 0.10])
y = torch.tensor([1, 1])

print("per-sample FL:", focal_loss_binary_probs(p, y, alpha=0.25, gamma=2.0))
print("per-sample FL (gamma=0):", focal_loss_binary_probs(p, y, alpha=0.25, gamma=0.0))