# attacks/gradient_based/pgd.py
import torch
import torch.nn.functional as F

def pgd_attack(model, x, y, eps=0.03, alpha=0.005, steps=40, clip_min=-1.0, clip_max=1.0, meta=None):
    """
    PGD attack (L_inf) in normalized space with optional meta-data.

    Args:
        model: PyTorch model (eval mode)
        x: input batch (N,C,H,W), normalized
        y: ground truth labels
        eps: maximum perturbation
        alpha: step size
        steps: number of steps
        clip_min: min pixel value (normalized)
        clip_max: max pixel value (normalized)
        meta: optional tensor with meta-data

    Returns:
        Adversarial example tensor (detached)
    """
    x_adv = x.detach().clone()
    x_adv = x_adv + 0.001 * torch.randn_like(x_adv)  # small random init
    x_adv = torch.clamp(x_adv, clip_min, clip_max)

    for _ in range(steps):
        x_adv.requires_grad = True
        model.zero_grad(set_to_none=True)

        logits = model(x_adv, meta) if meta is not None else model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = x_adv.clamp(clip_min, clip_max)
            x_adv = x_adv.detach()

    return x_adv