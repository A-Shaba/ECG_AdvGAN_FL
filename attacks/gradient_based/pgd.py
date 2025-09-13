# attacks/gradient_based/pgd.py
import torch
import torch.nn.functional as F

def pgd_attack(model, x, y, eps=0.03, alpha=0.005, steps=40,
               clip_min=-1.0, clip_max=1.0):
    """
    PGD (Projected Gradient Descent) adversarial attack.
    Operates in normalized space [-1,1].
    
    Args:
        model: PyTorch model (in eval() mode)
        x: input batch (N,C,H,W), normalized
        y: labels (N,)
        eps: L_inf budget
        alpha: step size
        steps: number of iterations
        clip_min, clip_max: bounds in normalized space
    Returns:
        x_adv: adversarial examples (detached)
    """
    # start from a perturbed point inside the epsilon-ball
    x_adv = x.detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, clip_min, clip_max).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        with torch.no_grad():
            # gradient ascent on the loss
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # project back to the epsilon ball around the original input
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            # clamp to valid image range
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

    return x_adv.detach()
