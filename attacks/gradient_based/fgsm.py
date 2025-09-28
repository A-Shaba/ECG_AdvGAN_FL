# attacks/gradient_based/fgsm.py
import torch
import torch.nn.functional as F

def fgsm_attack(model, x, y, eps=0.03, clip_min=-1.0, clip_max=1.0, meta=None):
    """
    FGSM in normalized space. Supports optional meta-data.

    Args:
        model: PyTorch model (in eval() mode)
        x: input batch (N,C,H,W), normalized
        y: labels (N,)
        eps: L_inf step size in normalized space
        clip_min, clip_max: bounds in normalized space
        meta: optional meta-data tensor

    Returns:
        x_adv: adversarial examples (detached)
    """
    x_adv = x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x_adv, meta) if meta is not None else model(x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()
        x_adv.clamp_(clip_min, clip_max)

    return x_adv.detach()