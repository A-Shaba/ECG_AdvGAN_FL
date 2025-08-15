# attacks/gradient_based/fgsm.py
import torch

def fgsm_attack(model, x, y, eps=0.03):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    x_adv = x + eps * x.grad.sign()
    return torch.clamp(x_adv, -1, 1)  # because we normalized to mean=.5 std=.5
