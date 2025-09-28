# federated/fl_attack_wrapper.py
import torch
from attacks.gan_based.advGAN.attack_advgan import AdvGAN_Attack

class PoisonWithAdvGAN:
    """
    Malicious client hook: replace a fraction of each batch with AdvGAN-perturbed images
    while keeping original labels (error-injection poisoning).
    """
    def __init__(self, target_model, ckpt_path, eps=0.03, frac=0.5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.atk = AdvGAN_Attack(target_model, ckpt_path=ckpt_path, eps=eps, device=self.device)
        self.frac = frac

    def __call__(self, model, x, y):
        # Poison only a fraction of the batch
        bsz = x.size(0)
        k = max(1, int(self.frac * bsz))
        idx = torch.randperm(bsz, device=x.device)[:k]
        x_poison = self.atk(x[idx])
        x = x.clone()
        x[idx] = x_poison
        return x, y
