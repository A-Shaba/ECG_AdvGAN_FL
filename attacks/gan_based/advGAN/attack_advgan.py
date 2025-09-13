# attacks/gan_based/advGAN/attack_advgan.py
import torch
from pathlib import Path

from .generator_2d import AdvGANGenerator
from .advgan import AdvGANWrapper

class AdvGAN_Attack:
    """
    Wrapper to use a trained AdvGAN generator at inference/eval time.
    Usage:
        atk = AdvGAN_Attack(target_model, ckpt_path="outputs/advgan/advgan_generator.pt", eps=0.03)
        x_adv = atk(x)   # returns adversarial images
    """
    def __init__(self, target_model, ckpt_path, eps=0.03, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # target model (frozen, eval mode)
        self.target = target_model.to(self.device).eval()
        for p in self.target.parameters():
            p.requires_grad = False

        # load trained generator
        G = AdvGANGenerator(in_ch=1).to(self.device)
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"AdvGAN generator checkpoint not found: {ckpt_path}")
        G.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        G.eval()

        # wrap
        self.wrapper = AdvGANWrapper(self.target, G, None, eps=eps).to(self.device)

    @torch.no_grad()
    def __call__(self, x):
        """Generate adversarial examples for input batch x."""
        x = x.to(self.device)
        x_adv, _ = self.wrapper.perturb(x)
        return x_adv
