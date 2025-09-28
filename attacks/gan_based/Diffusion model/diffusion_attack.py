import torch
from diffusion_gen import DiffusionGenerator

class Diffusion_Attack:
    def __init__(self, diff_ckpt, device="cuda", in_ch=1, img_size=(64,256), eps=0.03):
        self.device = device
        self.diffgen = DiffusionGenerator(in_ch=in_ch, img_size=img_size).to(device)
        sd = torch.load(diff_ckpt, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        self.diffgen.load_state_dict(sd)
        self.diffgen.eval()
        self.eps = eps
    @torch.no_grad()
    def __call__(self, x):
        x = x.to(self.device)
        delta = torch.clamp(self.diffgen(x), -self.eps, self.eps)
        x_adv = torch.clamp(x + delta, -1.0, 1.0)
        return x_adv, delta