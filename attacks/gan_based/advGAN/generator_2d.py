import torch.nn as nn

class AdvGANGenerator(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, in_ch, 1)  # output perturbation δ
        )
    def forward(self, x): return self.net(x)
