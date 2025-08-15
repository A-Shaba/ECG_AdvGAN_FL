import torch.nn as nn

class AdvGANDiscriminator(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)
