# generator_2d.py
import torch
import torch.nn as nn

def weights_init_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
        for p in m.parameters():
            if p is not None:
                try:
                    nn.init.ones_(p)
                except Exception:
                    pass

class ResidualBlockGN(nn.Module):
    """
    Residual block with GroupNorm -> ReLU -> Conv -> GroupNorm -> ReLU -> Conv and skip connection.
    GroupNorm stabilizes training for small batch sizes.
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=groups if channels >= groups else 1, num_channels=channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=groups if channels >= groups else 1, num_channels=channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        return self.relu(out + x)

class AdvGANGenerator(nn.Module):
    """
    Generator that produces a perturbation map δ of same spatial size as input.
    - Input: (B, in_ch, H, W)
    - Output: (B, in_ch, H, W) raw values (no tanh). The wrapper applies tanh * eps.
    Design:
      conv -> down-proj -> several residual blocks -> up proj -> conv to in_ch
    """
    def __init__(self, in_ch=1, base_ch=64, n_resblocks=4, groups=8):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups if base_ch >= groups else 1, num_channels=base_ch),
            nn.ReLU(inplace=True)
        )

        # A few residual blocks at base channel count
        res_blocks = []
        for _ in range(n_resblocks):
            res_blocks.append(ResidualBlockGN(base_ch, groups=groups))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Refinement & output
        self.refine = nn.Sequential(
            nn.Conv2d(base_ch, base_ch//2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, (base_ch//2)//groups), num_channels=base_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, in_ch, kernel_size=1, padding=0, bias=True)
            # Note: no activation here; wrapper will apply tanh and eps scaling
        )

        # init weights
        self.apply(weights_init_xavier)

    def forward(self, x):
        out = self.initial(x)
        out = self.res_blocks(out)
        out = self.refine(out)
        return out