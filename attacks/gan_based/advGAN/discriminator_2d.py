# discriminator_2d.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def conv_block(in_ch, out_ch, kernel=4, stride=2, padding=1, use_sn=True, norm=True):
    layers = []
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=not use_sn)
    if use_sn:
        conv = spectral_norm(conv)
    layers.append(conv)
    if norm:
        # InstanceNorm tends to work well for GAN discriminators; avoid affine parameters to reduce leakage
        layers.append(nn.InstanceNorm2d(out_ch, affine=False))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class AdvGANDiscriminator(nn.Module):
    """
    PatchGAN-style discriminator with spectral normalization.
    Input: (B, in_ch, H, W)
    Output: (B, 1) scalar logit per image (via global average pooling of patch map)
    Note: returning a scalar logit aligns with BCEWithLogitsLoss.
    """
    def __init__(self, in_ch=1, base_ch=64, n_layers=4, use_sn=True):
        super().__init__()
        layers = []
        # First layer: no normalization
        layers.append(conv_block(in_ch, base_ch, kernel=4, stride=2, padding=1, use_sn=use_sn, norm=False))
        ch = base_ch
        for n in range(1, n_layers):
            out_ch = min(ch * 2, 512)
            layers.append(conv_block(ch, out_ch, kernel=4, stride=2, padding=1, use_sn=use_sn, norm=True))
            ch = out_ch

        # Final conv to produce a "patch" score map
        final_conv = nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1)
        if use_sn:
            final_conv = spectral_norm(final_conv)
        self.features = nn.Sequential(*layers)
        self.final_conv = final_conv
        self.pool = nn.AdaptiveAvgPool2d(1)  # reduce patch map to 1x1
        # init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.features(x)
        p = self.final_conv(f)         # shape: (B,1,Hp,Wp)
        pooled = self.pool(p).view(x.size(0), -1)  # shape: (B,1)
        return pooled