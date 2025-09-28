import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, img_size=(64,256)):
        super().__init__()
        self.enc1 = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.enc2 = nn.Conv2d(base_ch, base_ch*2, 4, 2, 1)
        self.enc3 = nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1)
        self.middle = nn.Conv2d(base_ch*4, base_ch*4, 3, padding=1)
        self.dec3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1)
        self.dec1 = nn.Conv2d(base_ch, in_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x, noise_level=None):
        # noise_level: not used in simple version
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        m = self.relu(self.middle(e3))
        d3 = self.relu(self.dec3(m) + e2)
        d2 = self.relu(self.dec2(d3) + e1)
        out = self.tanh(self.dec1(d2))
        return out

class DiffusionGenerator(nn.Module):
    """
    DDPM toy wrapper for fast adversarial sample.
    For simplicity, adds random noise to the image, denoises with a U-Net, returns difference.
    """
    def __init__(self, in_ch=1, img_size=(64,256)):
        super().__init__()
        self.unet = SimpleUNet(in_ch=in_ch, img_size=img_size)
    def forward(self, x, noise_level=None):
        # Add noise, denoise (simulate one step)
        noise = torch.randn_like(x)
        x_noisy = x + 0.2 * noise    # scale as needed
        x_denoised = self.unet(x_noisy)
        delta = x_denoised - x
        return delta