# models/utils.py
import torch, torch.nn as nn

def init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)

def save_ckpt(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_ckpt(model, path, map_location="cpu"):
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model