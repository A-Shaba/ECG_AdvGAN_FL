# models/ecg_classifier_cnn.py
import torch
import torch.nn as nn
from torchvision.models import resnet18

class SmallECGCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=3, meta_dim=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.meta_dim = meta_dim
        self.fc1 = nn.Linear(128, 128)
        if meta_dim > 0:
            # embed meta to a small vector
            self.meta_fc = nn.Linear(meta_dim, 32)
            self.fc2 = nn.Linear(128 + 32, num_classes)
        else:
            self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, meta=None):
        x = self.net(x).flatten(1)
        x = torch.relu(self.fc1(x))
        if self.meta_dim > 0 and meta is not None:
            meta_out = torch.relu(self.meta_fc(meta))
            x = torch.cat([x, meta_out], dim=1)
        x = self.fc2(x)
        return x


class DeepECGCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=3, meta_dim=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.meta_dim = meta_dim
        self.fc1 = nn.Linear(256, 256)
        if meta_dim > 0:
            self.meta_fc = nn.Linear(meta_dim, 64)
            self.fc2 = nn.Linear(256 + 64, num_classes)
        else:
            self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, meta=None):
        x = self.net(x).flatten(1)
        x = torch.relu(self.fc1(x))
        if self.meta_dim > 0 and meta is not None:
            meta_out = torch.relu(self.meta_fc(meta))
            x = torch.cat([x, meta_out], dim=1)
        x = self.fc2(x)
        return x


class ResNet18Gray(nn.Module):
    """
    Wrapper around torchvision resnet18 that supports grayscale input and optional meta-data.
    If meta_dim > 0, meta is projected and concatenated before the final linear layer.
    """
    def __init__(self, num_classes=3, meta_dim=0):
        super().__init__()
        # Create a resnet backbone (fc replaced with Identity)
        backbone = resnet18(weights=None)
        # Modify first conv to accept 1-channel grayscale
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # we'll build our own head
        self.backbone = backbone

        self.meta_dim = meta_dim
        if meta_dim > 0:
            # meta embedding
            self.meta_fc = nn.Linear(meta_dim, 64)
            self.classifier = nn.Linear(feat_dim + 64, num_classes)
        else:
            self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x, meta=None):
        # backbone returns features (N, feat_dim)
        feats = self.backbone(x)
        if self.meta_dim > 0 and meta is not None:
            meta_out = torch.relu(self.meta_fc(meta))
            feats = torch.cat([feats, meta_out], dim=1)
        logits = self.classifier(feats)
        return logits


def make_model(name, num_classes, meta_dim=0):
    if name == "small_cnn":
        return SmallECGCNN(1, num_classes, meta_dim)
    elif name == "deep_cnn":
        return DeepECGCNN(1, num_classes, meta_dim)
    elif name == "resnet18":
        return ResNet18Gray(num_classes=num_classes, meta_dim=meta_dim)
    else:
        raise ValueError(f"Unknown model name: {name}")