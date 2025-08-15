# models/ecg_classifier_cnn.py
import torch, torch.nn as nn
from torchvision.models import resnet18

class SmallECGCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, num_classes)
    def forward(self,x):
        x = self.net(x).flatten(1)
        return self.fc(x)

def resnet18_gray(num_classes=3):
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
