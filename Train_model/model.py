# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleDBNet(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, res.layer3
        )
        self.neck = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.prob_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1)
        )
        self.thresh_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        feat = self.neck(self.backbone(x))
        size = x.shape[2:]
        prob = F.interpolate(self.prob_head(feat), size=size, mode="bilinear", align_corners=False)
        thresh = F.interpolate(self.thresh_head(feat), size=size, mode="bilinear", align_corners=False)
        return prob, thresh