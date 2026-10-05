"""
Component 1: Pre-trained Image Classification Network (Encoder).
Extracts multi-scale hierarchical feature maps from an ImageNet-pretrained backbone.
Focuses on shallow-to-mid representations (conv1, layer1, layer2) proven by our empirical
divergence test to carry strong, spatially-discriminative moiré sensitivity.
"""

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ClassifierEncoder(nn.Module):
    """
    Pretrained Classification Encoder extracting:
    - F0: Shallow high-res edge features (C=64, 1/2 res)
    - F1: Layer1 texture features (C=64, 1/4 res)
    - F2: Layer2 mid-level semantic features (C=128, 1/8 res)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        # Register ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if pretrained:
            url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
            state_dict = load_state_dict_from_url(url, progress=False, map_location="cpu")
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

        # Freeze encoder parameters so classification priors remain faithful
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def _make_layer(self, in_planes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(in_planes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_norm = (x - self.mean) / self.std
        f0 = self.relu(self.bn1(self.conv1(x_norm))) # (B, 64, H/2, W/2)
        p1 = self.maxpool(f0)
        f1 = self.layer1(p1)                         # (B, 64, H/4, W/4)
        f2 = self.layer2(f1)                         # (B, 128, H/8, W/8)

        return {"f0": f0, "f1": f1, "f2": f2}
