import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ColorizationCNN(nn.Module):
    """
    CNN model for image colorization inspired by
    'Learning Representations for Automatic Colorization'
    """
    def __init__(self, pretrained_backbone=True):
        super().__init__()
        # Encoder: pretrained VGG16 up to conv5
        vgg = models.vgg16_bn(pretrained=pretrained_backbone)
        features = list(vgg.features.children())
        # Use features up to layer that still has spatial resolution
        self.enc1 = nn.Sequential(*features[:  6])   # out ~64
        self.enc2 = nn.Sequential(*features[6 : 13]) # out ~128
        self.enc3 = nn.Sequential(*features[13: 23]) # out ~256
        self.enc4 = nn.Sequential(*features[23: 33]) # out ~512

        # Bottleneck conv
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec4 = nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1)
        self.dec3 = nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)

        # Output: 2 channels (ab)
        self.out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # low-level
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder with skip connections
        d4 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d4 = F.relu(self.dec4(torch.cat([d4, e4], dim=1)))

        d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = F.relu(self.dec3(torch.cat([d3, e3], dim=1)))

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = F.relu(self.dec2(torch.cat([d2, e2], dim=1)))

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = F.relu(self.dec1(torch.cat([d1, e1], dim=1)))

        out = torch.tanh(self.out(d1))
        return out
