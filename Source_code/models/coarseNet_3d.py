"""Lightweight 3D coarse network for stage-1 localization."""
import torch
import torch.nn as nn
from timm.layers import DropPath


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        return self.conv(x) + self.skip(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels * 2, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CoarseNet3D(nn.Module):
    """Lightweight 3D UNet for coarse whole tumor localization."""
    def __init__(self, in_channels=4, base_channels=8):
        super().__init__()

        self.enc1 = ConvBlock3D(in_channels, base_channels, stride=1)  # Same size
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2, stride=2)  # 1/2
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4, stride=2)  # 1/4
        self.enc4 = ConvBlock3D(base_channels * 4, base_channels * 8, stride=2)  # 1/8

        self.bottleneck = ConvBlock3D(base_channels * 8, base_channels * 8)

        self.dec3 = UpBlock3D(base_channels * 8, base_channels * 4)  # 1/4
        self.dec2 = UpBlock3D(base_channels * 4, base_channels * 2)  # 1/2
        self.dec1 = UpBlock3D(base_channels * 2, base_channels)  # Full

        self.out = nn.Conv3d(base_channels, 1, kernel_size=1)
        nn.init.constant_(self.out.bias, -3.5)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.out(d1)
        return torch.sigmoid(out)


class CoarseNet3D_Tiny(nn.Module):
    """Smaller CoarseNet3D variant for faster localization."""
    def __init__(self, in_channels=4, base_channels=4):
        super().__init__()

        self.enc1 = ConvBlock3D(in_channels, base_channels, stride=1)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2, stride=2)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4, stride=2)
        self.enc4 = ConvBlock3D(base_channels * 4, base_channels * 8, stride=2)

        self.bottleneck = ConvBlock3D(base_channels * 8, base_channels * 8)

        self.dec3 = UpBlock3D(base_channels * 8, base_channels * 4)
        self.dec2 = UpBlock3D(base_channels * 4, base_channels * 2)
        self.dec1 = UpBlock3D(base_channels * 2, base_channels)

        self.out = nn.Conv3d(base_channels, 1, kernel_size=1)
        nn.init.constant_(self.out.bias, -3.5)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.out(d1)
        return torch.sigmoid(out)


if __name__ == "__main__":
    model = CoarseNet3D(in_channels=4, base_channels=8)
    x = torch.randn(2, 4, 40, 40, 32)  # 4x downsampled from 160x160x128
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    model_tiny = CoarseNet3D_Tiny(in_channels=4, base_channels=4)
    out_tiny = model_tiny(x)
    print(f"Tiny - Input: {x.shape}, Output: {out_tiny.shape}")
    print(f"Tiny Parameters: {sum(p.numel() for p in model_tiny.parameters()) / 1e6:.2f}M")
