"""
Image model examples for AutoKernel -- demonstrates support for non-text architectures.

Includes:
    - DiffusionAutoencoder: A simple diffusion-style autoencoder with encoder/decoder
    - UNet2D: A basic 2D U-Net for image-to-image tasks
    - SimpleResNet: A minimal ResNet-style classifier
    - VolumetricSegNet: A 3D convolutional network for volumetric data

Usage:
    uv run profile.py --model models/image_models.py --class-name DiffusionAutoencoder --input-shape 2,3,64,64 --dtype float32
    uv run profile.py --model models/image_models.py --class-name UNet2D --input-shape 2,1,128,128 --dtype float32
    uv run profile.py --model models/image_models.py --class-name SimpleResNet --input-shape 4,3,224,224 --dtype float16
    uv run profile.py --model models/image_models.py --class-name VolumetricSegNet --input-shape 1,1,64,64,64 --dtype float32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Diffusion Autoencoder
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Residual block with optional channel change."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.silu(h + self.skip(x))


class DiffusionAutoencoder(nn.Module):
    """
    A simple diffusion-style autoencoder with convolutional encoder and decoder.

    Architecture:
        Encoder: Conv2d blocks with downsampling → latent space
        Decoder: ConvTranspose2d blocks with upsampling → reconstruction

    This demonstrates how AutoKernel can profile and optimize models with:
    - Conv2d / ConvTranspose2d operations
    - BatchNorm2d normalization
    - SiLU activations (common in diffusion models)

    Input:  (B, in_channels, H, W) float tensor
    Output: (B, in_channels, H, W) float tensor (reconstruction)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 16,
        num_blocks: int = 2,
    ):
        super().__init__()

        ch = base_channels

        # Encoder
        encoder_layers: list[nn.Module] = [
            nn.Conv2d(in_channels, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        ]
        for i in range(num_blocks):
            out_ch = ch * 2
            encoder_layers.append(_ResBlock(ch, out_ch))
            encoder_layers.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))  # downsample
            encoder_layers.append(nn.BatchNorm2d(out_ch))
            encoder_layers.append(nn.SiLU(inplace=True))
            ch = out_ch

        encoder_layers.append(nn.Conv2d(ch, latent_channels, 1))  # project to latent
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: list[nn.Module] = [
            nn.Conv2d(latent_channels, ch, 1),  # project from latent
            nn.SiLU(inplace=True),
        ]
        for i in range(num_blocks):
            out_ch = ch // 2
            decoder_layers.append(_ResBlock(ch, ch))
            decoder_layers.append(nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1))  # upsample
            decoder_layers.append(nn.BatchNorm2d(out_ch))
            decoder_layers.append(nn.SiLU(inplace=True))
            ch = out_ch

        decoder_layers.append(nn.Conv2d(ch, in_channels, 3, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"DiffusionAutoencoder: {n_params / 1e6:.1f}M parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode: returns reconstruction of the input."""
        z = self.encoder(x)
        return self.decoder(z)


# ---------------------------------------------------------------------------
# UNet2D -- simple U-Net for image-to-image tasks
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    """
    A basic 2D U-Net for image segmentation / image-to-image tasks.

    Architecture:
        Encoder path: DoubleConv → MaxPool (×3 levels)
        Bottleneck:   DoubleConv
        Decoder path: Upsample → DoubleConv (×3 levels, with skip connections)

    Input:  (B, in_channels, H, W) float tensor
    Output: (B, out_channels, H, W) float tensor
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_ch: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = _DoubleConv(in_channels, base_ch)
        self.enc2 = _DoubleConv(base_ch, base_ch * 2)
        self.enc3 = _DoubleConv(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _DoubleConv(base_ch * 4, base_ch * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = _DoubleConv(base_ch * 8, base_ch * 4)  # concat doubles channels
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = _DoubleConv(base_ch * 2, base_ch)

        # Final 1×1 conv
        self.final = nn.Conv2d(base_ch, out_channels, 1)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"UNet2D: {n_params / 1e6:.1f}M parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


# ---------------------------------------------------------------------------
# SimpleResNet -- lightweight ResNet classifier
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    """Standard ResNet basic block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class SimpleResNet(nn.Module):
    """
    A lightweight ResNet-style image classifier.

    Input:  (B, 3, H, W) float tensor  (e.g. H=W=224)
    Output: (B, num_classes) float tensor
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_BasicBlock(64, 64), _BasicBlock(64, 64))
        self.layer2 = nn.Sequential(_BasicBlock(64, 128, stride=2), _BasicBlock(128, 128))
        self.layer3 = nn.Sequential(_BasicBlock(128, 256, stride=2), _BasicBlock(256, 256))
        self.layer4 = nn.Sequential(_BasicBlock(256, 512, stride=2), _BasicBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"SimpleResNet: {n_params / 1e6:.1f}M parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# VolumetricSegNet -- 3D convolutional network for volumetric data
# ---------------------------------------------------------------------------

class VolumetricSegNet(nn.Module):
    """
    A 3D convolutional network for volumetric image segmentation.

    Demonstrates support for 3D inputs (e.g. medical imaging, CT scans).

    Input:  (B, in_channels, D, H, W) float tensor
    Output: (B, out_channels, D, H, W) float tensor
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_ch: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_ch, 3, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.BatchNorm3d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.BatchNorm3d(base_ch * 4),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2),
            nn.BatchNorm3d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch, out_channels, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"VolumetricSegNet: {n_params / 1e6:.1f}M parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
