"""
Component 3: Reconstruction Decoder with Correction & Recovery Blocks (CRBs).
Rebuilds clean, artifact-free image from prior-guided features and multi-scale encoder skips.
Uses base-frequency decoupling: starts from a moiré-attenuated structural base and utilizes
prior-guided attention features to restore clean edges, contrast, and fine natural details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectionRecoveryBlock(nn.Module):
    """
    Residual block with Channel Attention to selectively suppress corrupted channels
    and enhance clean textural representations.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Squeeze-and-Excitation Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 8), channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.relu(self.conv1(x)))
        res = res * self.ca(res)
        return x + res


class ReconstructionDecoder(nn.Module):
    """
    Multi-scale Progressive Decoder with CRBs:
    - Stage 1: (C=128, H/8, W/8) -> Upsample to H/4 + Skip F1 (64ch) -> 96ch
    - Stage 2: (C=96, H/4, W/4)  -> Upsample to H/2 + Skip F0 (64ch) -> 64ch
    - Stage 3: (C=64, H/2, W/2)  -> Upsample to H, W               -> 32ch
    - Final Output: Restores clean high-frequency edges on top of moiré-attenuated structural base.
    """
    def __init__(self, base_sigma: float = 3.5):
        super().__init__()
        self.base_sigma = base_sigma

        # Stage 1: H/8 -> H/4
        self.crb1 = CorrectionRecoveryBlock(128)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Stage 2: H/4 -> H/2
        self.crb2 = CorrectionRecoveryBlock(96)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(112, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Stage 3: H/2 -> H
        self.crb3 = CorrectionRecoveryBlock(64)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Output projection for clean edge and detail synthesis
        self.detail_head = nn.Sequential(
            CorrectionRecoveryBlock(32),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def extract_clean_structural_base(self, x: torch.Tensor) -> torch.Tensor:
        """Attenuates high-frequency periodic moiré ripples from the structural base."""
        sigma = self.base_sigma
        k = int(sigma * 4) | 1
        pad = k // 2
        x_axis = torch.arange(k, dtype=torch.float32, device=x.device) - pad
        g1d = torch.exp(-0.5 * (x_axis / sigma)**2)
        g1d = (g1d / g1d.sum()).view(1, 1, 1, -1).repeat(3, 1, 1, 1)

        smooth = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), g1d, groups=3)
        smooth = F.conv2d(F.pad(smooth, (0, 0, pad, pad), mode="reflect"), g1d.transpose(2, 3), groups=3)
        return smooth

    def forward(
        self,
        f_guided: torch.Tensor,
        skips: dict[str, torch.Tensor],
        base_img: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f_guided: Prior-guided features from attention module (B, 128, H/8, W/8)
            skips: Dictionary with 'f1' (B, 64, H/4, W/4) and 'f0' (B, 64, H/2, W/2)
            base_img: Base reference image (B, 3, H, W)

        Returns:
            i_decoder: Clean reconstructed image with suppressed moiré and crisp details.
        """
        # 1. Progressive feature decoding
        x = self.crb1(f_guided)
        x = self.up1(x)
        x = self.fuse1(torch.cat([x, skips["f1"]], dim=1))

        x = self.crb2(x)
        x = self.up2(x)
        x = self.fuse2(torch.cat([x, skips["f0"]], dim=1))

        x = self.crb3(x)
        x = self.up3(x)

        # Synthesize clean high-frequency detail delta from classification priors
        clean_detail = self.detail_head(x)

        # Combine with moiré-attenuated structural base
        base_clean = self.extract_clean_structural_base(base_img)
        i_decoder = (base_clean + clean_detail).clamp(0.0, 1.0)
        return i_decoder
