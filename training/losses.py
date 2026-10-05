"""
Loss Suite for Image De-moiréing with Classification Priors.
Implements:
1. L1 Spatial Pixel Loss
2. 2D FFT Frequency-Domain Loss (Focal Frequency Loss preventing periodic ripple leakage)
3. BCE + Dice Mask Supervision Loss against Teacher Detector
4. Decoder Auxiliary Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalFrequencyLoss(nn.Module):
    """
    Penalizes frequency discrepancies between restored and clean images in 2D Fourier domain.
    Directly suppresses periodic moiré harmonic spikes.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 2D Real FFT
        fft_pred = torch.fft.rfft2(pred, norm="ortho")
        fft_target = torch.fft.rfft2(target, norm="ortho")

        # Frequency distance in complex plane
        diff = torch.abs(fft_pred - fft_target)

        # Dynamic frequency weight: focus heavily on hard-to-restore high/mid frequency components
        weight = torch.abs(fft_target) ** self.alpha
        weight = weight / weight.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-4)

        loss = torch.mean(diff * (1.0 + weight))
        return loss


class DiceLoss(nn.Module):
    """
    Dice loss for spatial mask segmentation to avoid mask collapse.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_flat = pred.contiguous().view(-1)
        t_flat = target.contiguous().view(-1)
        intersection = (p_flat * t_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (p_flat.sum() + t_flat.sum() + self.smooth)
        return 1.0 - dice


class DemoireCompositeLoss(nn.Module):
    """
    Comprehensive loss function combining spatial, frequency, and mask supervision.
    """
    def __init__(
        self,
        weight_l1: float = 1.0,
        weight_fft: float = 0.20,
        weight_mask: float = 0.50,
        weight_decoder: float = 0.25,
    ):
        super().__init__()
        self.weight_l1 = weight_l1
        self.weight_fft = weight_fft
        self.weight_mask = weight_mask
        self.weight_decoder = weight_decoder

        self.l1_loss = nn.L1Loss()
        self.fft_loss = FocalFrequencyLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(
        self,
        pipeline_out: dict[str, torch.Tensor],
        clean_target: torch.Tensor,
        mask_teacher: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        restored = pipeline_out["restored"]
        decoder_out = pipeline_out["decoder_out"]
        mask_pred = pipeline_out["mask_pred"]

        # 1. Pixel L1 loss
        loss_l1 = self.l1_loss(restored, clean_target)

        # 2. 2D FFT Frequency Domain loss (kills ripple leakage)
        loss_fft = self.fft_loss(restored, clean_target)

        # 3. Supervised Mask loss (BCE + Dice against teacher detector)
        loss_bce = self.bce_loss(mask_pred.clamp(1e-6, 1.0 - 1e-6), mask_teacher)
        loss_dice = self.dice_loss(mask_pred, mask_teacher)
        loss_mask = loss_bce + loss_dice

        # 4. Decoder auxiliary loss
        loss_dec = self.l1_loss(decoder_out, clean_target)

        # Total combined loss
        loss_total = (
            self.weight_l1 * loss_l1
            + self.weight_fft * loss_fft
            + self.weight_mask * loss_mask
            + self.weight_decoder * loss_dec
        )

        return {
            "loss_total": loss_total,
            "loss_l1": loss_l1,
            "loss_fft": loss_fft,
            "loss_mask": loss_mask,
            "loss_dec": loss_dec,
        }
