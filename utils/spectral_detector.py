"""
Spectral and Chrominance Moiré Teacher Detector.
Acts as a fixed algorithmic teacher providing reliable pseudo-ground-truth supervision
for moiré localization masks M_gt.
Uses:
1. Local high-frequency ripple energy in chrominance and luminance.
2. 2D FFT spectral energy diagnostics.
"""

import numpy as np
import torch
import torch.nn.functional as F


class SpectralMoireDetector:
    """
    Detects moiré patterns using physics-based signal properties:
    - High-frequency chrominance and luminance ripple energy
    - 2D Fourier domain periodic peak energy
    """

    def __init__(self, kernel_size: int = 15):
        self.kernel_size = kernel_size

    def rgb_to_ycbcr_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        """Converts RGB tensor (B, 3, H, W) in [0, 1] to YCbCr."""
        r = rgb[:, 0:1, :, :]
        g = rgb[:, 1:2, :, :]
        b = rgb[:, 2:3, :, :]

        y  =  0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        return torch.cat([y, cb, cr], dim=1)

    def detect_moire_mask(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generates reliable algorithmic moiré mask in [0, 1] based on high-frequency ripple energy.
        """
        k = self.kernel_size
        pad = k // 2
        # Reflect padded low-pass filter
        low = F.avg_pool2d(F.pad(rgb_tensor, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
        ripple = torch.abs(rgb_tensor - low).mean(dim=1, keepdim=True)

        # Spatial smoothing to get regional continuous mask
        pooled = F.avg_pool2d(F.pad(ripple, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
        max_v = pooled.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-4)
        mask = (pooled / max_v).clamp(0.0, 1.0)
        return mask

    def measure_spectral_peak_energy(self, rgb_tensor: torch.Tensor) -> float:
        """
        Calculates peak high-frequency energy ratio in the 2D FFT spectrum using a 2D Hann window.
        """
        ycbcr = self.rgb_to_ycbcr_torch(rgb_tensor)
        gray = ycbcr[:, 0, :, :].cpu().numpy()
        energies = []
        for b in range(gray.shape[0]):
            H, W = gray[b].shape
            hann = np.outer(np.hanning(H), np.hanning(W))
            f = np.fft.fftshift(np.fft.fft2(gray[b] * hann))
            mag = np.abs(f)
            # Mask out central DC component
            mag[H//2-4:H//2+5, W//2-4:W//2+5] = 0
            peak = float(np.max(mag))
            energies.append(peak)
        return float(np.mean(energies))
