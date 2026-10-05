"""
Synthetic Moiré Pattern Generator.
Simulates realistic screen-photo moiré artifacts:
1. Multi-frequency luminance interference (beating between display pixel pitch and camera sensor grid).
2. Chromatic phase shifts across R, G, B channels creating rainbow fringes.
3. Spatially localized moiré masks with non-uniform perspective warp and smooth boundary decay.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class SyntheticMoireGenerator:
    """
    Generates realistic, physically-grounded screen photo moiré patterns
    superimposed on clean images, along with true moiré contamination masks.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_moire_pattern(
        self,
        height: int,
        width: int,
        frequencies: list = None,
        angles: list = None,
        chromatic_shift: float = 0.35,
        warp_intensity: float = 0.08,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates a multi-component moiré ripple pattern and its true spatial mask.

        Returns:
            moire_pattern: (H, W, 3) float32 in [-1.0, 1.0] representing additive/multiplicative ripple.
            spatial_mask: (H, W) float32 in [0.0, 1.0] indicating moiré severity per pixel.
        """
        if frequencies is None:
            # Multi-scale frequencies typical of screen photos
            frequencies = [
                self.rng.uniform(0.04, 0.12),  # Mid-high frequency ripples
                self.rng.uniform(0.015, 0.04), # Broad wavy bands
            ]
        if angles is None:
            base_angle = self.rng.uniform(0, math.pi)
            angles = [base_angle, base_angle + self.rng.uniform(0.2, 0.6)]

        # Generate coordinate grid
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)

        # Apply slight non-linear perspective warp (screen curvature / lens distortion)
        x_norm = (x / width) * 2 - 1
        y_norm = (y / height) * 2 - 1
        r_sq = x_norm**2 + y_norm**2
        x_warped = x + warp_intensity * width * (x_norm * r_sq)
        y_warped = y + warp_intensity * height * (y_norm * r_sq)

        # Multi-frequency interference field
        interference_r = np.zeros((height, width), dtype=np.float32)
        interference_g = np.zeros((height, width), dtype=np.float32)
        interference_b = np.zeros((height, width), dtype=np.float32)

        for freq, angle in zip(frequencies, angles):
            kx = freq * np.cos(angle)
            ky = freq * np.sin(angle)
            phase = kx * x_warped + ky * y_warped

            # Chromatic phase shift (subpixels R, G, B are offset in physical screen layout)
            shift = chromatic_shift * math.pi
            interference_r += 0.5 * np.sin(phase - shift)
            interference_g += 0.5 * np.sin(phase)
            interference_b += 0.5 * np.sin(phase + shift)

        # Localized spatial mask: moiré is concentrated in specific regions (e.g., center or diagonal strip)
        # using a mixture of 2D Gaussians or smooth sigmoids
        cx = self.rng.uniform(0.35, 0.65) * width
        cy = self.rng.uniform(0.35, 0.65) * height
        sigma_x = self.rng.uniform(0.25, 0.45) * width
        sigma_y = self.rng.uniform(0.25, 0.45) * height

        dist = ((x - cx) ** 2) / (2 * sigma_x**2) + ((y - cy) ** 2) / (2 * sigma_y**2)
        spatial_mask = np.exp(-dist).astype(np.float32)
        # Normalize mask to [0, 1] with smooth sigmoid thresholding
        spatial_mask = np.clip((spatial_mask - 0.1) / 0.8, 0.0, 1.0)

        # Stack into (H, W, 3)
        moire_pattern = np.stack([interference_r, interference_g, interference_b], axis=-1)
        # Scale pattern by the spatial contamination mask
        moire_pattern = moire_pattern * spatial_mask[:, :, None]

        return moire_pattern, spatial_mask

    def apply_moire(
        self,
        clean_img: np.ndarray,
        intensity: float = 0.45,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies synthetic moiré pattern onto a clean image (H, W, 3) float32 in [0, 1].

        Returns:
            corrupted_img: (H, W, 3) float32 in [0, 1]
            mask: (H, W) float32 in [0, 1]
        """
        H, W, C = clean_img.shape
        pattern, mask = self.generate_moire_pattern(H, W)

        # Blend pattern: both additive ripples and multiplicative modulation
        additive = pattern * intensity
        multiplicative = 1.0 + (pattern * 0.25 * intensity)

        corrupted = (clean_img * multiplicative) + additive
        corrupted = np.clip(corrupted, 0.0, 1.0).astype(np.float32)

        return corrupted, mask

    def create_synthetic_pair_tensor(
        self,
        clean_tensor: torch.Tensor,
        intensity: float = 0.45
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given clean tensor (B, 3, H, W) in [0, 1], returns (corrupted_tensor, clean_tensor, mask_tensor).
        """
        B, C, H, W = clean_tensor.shape
        corrupted_list = []
        mask_list = []

        for b in range(B):
            clean_np = clean_tensor[b].permute(1, 2, 0).cpu().numpy()
            corr_np, mask_np = self.apply_moire(clean_np, intensity=intensity)

            corrupted_list.append(torch.from_numpy(corr_np).permute(2, 0, 1))
            mask_list.append(torch.from_numpy(mask_np).unsqueeze(0))

        corrupted_batch = torch.stack(corrupted_list).to(clean_tensor.device)
        mask_batch = torch.stack(mask_list).to(clean_tensor.device)

        return corrupted_batch, clean_tensor, mask_batch


def generate_benchmark_pattern(height: int = 256, width: int = 256) -> np.ndarray:
    """Creates a synthetic clean benchmark image (screen content: text, boxes, gradients)."""
    img = np.ones((height, width, 3), dtype=np.float32) * 0.95
    # Add colored header
    img[:40, :] = [0.12, 0.35, 0.72]
    # Add dark text-like bars
    for i in range(55, height - 30, 20):
        w_bar = int(width * (0.4 + 0.4 * np.sin(i)))
        img[i:i+6, 20:20+w_bar] = 0.15
    # Add a soft photographic gradient patch
    y, x = np.mgrid[0:80, 0:80]
    grad = (x + y) / 160.0
    img[120:200, width-100:width-20, 0] = grad * 0.9
    img[120:200, width-100:width-20, 1] = (1 - grad) * 0.8
    img[120:200, width-100:width-20, 2] = 0.6
    return img
