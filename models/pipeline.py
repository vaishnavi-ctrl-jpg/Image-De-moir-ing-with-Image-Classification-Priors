"""
Unified Demoiré Pipeline with Image Classification Priors.
Coordinates:
1. Pretrained Classifier Encoder (Component 1)
2. Prior Attention Module & Fixed Clean Library (Component 2)
3. Multi-scale Reconstruction Decoder (Component 3)
4. Edge-Filtered Direct Guidance Branch (Component 4)
With Soft-Gated Anti-Seam Adaptive Fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier_encoder import ClassifierEncoder
from .prior_attention import PriorAttentionModule
from .reconstruction_decoder import ReconstructionDecoder
from .guidance_branch import EdgeFilteredGuidanceBranch


class DemoirePriorPipeline(nn.Module):
    """
    End-to-End Image De-moiréing Network with Image Classification Priors.
    """
    def __init__(
        self,
        pretrained_encoder: bool = True,
        num_prototypes: int = 128,
        soft_gate_sigma: float = 4.0
    ):
        super().__init__()
        # 1. Component 1: Pre-trained Image Classification Network (Encoder)
        self.encoder = ClassifierEncoder(pretrained=pretrained_encoder)

        # 2. Component 2: Attention Module consulting Fixed Library of Classification Features
        self.attention = PriorAttentionModule(in_channels=128, num_prototypes=num_prototypes)

        # 3. Component 3: Reconstruction Decoder
        self.decoder = ReconstructionDecoder()

        # 4. Component 4: Guidance Branch from input to output
        self.guidance = EdgeFilteredGuidanceBranch(in_channels=3)

        self.soft_gate_sigma = soft_gate_sigma

    def soft_gate_mask(self, mask: torch.Tensor, sigma: float = None) -> torch.Tensor:
        """
        Applies low-pass Gaussian smoothing to the predicted moiré mask
        to ensure zero seams at the boundary between decoder and guidance branch.
        """
        if sigma is None:
            sigma = self.soft_gate_sigma

        kernel_size = int(sigma * 4) | 1
        x = torch.arange(kernel_size, dtype=torch.float32, device=mask.device) - (kernel_size // 2)
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_1d = (gauss_1d / gauss_1d.sum()).view(1, 1, -1, 1)

        # 2D separable Gaussian blur
        smoothed = F.conv2d(mask, gauss_1d, padding=(kernel_size // 2, 0))
        smoothed = F.conv2d(smoothed, gauss_1d.transpose(2, 3), padding=(0, kernel_size // 2))
        return smoothed.clamp(0.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        custom_mask: torch.Tensor = None,
        guidance_strength: float = 1.0
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Input moiré photo (B, 3, H, W) in [0, 1]
            custom_mask: Optional external mask override
            guidance_strength: Scaling factor for guidance branch weight

        Returns:
            Dictionary containing restored image, intermediate maps, and diagnostic statistics.
        """
        B, C, H, W = x.shape

        # Step 1: Pretrained Classification Encoding
        features = self.encoder(x)
        f0, f1, f2 = features["f0"], features["f1"], features["f2"]

        # Step 2: Attention Module consulting Clean Classification Prior Library
        f_guided, m_pred, attn_vis = self.attention(f2, target_size=(H, W))

        # Step 3: Guidance Branch directly from input (filtering chrominance ripples)
        i_guidance = self.guidance(x)

        # Step 4: Reconstruction Decoder using prior features + guidance base
        i_decoder = self.decoder(f_guided, skips={"f0": f0, "f1": f1}, base_img=i_guidance)

        # Step 5: Soft-Gated Anti-Seam Adaptive Fusion
        active_mask = custom_mask if custom_mask is not None else m_pred
        m_soft = self.soft_gate_mask(active_mask)

        # In moiré areas (m_soft -> 1), use decoder; in clean areas (m_soft -> 0), use guidance bypass
        m_blended = m_soft * guidance_strength
        m_blended = m_blended.clamp(0.0, 1.0)

        i_final = m_blended * i_decoder + (1.0 - m_blended) * i_guidance
        i_final = i_final.clamp(0.0, 1.0)

        return {
            "restored": i_final,
            "decoder_out": i_decoder,
            "guidance_out": i_guidance,
            "mask_pred": m_pred,
            "mask_soft": m_soft,
            "attn_weights": attn_vis,
        }
