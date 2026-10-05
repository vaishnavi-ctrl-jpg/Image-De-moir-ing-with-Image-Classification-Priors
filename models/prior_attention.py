"""
Component 2: Prior-Consulting Attention Module & Moiré Mask Predictor.
1. Lets encoded features consult the fixed library of clean classification features via cross-attention.
2. Reconstructs prior-guided clean feature representations.
3. Computes feature divergence against the clean library to predict the spatial Moiré Localization Mask M_pred.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .prior_library import CleanPriorLibrary


class PriorAttentionModule(nn.Module):
    def __init__(self, in_channels: int = 128, num_prototypes: int = 128, num_heads: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        # Fixed clean classification prior library
        self.library = CleanPriorLibrary(num_prototypes=num_prototypes, feature_dim=in_channels)

        # Cross-attention projections
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Layer normalization & residual fusion
        self.norm = nn.GroupNorm(8, in_channels)

        # Moiré Localization Mask Predictor Head
        # Analyzes discrepancy between encoded features and clean prior retrieval
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        f_enc: torch.Tensor,
        target_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f_enc: Encoded classification features (B, C, H_f, W_f)
            target_size: (H, W) original image spatial resolution for the mask

        Returns:
            f_guided: Prior-guided clean features (B, C, H_f, W_f)
            m_pred: Predicted moiré localization mask (B, 1, H, W) in [0, 1]
            attn_weights: Attention affinity matrix (B, H_f*W_f, num_prototypes)
        """
        B, C, H_f, W_f = f_enc.shape
        N = H_f * W_f

        # 1. Prepare Queries from image features
        q = self.q_proj(f_enc) # (B, C, H_f, W_f)
        q = q.flatten(2).permute(0, 2, 1) # (B, N, C)

        # 2. Prepare Keys & Values from fixed clean library
        proto_k, proto_v = self.library.get_keys_and_values() # (K, C)
        k = self.k_proj(proto_k).unsqueeze(0).expand(B, -1, -1) # (B, K, C)
        v = self.v_proj(proto_v).unsqueeze(0).expand(B, -1, -1) # (B, K, C)

        # Multi-head reshaping
        # (B, num_heads, N, head_dim)
        q_heads = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, num_heads, K, head_dim)
        k_heads = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Cross Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale # (B, heads, N, K)
        attn = F.softmax(scores, dim=-1)

        # Retrieve clean prior representations
        guided_heads = torch.matmul(attn, v_heads) # (B, heads, N, head_dim)
        guided = guided_heads.transpose(1, 2).contiguous().view(B, N, C) # (B, N, C)
        guided = guided.permute(0, 2, 1).view(B, C, H_f, W_f)
        f_prior = self.out_proj(guided)

        # Residual gating & feature enhancement
        f_guided = self.norm(f_enc + f_prior)

        # 4. Predict Moiré Localization Mask from Divergence
        # Features that deviate strongly from clean prototypes represent moiré artifacts
        discrepancy = torch.abs(f_enc - f_prior)
        combined = torch.cat([discrepancy, f_enc], dim=1) # (B, 2*C, H_f, W_f)

        mask_lowres = self.mask_head(combined) # (B, 1, H_f, W_f)
        # Bilinear upsample to full image resolution
        m_pred = F.interpolate(mask_lowres, size=target_size, mode="bilinear", align_corners=False)

        # Mean attention weights across heads for visualization
        attn_vis = attn.mean(dim=1) # (B, N, K)

        return f_guided, m_pred, attn_vis
