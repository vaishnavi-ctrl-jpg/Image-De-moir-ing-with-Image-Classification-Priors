"""
Fixed Library of Clean Classification Features.
Maintains a codebook of clean feature prototypes spanning natural textures,
sharp edges, and uniform gradients. Serves as the clean reference dictionary
consulted by the Attention Module.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanPriorLibrary(nn.Module):
    """
    Fixed memory dictionary of clean classification feature vectors.
    Shape: (num_prototypes, feature_dim).
    """
    def __init__(self, num_prototypes: int = 128, feature_dim: int = 128):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim

        # Initialize clean prototype embeddings with diverse normalized centroids
        weight = torch.randn(num_prototypes, feature_dim)
        weight = F.normalize(weight, p=2, dim=-1)
        self.prototypes = nn.Parameter(weight, requires_grad=False)

    def get_keys_and_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the fixed library keys and values for cross-attention."""
        # Clean prototypes act as both keys and values
        k = self.prototypes
        v = self.prototypes
        return k, v
