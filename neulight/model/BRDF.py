#!/usr/bin/env python3
#
# Created on Wed Apr 23 2025 01:55:47
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from neulight.model.harmonic import HarmonicEmbedding
from neulight.model.MLP import MLP


class BRDF(nn.Module):
    def __init__(
        self,
        num_positional_levels: int,
        num_normal_levels: int,
        num_directional_levels: int,
        hidden_dims: list[int],
        skip_connections: bool = True,
        activation: str = "ReLU",
    ):
        super().__init__()
        # embeddings
        self.positional_embedding = HarmonicEmbedding(
            3, L=num_positional_levels, include_x=True
        )
        self.normal_embedding = HarmonicEmbedding(
            3, L=num_normal_levels, include_x=True
        )
        self.directional_embedding = HarmonicEmbedding(
            3, L=num_directional_levels, include_x=True
        )

        in_channels = (
            self.positional_embedding.out_channels
            + self.normal_embedding.out_channels
            + 2 * self.directional_embedding.out_channels
        )
        # MLP + sigmoid
        self.net = nn.Sequential(
            MLP(
                in_channels,
                out_channels=3,
                hidden_dims=hidden_dims,
                activation=activation,
                skip_connections=skip_connections,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,  # (..., 3)
        n: torch.Tensor,  # (..., 3)
        wi: torch.Tensor,  # (..., 3)
        wo: torch.Tensor,  # (..., 3)
    ) -> torch.Tensor:
        p_e = self.positional_embedding(x)
        n_e = self.normal_embedding(n)
        wi_e = self.directional_embedding(wi)
        wo_e = self.directional_embedding(wo)
        h = torch.cat([p_e, n_e, wi_e, wo_e], dim=-1)
        return self.net(h)  # (...,3) RGB reflectance in [0,1]
