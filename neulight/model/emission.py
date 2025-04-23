#!/usr/bin/env python3
#
# Created on Wed Apr 23 2025 01:55:55
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


class Emission(nn.Module):
    def __init__(
        self,
        num_positional_levels: int,
        num_directional_levels: int,
        hidden_dims: list[int],
        skip_connections: bool = True,
        activation: str = "ReLU",
    ):
        """
        A view-dependent radiance (color) network for NeRF/NeuS.

        Args:
            num_positional_levels (int): number of harmonic levels for position embedding.
            num_directional_levels (int): number of harmonic levels for direction embedding.
            hidden_dims (list[int]): widths of hidden layers in the MLP.
            skip_connections (bool): whether to use residual blocks. If skip_connections=True, all hidden_dims must be equal.
            activation (str): name of nn activation (e.g. "ReLU", "LeakyReLU").
        """
        super().__init__()

        # 1) positional embedding for the surface point
        self.positional_embedding = HarmonicEmbedding(
            in_channels=3,
            L=num_positional_levels,
            include_x=True,
        )

        # 2) directional embedding for the view vector
        self.directional_embedding = HarmonicEmbedding(
            in_channels=3,
            L=num_directional_levels,
            include_x=True,
        )

        # 3) an MLP to map [pos_emb ⧺ dir_emb] -> RGB
        self.rgb_mlp = nn.Sequential(
            MLP(
                in_channels=self.positional_embedding.out_channels
                + self.directional_embedding.out_channels,
                out_channels=3,
                hidden_dims=hidden_dims,
                activation=activation,
                skip_connections=skip_connections,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        points: torch.Tensor,
        view_directions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict color for each 3D point + view direction.

        Args:
            points (torch.Tensor): (..., 3) surface points in world coords.
            view_directions (torch.Tensor): (..., 3) normalized view directions (camera→point).

        Returns:
            rgb (torch.Tensor): (..., 3)
        """
        # 1) embed and flatten
        positional_embedding = self.positional_embedding(points)  # (..., P)
        directional_embedding = self.directional_embedding(view_directions)  # (..., D)

        # 2) concatenate and run through MLP
        embedding = torch.cat(
            [positional_embedding, directional_embedding], dim=-1
        )  # (..., P+D)
        rgb = self.rgb_mlp(embedding)  # (..., 3)

        return rgb
