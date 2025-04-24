#!/usr/bin/env python
#
# Created on Wed Apr 23 2025 01:22:01
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch
import torch.nn as nn


class HarmonicEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        L: int,
        basis: float = torch.pi,
        include_x: bool = True,
    ):
        """Harmonic Embedding

        Args:
            in_channels (int): number of input channels.
            L (int): number of harmonic levels.
            basis (float, optional): the basis for the harmonic embedding. Defaults to pi.
            include_x (bool, optional): Whether to prepend the original x. Defaults to True.
        """
        super().__init__()

        assert L >= 0, f"Number of harmonic levels L must be non-negative, got {L}"
        assert basis > 0, f"Harmonic basis must be positive, got {basis}"

        self.in_channels = in_channels
        self.L = L
        self.basis = basis
        self.include_x = include_x

        # Create a vector of factors of shape (L,)
        self.register_buffer(
            "factors",
            (2 ** torch.arange(self.L)) * self.basis,
        )  # (L,)

    @property
    def out_channels(self) -> int:
        return self.in_channels * (int(self.include_x) + 2 * self.L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps vector x to [(x, )sin(2^0*basis*x), ..., sin(2^(L-1)*basis*x), cos(2^0*basis*x), ..., cos(2^(L-1)*basis*x)].

        Args:
            x (torch.Tensor): input of shape (..., D)

        Returns:
            torch.Tensor: shape (..., D x 2 x (L( + 1)))
        """
        assert (
            x.shape[-1] == self.in_channels
        ), f"Input shape must match in_channels, input has {x.shape[-1]} channels, expect {self.in_channels} in_channels"

        if self.L == 0:
            return x if self.include_x else x.new_empty((*x.shape[:-1], 0))

        x = x.unsqueeze(-1)  # shape (..., D, 1)

        embed_list = [x] if self.include_x else []  # shape (..., D, 1)

        # Multiply x (shape: (..., D)) by factors via broadcasting -> (..., D, L)
        x_expanded = x * self.factors  # shape: (..., D, L)

        embed_list.extend([torch.sin(x_expanded), torch.cos(x_expanded)])

        embed = torch.cat(embed_list, dim=-1)  # shape (..., D, (1 +) 2*L)

        return embed.flatten(start_dim=-2)  # shape (..., D (x 2 x (L( + 1))))
