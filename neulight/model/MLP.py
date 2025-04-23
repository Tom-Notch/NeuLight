#!/usr/bin/env python
#
# Created on Wed Apr 23 2025 01:29:47
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: List[int],
        activation: str,
        skip_connections: bool,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.skip_connections = skip_connections
        activation_function = getattr(nn, activation)

        if skip_connections:
            common_dim = hidden_dims[0]
            assert all(
                dim == common_dim for dim in hidden_dims
            ), "All hidden dimensions must be the same if skip connections are used"

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                nn.Linear(in_channels, hidden_dims[0]),
                activation_function(inplace=True),
            )
        )

        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    activation_function(inplace=True),
                )
            )

        self.layers.append(nn.Linear(hidden_dims[-1], out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_channels)

        Returns:
            torch.Tensor: Output tensor of shape (..., out_channels)
        """
        assert (
            x.shape[-1] == self.in_channels
        ), f"Input tensor must have {self.in_channels} channels, got {x.shape[-1]} channels"

        x = self.layers[0](x)

        if self.skip_connections:
            for layer in self.layers[1:-1]:
                x = layer(x) + x
        else:
            for layer in self.layers[1:-1]:
                x = layer(x)

        x = self.layers[-1](x)

        return x
