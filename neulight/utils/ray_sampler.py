#!/usr/bin/env python3
#
# Created on Wed Apr 23 2025 03:26:06
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from typing import Optional

import torch
from torch import nn


class RaySampler(nn.Module):
    def __init__(
        self,
        num_points_per_ray: int,
        max_distance: float,
    ):
        super().__init__()
        self.num_points_per_ray = num_points_per_ray
        self.max_distance = max_distance

    def forward(
        self,
        rays: torch.Tensor,
        max_distance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample points along rays.

        Args:
            rays (torch.Tensor): Rays to sample points from. Shape: (..., 6)
            max_distance (torch.Tensor): Maximum distance to sample points from. Shape: (..., 1)

        Returns:
            torch.Tensor: Sampled points. Shape: (..., num_points_per_ray, 3)
        """
        device = rays.device
        dtype = rays.dtype

        rays_o = rays[..., :3]  # (..., 3)
        rays_d = rays[..., 3:6]  # (..., 3)

        if max_distance is None:
            max_distance = torch.full(
                (*rays.shape[:-1], 1), self.max_distance, device=device, dtype=dtype
            )  # (..., 1)

        # Replace any infinite far‐planes with the default
        mask_inf = torch.isinf(max_distance)
        if mask_inf.any():
            max_distance = torch.where(
                mask_inf,
                torch.tensor(self.max_distance, device=device, dtype=dtype),
                max_distance,
            )

        # normalized sample positions in [0,1]
        t_steps = torch.linspace(
            0.0,
            1.0,
            self.num_points_per_ray,
            device=device,
            dtype=dtype,
        )  # (N,)
        # broadcast to (..., N)
        t_vals = max_distance * t_steps  # (..., N)

        # compute points: o + d * t
        #   rays_o.unsqueeze(-2): (..., 1, 3)
        #   rays_d.unsqueeze(-2): (..., 1, 3)
        #   t_vals.unsqueeze(-1): (..., N, 1)
        points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t_vals.unsqueeze(-1)

        return points
