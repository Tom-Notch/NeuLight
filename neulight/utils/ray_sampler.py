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
        perturb: bool = True,
    ):
        """
        Samples N points along each ray, optionally perturbing within each bin.

        Args:
          num_points_per_ray: how many samples per ray
          max_distance:       far bound (world units)
          perturb:            if True, randomly jitter samples inside each bin;
                              if False, use the exact bin centers.
        """
        super().__init__()
        self.num_points_per_ray = num_points_per_ray
        self.max_distance = max_distance
        self.perturb = perturb

        # Precompute the N+1 edges of the [0,1] interval
        edges = torch.linspace(0.0, 1.0, num_points_per_ray + 1)
        self.register_buffer("t_lower", edges[:-1])  # shape (N,)
        self.register_buffer("t_upper", edges[1:])  # shape (N,)

    def forward(
        self,
        rays: torch.Tensor,
        max_distance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          rays:         (...,6) last dim=(ox,oy,oz, dx,dy,dz)
          max_distance: (...,1) optional per-ray far bound
        Returns:
          pts: (...,N,3) in world units
        """
        device = rays.device
        dtype = rays.dtype

        o = rays[..., :3]  # (...,3)
        d = rays[..., 3:]  # (...,3)

        # build per-ray far bound
        if max_distance is None:
            max_d = torch.full(
                (*o.shape[:-1], 1),
                self.max_distance,
                device=device,
                dtype=dtype,
            )
        else:
            max_d = max_distance.to(device=device, dtype=dtype)

        # clamp any infs
        inf = torch.isinf(max_d)
        if inf.any():
            max_d = torch.where(
                inf,
                torch.tensor(self.max_distance, device=device, dtype=dtype),
                max_d,
            )

        # generate t in [0,1]
        shape = (*o.shape[:-1], self.num_points_per_ray)
        if self.perturb:
            # stratified with random jitter
            u = torch.rand(shape, device=device, dtype=dtype)
            t = (
                self.t_lower.view(*(1,) * (o.ndim - 1), -1)
                + (self.t_upper - self.t_lower).view(*(1,) * (o.ndim - 1), -1) * u
            )
        else:
            # exactly the centers of each bin
            mids = 0.5 * (self.t_lower + self.t_upper)
            t = mids.view(*(1,) * (o.ndim - 1), -1).expand(shape)

        # scale to [0, far]
        t_world = max_d * t  # (...,N)

        # sample points
        pts = o.unsqueeze(-2) + d.unsqueeze(-2) * t_world.unsqueeze(-1)

        return pts
