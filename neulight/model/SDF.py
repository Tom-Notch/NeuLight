#!/usr/bin/env python3
#
# Created on Wed Apr 23 2025 01:55:40
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


class SDF(nn.Module):
    def __init__(
        self,
        num_harmonic_levels: int,
        hidden_dims: list[int],
        skip_connections: bool = True,
        activation: str = "ReLU",
    ):
        """
        A simple SDF network: 𝑓: ℝ³ → ℝ

        Args:
          hidden_dims:   list of hidden layer widths (must all equal if skip_connections)
          skip_connections: whether to use residual blocks
          activation:    name of Torch activation class, e.g. "ReLU", "LeakyReLU"
        """
        super().__init__()

        harmonic_embedding = HarmonicEmbedding(
            in_channels=3,
            L=num_harmonic_levels,
        )

        self.network = nn.Sequential(
            harmonic_embedding,
            MLP(
                in_channels=harmonic_embedding.out_channels,
                out_channels=1,
                hidden_dims=hidden_dims,
                activation=activation,
                skip_connections=skip_connections,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (..., 3)

        Returns:
            (torch.Tensor): (..., 1) signed distance values
        """
        return self.network(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇ₓ SDF(x) for eikonal loss.
        Returns a tensor of shape (..., 3).

        Args:
            x (torch.Tensor): (..., 3)

        Returns:
            grads (torch.Tensor): (..., 3)
        """
        has_grad = torch.is_grad_enabled()

        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            y = self(x)

            # grad_outputs ones ensures ∂y/∂x has the same shape
            grads = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=torch.ones_like(y),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True,
            )[0]

        return grads

    @torch.no_grad()
    def sphere_trace(
        self,
        rays: torch.Tensor,
        num_steps: int = 100,
        epsilon: float = 1e-4,
        max_distance: float = 40.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Basic sphere-tracing: march along each ray until |SDF|<epsilon.

        Args:
            rays (torch.Tensor): (N, 6)
            num_steps (int): max iterations
            epsilon (float): stop threshold on |SDF|
            max_distance (float): clamp total travel distance

        Returns:
            hit_pts (torch.Tensor): (N,3) final sample positions
            mask (torch.Tensor): (N,) bool, True if converged within epsilon
        """
        N = rays.shape[0]
        device = rays.device

        origins = rays[:, :3]
        directions = rays[:, 3:]

        # current travel distance along ray
        t = torch.zeros(N, device=device)

        # Boolean mask for rays that have converged (hit the surface)
        converged = torch.zeros((N,), dtype=torch.bool, device=device)

        for _ in range(num_steps):
            points = origins + t.unsqueeze(-1) * directions

            signed_distance = self(points).squeeze(-1)  # (N,)

            # Determine which active rays have reached the surface (within eps)
            hit = signed_distance.abs() < epsilon

            # Only update the mask for rays that are still active (haven't converged and t < max_distance)
            active = (t < max_distance) & (~converged)
            # For active rays, if they now satisfy the hit condition, mark them as converged.
            converged = converged | (active & hit)

            # mark those now within epsilon
            converged_now = signed_distance.abs() < epsilon
            converged = converged | converged_now

            t = torch.where(active & (~hit), t + signed_distance, t)
            t = torch.clamp(t, min=0, max=max_distance)

            # early exit if all converged
            if (~(converged | (t >= max_distance))).sum() == 0:
                break

        # Final estimated intersection points.
        final_points = origins + t.unsqueeze(-1) * directions

        return final_points, converged
