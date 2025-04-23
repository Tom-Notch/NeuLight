#!/usr/bin/env python
#
# Created on Wed Apr 23 2025 00:50:56
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict
from typing import Any
from typing import List
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential

from neulight.model.emission import Emission
from neulight.model.SDF import SDF
from neulight.utils.ray_sampler import RaySampler
from neulight.utils.torch_numpy import detect_nan


class NeuSLightningModel(pl.LightningModule):
    def __init__(
        self,
        config: dict[str],
    ):
        super().__init__()
        self.config = config

        self.SDF = SDF(**config["SDF"])
        self.emission = Emission(**config["emission"])
        self.ray_sampler = RaySampler(**config["ray_sampler"])
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.eikonal_loss_weight = config["eikonal_loss_weight"]
        self.normalize_factor = config["normalize_factor"]

        self.register_buffer("geometry_bound", torch.Tensor(config["geometry_bound"]))

        if config.get("checkpoint", None) is not None:
            self.load_state_dict(config["checkpoint"]["state_dict"])

    @staticmethod
    def sample_random_points(
        num_points: int,
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        min_bound = bounds[:3].unsqueeze(0)
        max_bound = bounds[3:].unsqueeze(0)

        return (
            torch.rand((num_points, 3), device=bounds.device) * (max_bound - min_bound)
            + min_bound
        )

    @staticmethod
    def sdf_to_density(
        sdf: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Convert signed distance to density.

        Args:
            sdf (torch.Tensor): Signed distance to the surface. Shape: (..., 1)
            alpha (torch.Tensor): Alpha value.
            beta (torch.Tensor): Beta value.

        Returns:
            torch.Tensor: Density. Shape: (..., 1)
        """
        # signed_distance = + outside, – inside

        s = -sdf
        return alpha * torch.where(
            s <= 0,
            0.5 * torch.exp(s / beta),
            1 - 0.5 * torch.exp(-s / beta),
        )

    def forward(self, batch: dict) -> dict:
        rays = batch["rays"]  # (..., 6)
        device = rays.device

        points = self.ray_sampler(rays)  # (..., num_points_per_ray, 3)
        normalized_points = points / self.normalize_factor

        detect_nan(normalized_points, "normalized points")

        sdf = self.SDF(normalized_points)  # (..., num_points_per_ray, 1)

        detect_nan(sdf, "sdf")

        density = self.sdf_to_density(
            sdf, self.alpha, self.beta
        )  # (..., num_points_per_ray, 1)

        detect_nan(density, "density")

        directions = (
            rays[..., 3:].unsqueeze(-2).expand_as(normalized_points)
        )  # (..., num_points_per_ray, 3)

        detect_nan(directions, "directions")

        rgb = self.emission(
            normalized_points, directions
        )  # (..., num_points_per_ray, 3)

        detect_nan(rgb, "rgb")

        delta = torch.norm(
            normalized_points[..., 1:, :] - normalized_points[..., :-1, :],
            dim=-1,
            keepdim=True,
        )  # (..., num_points_per_ray - 1, 1)
        delta = torch.cat(
            [delta, torch.tensor([1e10], device=device).expand_as(delta[..., :1, :])],
            dim=-2,
        )  # (..., num_points_per_ray, 1)

        detect_nan(delta, "delta")

        exponents = density * delta  # (..., num_points_per_ray, 1)

        detect_nan(exponents, "exponents")

        cumulative_exponents = torch.cumsum(
            exponents, dim=-2
        )  # (..., num_points_per_ray, 1)
        shifted_cumulative_exponents = torch.cat(
            [
                torch.zeros_like(exponents[..., :1, :], device=device),
                cumulative_exponents,
            ],
            dim=-2,
        )  # prepend 0 since T = 1 for the first segments, shape: (..., num_points_per_ray + 1, 1)

        detect_nan(shifted_cumulative_exponents, "shifted cumulative exponents")

        transmittance = torch.exp(
            -shifted_cumulative_exponents
        )  # (..., num_points_per_ray + 1, 1)

        detect_nan(transmittance, "transmittance")

        alpha = 1 - torch.exp(-exponents)  # (..., num_points_per_ray, 1)

        detect_nan(alpha, "alpha")

        weights = transmittance[..., :-1, :] * alpha  # (..., num_points_per_ray, 1)

        detect_nan(weights, "weights")

        colors = (weights * rgb).sum(dim=-2)  # (..., 3)

        detect_nan(colors, "predicted colors")

        return {"colors": colors}

    def loss(
        self,
        predict_colors: torch.Tensor,
        gt_colors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss function for panoramic semantic segmentation.

        Args:
            predict_colors (torch.Tensor): Predicted colors, batch_value of shape (B, N, 3)
            gt_colors (torch.Tensor): Ground truth colors, batch_value of shape (B, N, 3)

        Returns:
            torch.Tensor: The loss.
        """
        photometric_loss = F.mse_loss(predict_colors, gt_colors)

        num_points = predict_colors.numel() // 3
        random_points = self.sample_random_points(
            num_points=num_points,
            bounds=self.geometry_bound,
        )
        detect_nan(random_points, "random points")
        gradients = self.SDF.gradient(random_points)
        detect_nan(gradients, "SDF gradients")
        eikonal_loss = ((gradients.norm(dim=-1) - 1.0) ** 2).mean()

        loss = photometric_loss + self.eikonal_loss_weight * eikonal_loss

        return loss

    def training_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        batch_size = batch["inputs"]["rays"].shape[0]

        batch["predicts"] = self(batch["inputs"])

        loss = self.loss(
            predict_colors=batch["predicts"]["colors"],
            gt_colors=batch["labels"]["colors"],
        )
        loss = loss.contiguous()

        with torch.no_grad():
            self.log(
                "train loss",
                loss,
                prog_bar=True,
                logger=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        return loss

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        dataset = self.trainer.val_dataloaders.dataset

        # visualize ground truth and predicts
        if batch_idx == 0 and self.logger is not None:
            # chunk the rays
            rays = batch["inputs"]["rays"]
            batch_size = rays.shape[0]
            rays = rays.view(batch_size, -1, 6)
            num_chunks = rays.shape[1] // dataset.num_rays_per_image
            chunks = torch.chunk(rays, num_chunks, dim=1)

            rendered_colors = []
            for chunk in chunks:
                rendered_colors.append(self({"rays": chunk})["colors"])

            batch["predicts"] = {"colors": torch.cat(rendered_colors, dim=1)}

            rendered_images = dataset.visualize_batch(
                batch={
                    "inputs": batch["inputs"],
                    "labels": batch["predicts"],
                },
                num_vis=self.config["num_vis"],
            )

            caption = [
                f"Epoch {self.trainer.current_epoch} image {i}"
                for i in range(len(rendered_images))
            ]
            self.logger.log_image(
                key="Validation Prediction",
                images=rendered_images,
                caption=caption,
            )

        return torch.tensor(0.0)

    def configure_optimizers(self):
        for param in self.parameters():
            param.data = param.data.contiguous()
        return torch.optim.Adam(self.parameters(), lr=self.config["lr"])
