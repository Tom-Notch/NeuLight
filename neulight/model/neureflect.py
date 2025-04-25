#!/usr/bin/env python3
#
# Created on Wed Apr 23 2025 01:55:30
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import math
from collections import OrderedDict
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential

from neulight.model.BRDF import BRDF
from neulight.model.neus import NeuSLightningModel
from neulight.utils.ray_sampler import RaySampler
from neulight.utils.torch_numpy import detect_nan


class NeuReflectLightningModel(pl.LightningModule):
    def __init__(
        self,
        config: dict[str],
    ):
        super().__init__()
        self.config = config

        self.neus = NeuSLightningModel(config["neus"]).eval()
        self.BRDF = BRDF(**config["BRDF"])

        self.surface_render_offset = config["surface_render_offset"]
        self.num_incident_rays = config["num_incident_rays"]
        self.query_color_chunk_size = config["query_color_chunk_size"]

        self.normalize_factor = config["normalize_factor"]

        if config.get("checkpoint", None) is not None:
            self.load_state_dict(config["checkpoint"]["state_dict"])

        for param in self.neus.parameters():
            param.requires_grad = False

    @staticmethod
    @torch.no_grad()
    def sample_incident_rays(points, normals, num_samples):
        """
        Cosine-weighted hemisphere sampling around each normal.
        Returns:
          rays: (M, K, 6)   world-space rays = (ox,oy,oz, dx,dy,dz)
          pdf:  (M, K)      cos-hemisphere PDF = cosθ/π
        where M = points.numel()/3, K = num_samples.
        """
        device, dtype = points.device, points.dtype
        M = points.shape[0]
        K = num_samples

        # expand
        o = points.unsqueeze(1).expand(M, K, 3)
        n = normals.unsqueeze(1).expand(M, K, 3)

        # two uniforms
        u1 = torch.rand(M, K, device=device, dtype=dtype)
        u2 = torch.rand_like(u1)
        r = torch.sqrt(u1)
        phi = 2 * torch.pi * u2

        # local hemisphere
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = torch.sqrt(1 - u1)
        wi_local = torch.stack([x, y, z], dim=-1)  # (M,K,3)

        # build T,B
        helper = (
            torch.tensor([0, 0, 1], device=device, dtype=dtype)
            .view(1, 1, 3)
            .expand_as(n)
        )
        dot = (n * helper).sum(-1, keepdim=True)
        alt = (
            torch.tensor([1, 0, 0], device=device, dtype=dtype)
            .view(1, 1, 3)
            .expand_as(n)
        )
        helper = torch.where(dot.abs() > 0.99, alt, helper)

        t = torch.cross(helper, n, dim=-1)
        t = F.normalize(t, p=2, dim=-1)
        b = torch.cross(n, t, dim=-1)

        # rotate into world
        dirs = wi_local[..., :1] * t + wi_local[..., 1:2] * b + wi_local[..., 2:3] * n
        dirs = F.normalize(dirs, p=2, dim=-1)

        # pdf = cosθ/π
        pdf = wi_local[..., 2] / torch.pi

        # pack rays
        rays = torch.cat([o, dirs], dim=-1)  # (M,K,6)
        return rays, pdf

    def chunk_forward(
        self, rays: torch.Tensor, chunk_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """chunked forward

        Args:
            rays (torch.Tensor): in shape (..., 6)

        Returns:
            torch.Tensor: colors in shape (..., 3), masks in shape (...)
        """
        _rays = rays.view(-1, 6)

        # chunk the rays
        num_chunks = max(1, math.ceil(_rays.shape[0] // chunk_size))
        ray_chunks = torch.chunk(_rays, num_chunks, dim=0)

        rendered_colors = []
        masks = []
        for ray_chunk in ray_chunks:
            color, mask = self(ray_chunk)
            rendered_colors.append(color)
            masks.append(mask)

        colors = torch.cat(rendered_colors, dim=0).view(*rays.shape[:-1], 3)
        masks = torch.cat(masks, dim=0).view(*rays.shape[:-1])

        return colors, masks

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        _rays = rays.view(-1, 6)  # (N, 6)
        device = rays.device

        with torch.no_grad():
            # sphere tracing to get hit points
            hit_points, hit_masks = self.neus.sphere_tracing(_rays)  # (N, 3), (N,)

            detect_nan(hit_points, "hit_points")
            detect_nan(hit_masks, "hit_masks")

        # extract normals
        normals = torch.zeros_like(hit_points, device=device)  # (N, 3)
        normals[hit_masks] = self.neus.normal(hit_points[hit_masks])  # (N', 3)

        detect_nan(normals, "normals")

        with torch.no_grad():
            # offset the hit_points a bit along normals to avoid self shadowing
            shifted_hit_points = (
                hit_points + normals * self.surface_render_offset
            )  # (N, 3)

            # monte carlo sampling
            incident_rays, incident_pdf = self.sample_incident_rays(
                shifted_hit_points[hit_masks],
                normals[hit_masks],
                self.num_incident_rays,
            )  # (N', K, 6), (N', K)

            detect_nan(incident_rays, "incident_rays")
            detect_nan(incident_pdf, "incident_pdf")

            # incident color
            incident_colors = self.neus.chunk_forward(
                incident_rays,
                self.query_color_chunk_size,
            )  # (N', K, 3)

            detect_nan(incident_colors, "incident_colors")

        valid_hit_points = (
            hit_points[hit_masks].unsqueeze(-2).expand(-1, self.num_incident_rays, 3)
        )  # (N', K, 3)
        normalized_valid_hit_points = valid_hit_points / self.normalize_factor
        valid_normals = (
            normals[hit_masks].unsqueeze(-2).expand(-1, self.num_incident_rays, 3)
        )  # (N', K, 3)
        valid_incident_directions = incident_rays[..., 3:]  # (N', K, 3)
        valid_outgoing_directions = (
            _rays[hit_masks, 3:].unsqueeze(-2).expand(-1, self.num_incident_rays, 3)
        )  # (N', K, 3)

        reflectance = self.BRDF(
            normalized_valid_hit_points,
            valid_incident_directions,
            valid_outgoing_directions,
        )  # (N', num_incident_rays, 3)

        detect_nan(reflectance, "reflectance")

        # 4) Monte Carlo one-bounce: Lo = E[ fr * Li * (n·wi) / pdf ]
        cos_theta = (
            (valid_normals * valid_incident_directions)
            .sum(-1, keepdim=True)
            .clamp(min=0)
        )  # (N', K, 1)
        weights = cos_theta / (incident_pdf.unsqueeze(-1) + 1e-8)  # (N', K, 1)
        outgoing_colors = (reflectance * incident_colors * weights).mean(
            dim=1
        )  # (N', 3)

        detect_nan(cos_theta, "cos_theta")
        detect_nan(weights, "weights")
        detect_nan(outgoing_colors, "outgoing_colors")

        colors = torch.zeros((_rays.shape[0], 3), device=device)  # (N, 3)
        colors[hit_masks] = outgoing_colors

        return colors.view(*rays.shape[:-1], 3), hit_masks.view(*rays.shape[:-1])

    def loss(
        self,
        predict_colors: torch.Tensor,
        gt_colors: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss function for panoramic semantic segmentation.

        Args:
            predict_colors (torch.Tensor): Predicted colors, batch_value of shape (B, N, 3)
            gt_colors (torch.Tensor): Ground truth colors, batch_value of shape (B, N, 3)
            masks (torch.Tensor): in shape (B, N)

        Returns:
            torch.Tensor: The loss.
        """
        photometric_loss = F.mse_loss(predict_colors[masks], gt_colors[masks])

        return photometric_loss

    def training_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        rays = batch["inputs"]["rays"]
        batch_size = rays.shape[0]

        colors, masks = self(rays)
        batch["predicts"] = {"colors": colors, "masks": masks}

        loss = self.loss(
            predict_colors=colors,
            gt_colors=batch["labels"]["colors"],
            masks=masks,
        )

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
            rays = batch["inputs"]["rays"]
            batch_size = rays.shape[0]

            colors, masks = self.chunk_forward(
                rays,
                chunk_size=batch_size * dataset.num_rays_per_image * 2,
            )
            batch["predicts"] = {
                "colors": colors,
                "masks": masks,
            }

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
        for param in self.BRDF.parameters():
            param.data = param.data.contiguous()
        return torch.optim.Adam(self.BRDF.parameters(), lr=self.config["lr"])
