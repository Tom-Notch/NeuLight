#!/usr/bin/env python3
#
# Created on Tue Apr 22 2025 23:01:12
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import os.path as osp
from copy import deepcopy
from glob import glob
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from neulight.utils.files import read_file
from neulight.utils.torch_numpy import to_numpy
from neulight.utils.torch_numpy import to_torch

mp.set_sharing_strategy("file_system")


class FreeViewpointDataset(Dataset):
    def __init__(
        self,
        scene_path: str,
        dataset_type: str,
        num_rays_per_image: int = 1024,
        visualize_image_shape: Tuple[int, int] = (480, 640),
    ):
        """Free Viewpoint Dataset

        Args:
            scene_path (str): Path to the scene, should have direct subfolder called scene_share
            dataset_type (str): "train" or "test"
            num_rays_per_image (int, optional): Number of rays to sample from the image. Defaults to 1024.
            visualize_image_shape (Tuple[int, int], optional): Shape of the image in (H, W) to visualize. Defaults to (480, 640).
        """
        assert dataset_type in [
            "train",
            "test",
        ], f"Invalid dataset type: {dataset_type}, must be one of ['train', 'test']"

        self.dataset_type = dataset_type
        self.num_rays_per_image = num_rays_per_image
        self.visualize_image_shape = visualize_image_shape

        self.images = sorted(
            glob(osp.join(scene_path, "scene_share", "images", "*.jpg"))
        )
        self.masks = sorted(
            glob(osp.join(scene_path, "scene_share", "images", "*.png"))
        )
        self.cameras = read_file(
            osp.join(scene_path, "scene_share", "cameras", "bundle.out")
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        if self.dataset_type == "train":
            image = (
                to_torch(read_file(self.images[idx]), dtype=torch.float32) / 255.0
            )  # (H, W, 3)
            mask = to_torch(read_file(self.masks[idx]), dtype=torch.bool)  # (H, W)
            rays = self.generate_ray_bundle(
                self.cameras[idx], image.shape[:2]
            )  # (H, W, 6)

            # mask out invalid pixels
            rays = rays[mask]  # (N, 6)
            colors = image[mask]  # (N, 3)

            # randomly sample num_rays_per_image indices in range [0, N)
            sample_idx = torch.randperm(rays.shape[0])[: self.num_rays_per_image]
            rays = rays[sample_idx]  # (num_rays_per_image, 6)
            colors = colors[sample_idx]  # (num_rays_per_image, 3)
        elif self.dataset_type == "test":
            image = torch.ones((*self.visualize_image_shape, 3))  # (H, W, 3)
            colors = torch.ones((*self.visualize_image_shape, 3))  # (H, W, 3)
            rays = self.generate_ray_bundle(
                self.cameras[idx], self.visualize_image_shape
            )  # (H, W, 6)

        return {
            "inputs": {"rays": rays.unsqueeze(0)},
            "labels": {
                "colors": colors.unsqueeze(0),
                "raw_images": [image],
            },
        }

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """
        Collate function for the dataset.
        """
        return {
            "inputs": {
                "rays": torch.cat([sample["inputs"]["rays"] for sample in batch], dim=0)
            },
            "labels": {
                "colors": torch.cat(
                    [sample["labels"]["colors"] for sample in batch], dim=0
                ),
                "raw_images": [sample["labels"]["raw_images"][0] for sample in batch],
            },
        }

    @staticmethod
    def move_batch_to(batch: dict, *args, **kwargs) -> dict:
        """
        Moves all tensors in the batch to the specified device and dtype.

        Args:
            batch (dict): The batch dictionary returned by `collate_fn`.

        Returns:
            dict: A batch where all tensors are moved to the specified device and dtype.
        """

        def move_object(x):
            """Helper function to move a tensor/spherical object to the specified device and dtype."""
            if isinstance(x, torch.Tensor):
                return x.to(*args, **kwargs)
            return x

        def move_nested_structure(data):
            """Recursively move tensors in nested dictionaries or lists."""
            if isinstance(data, dict):
                return {
                    key: move_nested_structure(value) for key, value in data.items()
                }
            elif isinstance(data, list):
                return [move_nested_structure(item) for item in data]
            return move_object(data)

        return move_nested_structure(batch)

    @staticmethod
    def generate_ray_bundle(
        camera: dict,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        camera: {
        'f': float,             # focal length in pixels
        'k': (k1, k2),          # radial-distortion (ignored here)
        'R': [[..],[..],[..]],  # world→camera rotation
        't': [..]               # world→camera translation
        }
        image_size: (H, W)
        returns: rays (H, W, 6) tensor, last dim = (ox, oy, oz, dx, dy, dz)
        """
        H, W = image_size
        f = camera["f"]

        # 1) build R, t and compute camera center C in world coords
        R = torch.tensor(camera["R"], dtype=torch.float32)  # (3,3)
        t = torch.tensor(camera["t"], dtype=torch.float32)  # (3,)
        C = -R.T @ t  # (3,)  camera center = -Rᵀ t

        # 2) pixel grid
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )  # both (H, W)

        # principal point at image center, origin = (0,0) at center
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        # 3) directions in camera coords (camera looks down –Z)
        x_camera = (xs - cx) / f
        y_camera = (ys - cy) / f
        z_camera = -torch.ones_like(x_camera)
        directions_camera = torch.stack(
            [x_camera, y_camera, z_camera], dim=-1
        )  # (H, W, 3)

        # 4) rotate to world: since R is world→cam, cam→world = Rᵀ
        #    and for a row vector v: v_world = v_cam @ R
        directions_world = directions_camera @ R  # (H, W, 3)
        directions_world = F.normalize(directions_world, p=2, dim=-1)  # normalize

        # 5) origins (broadcast camera center)
        origins = C.view(1, 1, 3).expand(H, W, 3)  # (H, W, 3)

        # 6) stack into (H, W, 6)
        rays = torch.cat([origins, directions_world], dim=-1)  # (H, W, 6)

        return rays

    def visualize_batch(
        self,
        batch: dict,
        num_vis: int = -1,
    ) -> List[np.ndarray]:
        """
        Visualize a batch of data.

        Args:
            batch (dict): A batch of data.
            num_vis (int): The number of images to visualize. If -1, all images will be visualized.

        Returns:
            List[np.ndarray]: A list of images.
        """
        assert (
            self.dataset_type == "test"
        ), f"Visualization is only supported for test dataset"

        if num_vis == -1:
            num_vis = batch["labels"]["colors"].shape[0]
        elif num_vis > 0:
            num_vis = min(num_vis, batch["labels"]["colors"].shape[0])
        else:
            raise ValueError(f"Invalid number of images to visualize: {num_vis}")

        images = []

        for i in range(num_vis):
            colors = (
                batch["labels"]["colors"][i]
                .reshape(*self.visualize_image_shape, 3)
                .clamp(0, 1)
                * 255
            )
            images.append(to_numpy(colors).astype(np.uint8))

        return images


class FreeViewpointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        scene_path: str,
        batch_size: int,
        num_workers: int,
        num_rays_per_image: int = 1024,
        visualize_image_shape: Tuple[int, int] = (480, 640),
    ):
        super().__init__()
        self.scene_path = scene_path

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_rays_per_image = num_rays_per_image
        self.visualize_image_shape = visualize_image_shape

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = FreeViewpointDataset(
            scene_path=self.scene_path,
            dataset_type="train",
            num_rays_per_image=self.num_rays_per_image,
        )
        self.val_dataset = FreeViewpointDataset(
            scene_path=self.scene_path,
            dataset_type="test",
            visualize_image_shape=self.visualize_image_shape,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self.val_dataset.collate_fn,
        )
