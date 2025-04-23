#!/usr/bin/env python3
#
# Created on Tue Nov 12 2024 17:37:01
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import contextlib
import os
import sys
from typing import Optional
from typing import Union

import numpy as np
import open3d as o3d
import torch
from open3d.visualization import rendering

from neulight.utils.torch_numpy import to_numpy


os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ["OPEN3D_CPU_RENDERING"] = "true"

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


@contextlib.contextmanager
def _suppress_driver_logs():
    # monkey‑patch sys.stdout & sys.stderr
    with open(os.devnull, "w") as fnull:
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = fnull, fnull
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def visualize_rays(
    rays: Union[np.ndarray, torch.Tensor],
    colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ray_length: float = 2.0,
    interactive: bool = True,
    image_size: tuple[int, int] = (800, 600),
    look_at: Optional[np.ndarray] = None,
    front: Optional[np.ndarray] = None,
    vertical_fov: float = 90.0,
    zoom: float = 1.0,
    show_axis: bool = True,
) -> Optional[np.ndarray]:
    """
    Visualize ray bundle.

    Args:
        rays (np.ndarray): Ray bundle in shape (..., 6)
        colors (np.ndarray, optional): Colors of the rays in shape (..., 3). Defaults to None.
        ray_length (float, optional): Length of the rays. Defaults to 2.0.
        interactive (bool, optional): Whether to use interactive rendering. Defaults to True.
        image_size (tuple[int, int], optional): Size of the image to render. Defaults to (800, 600).
        look_at (np.ndarray, optional): Center of the camera. Defaults to (0, 0, 0).
        front (np.ndarray, optional): Front vector of the camera. Defaults to (1, 0, 0).
        vertical_fov (float, optional): Vertical field of view of the camera. Defaults to 90.0.
        zoom (float, optional): Zoom level of the camera. Defaults to 1.0.

    Returns:
        Optional[np.ndarray]: The rendered image if static is True, otherwise None.
    """

    _rays = to_numpy(rays).copy().reshape(-1, 6)
    N = _rays.shape[0]

    origins = _rays[:, :3]  # (N, 3)
    directions = _rays[:, 3:]  # (N, 3)
    ends = (
        origins
        + directions / np.linalg.norm(directions, axis=1, keepdims=True) * ray_length
    )  # (N, 3)

    points = np.vstack([origins, ends])  # (2N, 3)
    lines = [[i, i + N] for i in range(N)]

    # Normalize colors to [0,1]
    if colors is None:
        colors = np.ones((N, 3)) * 0.5  # Gray
    else:
        colors = to_numpy(colors).copy().reshape(-1, 3)  # (N, 3)

    _colors = colors.astype(np.float64)
    if _colors.max() > 1.0:
        _colors = _colors / 255.0

    assert _rays.shape[0] == _colors.shape[0], "# rays and # colors mismatch"

    line_colors = _colors.tolist()

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    if look_at is None:
        look_at = origins.mean(axis=0)

    if front is None:
        front = np.array([-1, 0, -1])

    camera_front = -front

    if interactive:
        # --- interactive render ---
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.add_geometry(line_set)

        if show_axis:
            vis.add_geometry(
                o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            )

        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1.0, 1.0, 1.0])  # White background

        # Basic camera setup
        view_control = vis.get_view_control()
        view_control.set_front(camera_front)
        view_control.set_lookat(look_at)
        view_control.set_up([0, 0, 1])
        view_control.set_zoom(1 / zoom)

        try:
            while vis.poll_events():
                vis.update_renderer()
        except KeyboardInterrupt:
            pass

        vis.destroy_window()

        return None
    else:
        # --- off-screen render ---
        with _suppress_driver_logs():
            renderer = rendering.OffscreenRenderer(*image_size)

        renderer.setup_camera(
            float(vertical_fov) * float(1 / zoom),
            look_at,
            look_at + camera_front,
            [0, 0, 1],
        )

        scene = renderer.scene
        scene.clear_geometry()

        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = ray_length
        scene.add_geometry("ray bundle", line_set, material)

        if show_axis:
            scene.show_axes(True)

        scene.set_background([1, 1, 1, 1])

        img = renderer.render_to_image()
        img = np.asarray(img)

        return img
