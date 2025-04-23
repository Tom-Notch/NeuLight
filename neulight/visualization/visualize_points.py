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


def visualize_points(
    points: Union[np.ndarray, torch.Tensor],
    colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
    point_size: float = 10.0,
    interactive: bool = True,
    image_size: tuple[int, int] = (800, 600),
    look_at: Optional[np.ndarray] = None,
    front: Optional[np.ndarray] = None,
    vertical_fov: float = 90.0,
    zoom: float = 1.0,
    show_axis: bool = True,
) -> Optional[np.ndarray]:
    """
    Visualize a PointCloud data using Open3D.

    The PointCloud has:
      - value (N, 3): point colors in [0..255] for each of N points.
      - vec (N, 3): Point coordinates in Cartesian coordinates.

    This function displays a point cloud whose points are given by vec and whose colors (normalized to [0,1]) are updated for each frame.
    The function is written in a pure functional style: it does not modify the input instance.

    Args:
        point_cloud (PointCloud): Anything that can be converted to a PointCloud
        point_size (float, optional): Size of points in the visualization. Defaults to 2.0.
        interactive (bool, optional): Whether to use interactive rendering. Defaults to True.
        image_size (tuple[int, int], optional): Size of the image to render. Defaults to (800, 600).
        look_at (np.ndarray, optional): Center of the camera. Defaults to (0, 0, 0).
        front (np.ndarray, optional): Front vector of the camera. Defaults to (1, 0, 0).
        vertical_fov (float, optional): Vertical field of view of the camera. Defaults to 90.0.
        zoom (float, optional): Zoom level of the camera. Defaults to 1.0.

    Returns:
        Optional[np.ndarray]: The rendered image if static is True, otherwise None.
    """
    _points = to_numpy(points).copy()

    if colors is None:
        _colors = np.ones_like(_points) * 0.5
    else:
        _colors = to_numpy(colors).copy()

    num_raw_input_channels = colors.shape[-1]
    assert (
        num_raw_input_channels == 1 or num_raw_input_channels == 3
    ), f"Expect # channel to be 1 or 3, but got {num_raw_input_channels}"
    if num_raw_input_channels == 1:
        colors = np.repeat(colors, 3, axis=-1)

    assert (
        _points.shape == _colors.shape
    ), f"points and colors must have matching shapes, got points.shape == {_points.shape} and colors.shape == {_colors.shape}"

    _points = _points.reshape(-1, 3)
    _colors = _colors.reshape(-1, 3).astype(np.float64)

    if look_at is None:
        look_at = _points.mean(axis=0)

    if front is None:
        front = np.array([-1, 0, -1])

    # Precompute normalized colors once (convert from [0, 255] to [0, 1]).
    if _colors.max() > 1.0:
        _colors = _colors / 255.0

    # Create the point cloud and set its points once.
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(_colors)
    pcd.points = o3d.utility.Vector3dVector(_points)

    camera_front = -front

    if interactive:
        # --- interactive render ---
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.add_geometry(pcd)

        if show_axis:
            vis.add_geometry(
                o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            )

        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1.0, 1.0, 1.0])  # White background
        render_option.point_size = point_size
        render_option.light_on = False

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
        material.shader = "defaultUnlit"
        material.point_size = point_size
        scene.add_geometry("point cloud", pcd, material)

        if show_axis:
            scene.show_axes(True)

        scene.set_background([1, 1, 1, 1])

        img = renderer.render_to_image()
        img = np.asarray(img)

        return img
