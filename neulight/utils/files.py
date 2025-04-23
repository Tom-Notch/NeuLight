#!/usr/bin/env python3
#
# Created on Thu Jun 29 2023 18:56:15
# Author: Mukai (Tom Notch) Yu, Yao He
# Email: mukaiy@andrew.cmu.edu, yaohe@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute, the AirLab
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu, Yao He
#
import os
import os.path as osp
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import magic
import numpy as np
import yaml


mime = magic.Magic(mime=True, uncompress=True)

dtype_map: dict[str, Any] = {
    "u": np.uint8,
    "i": np.int32,
    "f": np.float32,
    "d": np.float64,
}  # Map OpenCV data types to NumPy data types


def opencv_matrix_constructor(loader, node):
    """Custom constructor for !!opencv-matrix tag."""

    # Parse the node as a dictionary
    matrix_data = loader.construct_mapping(node, deep=True)

    # Extract rows, cols, dt, and data
    rows = matrix_data["rows"]
    cols = matrix_data["cols"]
    dt = matrix_data["dt"]
    data = matrix_data["data"]

    # Determine the NumPy data type
    dtype = dtype_map.get(dt, None)

    assert (
        dtype is not None
    ), f"Invalid data type annotation {dt}, only supports {dtype_map.keys()}"

    # Convert data to a NumPy array and reshape
    matrix = np.array(data, dtype=dtype).reshape(rows, cols)

    return matrix


yaml.add_constructor("tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)


def print_dict(d: dict, indent: int = 0) -> None:
    """print a dictionary with indentation, uses recursion to print nested dictionaries

    Args:
        d (dict): dictionary to be printed
        indent (int, optional): indentation level. Defaults to 0.

    Returns:
        None
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + str(key) + ": ")
            print_dict(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print(
                "  " * indent + str(key) + ":\n" + str(value)
            )  # new line before printing matrix for better readability
        else:
            print("  " * indent + str(key) + ": " + str(value))


def parse_path(probe_path: str, base_path: Optional[str] = None) -> Union[str, bool]:
    """parse a potential path, expand ~ to user home, and check if the path exists

    Args:
        probe_path (str): the path to be parsed
        base_path (str, optional): the base path of the current (yaml) file. Defaults to None.

    Returns:
        the parsed path if it exists, False otherwise
    """
    expand_path = osp.expanduser(probe_path)  # expand ~ to usr home
    if base_path is None:
        base_path = os.getcwd()
    if osp.isabs(expand_path) and osp.exists(expand_path):
        return osp.realpath(expand_path)
    elif osp.exists(osp.join(base_path, probe_path)):
        return osp.realpath(osp.join(base_path, probe_path))
    else:
        return False


def parse_content(node, yaml_base_path: str, file_cache: dict[str, Any] = {}):
    """recursively look into the leaf node of a yaml file, if the leaf node is a string, try to parse it as a path and read the file, could be an image or nested yaml config

    Args:
        node: file node to be parsed
        yaml_base_path (str): the base path of the current yaml file

    Returns:
        the read content
    """

    if isinstance(node, dict):
        for key, value in node.items():
            node[key] = parse_content(value, yaml_base_path, file_cache=file_cache)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            node[index] = parse_content(value, yaml_base_path, file_cache=file_cache)
    elif isinstance(node, str):
        path = parse_path(node, yaml_base_path)
        if path:
            node = read_file(path, file_cache=file_cache)

    return node


def read_exr(image_file_path: str) -> np.ndarray:
    """
    Read an OpenEXR file and return an (H, W, 3) RGB (or XYZ) image as float32.

    Args:
        image_file_path: path to a .exr file with R, G, B channels stored as 32-bit floats.

    Returns:
        image: numpy array of shape (H, W, 3), dtype float32.
    """
    import OpenEXR
    import Imath

    exr_file = OpenEXR.InputFile(image_file_path)
    header = exr_file.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    # Read as 32‑bit float
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    r_str = exr_file.channel("R", pixel_type)
    g_str = exr_file.channel("G", pixel_type)
    b_str = exr_file.channel("B", pixel_type)

    # Convert raw bytes to NumPy arrays
    r = np.frombuffer(r_str, dtype=np.float32).reshape((height, width))
    g = np.frombuffer(g_str, dtype=np.float32).reshape((height, width))
    b = np.frombuffer(b_str, dtype=np.float32).reshape((height, width))

    # Stack into H × W × 3
    image = np.stack((r, g, b), axis=-1)

    # clean up
    exr_file.close()
    return image


def read_bundler(path: str):
    """Parse a Bundler bundle.out file into camera and point lists."""
    cameras: List[Dict[str, Any]] = []
    with open(path, "r") as file_handle:
        # skip optional header
        first = file_handle.readline().strip()
        if not first or first.startswith("#"):
            first = file_handle.readline().strip()
        num_cameras, _ = map(int, first.split())

        # read cameras
        for _ in range(num_cameras):
            f_len, k1, k2 = map(float, file_handle.readline().split())
            R = [list(map(float, file_handle.readline().split())) for _ in range(3)]
            t = list(map(float, file_handle.readline().split()))
            cameras.append({"f": f_len, "k": (k1, k2), "R": R, "t": t})

    return cameras


def read_file(path: str, file_cache: dict[str, Any] = {}):
    """test the path, read a file.
       supports multiple file type and interleaving config file types.

    Args:
        path (str): path to the file, can be absolute or relative

    Returns:
        content of the file
    """
    path = parse_path(path)

    if not path:
        raise FileNotFoundError(f"File {path} does not exist")

    if path in file_cache.keys():
        return file_cache[path]

    if osp.isdir(path):
        parsed_content = str(path)
    else:
        # determine file type with python-magic, useful for massive image types
        mime_type = mime.from_file(path)

        file_base_path = osp.dirname(path)

        if mime_type in {
            "application/yaml",
            "application/x-yaml",
            "text/yaml",
        } or path.endswith((".yaml", ".yml")):
            import yaml

            with open(path, "r") as f:
                parsed_yaml = yaml.load(f.read(), Loader=yaml.FullLoader)
            parsed_content = parse_content(
                parsed_yaml, file_base_path, file_cache=file_cache
            )
        elif path.lower().endswith(".exr"):
            parsed_content = read_exr(path)
        elif path.lower().endswith(".out"):
            parsed_content = read_bundler(path)
        elif mime_type.startswith("image/") and not path.lower().endswith(".exr"):
            from PIL import Image

            with Image.open(path) as img:
                parsed_content = np.array(img)
        elif mime_type in {"text/csv", "application/csv"} or path.endswith(".csv"):
            import csv

            with open(path, "r") as f:
                reader = csv.reader(f)
                parsed_content = list(reader)
        elif mime_type in {"application/json", "text/json"} or path.endswith(".json"):
            import json

            with open(path, "r") as f:
                parsed_content = parse_content(
                    json.load(f), file_base_path, file_cache=file_cache
                )
        elif mime_type in {"application/toml", "text/toml"} or path.endswith(".toml"):
            import toml

            parsed_content = parse_content(
                toml.load(path), file_base_path, file_cache=file_cache
            )
        elif path.endswith((".npy", ".npz")):
            try:
                parsed_content = np.load(path, allow_pickle=True)
            except Exception as e:
                raise RuntimeError(f"Error loading NumPy array from {path}: {e}")
        elif path.endswith((".ckpt")):
            import torch

            parsed_content = torch.load(path)
        elif mime_type in {
            "application/x-torchscript",
            "application/x-tensorrt",
        } or path.endswith((".ts", ".trt")):
            import torch
            import torch_tensorrt

            parsed_content = torch.jit.load(path)
        else:
            warnings.warn(f"File {path} is not supported, reading as string")
            parsed_content = path

    file_cache[path] = parsed_content
    return parsed_content
