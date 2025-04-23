#!/usr/bin/env python3
#
# Created on Thu Feb 06 2025 11:51:01
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import hashlib
from contextlib import contextmanager
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from multimethod import multimethod


@contextmanager
def fixed_seed(seed: int):
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def string_to_seed(s: str, max_value: int = 2**32 - 1) -> int:
    """Convert a string to a deterministic integer seed within [0, max_value)."""
    hash_digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(hash_digest, 16) % max_value


def million_trainable_params(model: nn.Module) -> float:
    total_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    return total_params / 1e6


def count_layers(model: nn.Module, layer_type: type) -> int:
    return sum(1 for layer in model.modules() if isinstance(layer, layer_type))


@multimethod
def detect_nan(x: torch.Tensor, variable_name: str) -> None:
    if torch.isnan(x).any():
        nan_indices = torch.nonzero(torch.isnan(x))
        print(f"{variable_name} contains nan at indices:\n{nan_indices.tolist()}")


@multimethod
def detect_nan(x: np.ndarray, variable_name: str) -> Optional[bool]:
    if np.isnan(x).any():
        nan_indices = np.argwhere(np.isnan(x))
        print(f"{variable_name} contains nan at indices:\n{nan_indices.tolist()}")
        return True


def module_weights_hash(module: torch.nn.Module) -> str:
    """
    Computes an MD5 hash for the weights of a module.

    Args:
        module (torch.nn.Module): The PyTorch module whose weights are to be hashed.

    Returns:
        str: An MD5 hash representing the weights of the module.
    """
    md5 = hashlib.md5()
    for _, weight in module.named_parameters():
        # Move to CPU and cast to a fixed dtype for consistency.
        array = (
            to_numpy(copy_or_clone(weight)).astype(np.float32).astype("<f4", copy=False)
        )
        md5.update(array.tobytes())
    return md5.hexdigest()


@multimethod
def array_hash(array: np.ndarray) -> str:
    """Hash the numpy array based on its content, cross-comparable with torch tensors

    Args:
        array (np.ndarray)

    Returns:
        str: md5
    """
    assert isinstance(array, np.ndarray), "Expect input to be a numpy array"

    # Standardize: copy the array, cast to float32, and force little-endian byte order.
    standardized = np.array(array, copy=True)
    standardized = standardized.astype(np.float32)  # Convert to float32
    standardized = standardized.astype("<f4", copy=False)  # Force little-endian

    return hashlib.md5(standardized.tobytes()).hexdigest()


@multimethod
def array_hash(array: torch.Tensor) -> str:
    """Hash the tensor based on its content, cross-comparable with numpy arrays

    Args:
        array (torch.Tensor)

    Returns:
        str: md5
    """
    assert isinstance(array, torch.Tensor), "Expect input to be a torch.Tensor"

    return array_hash(to_numpy(copy_or_clone(array)))


@multimethod
def array_size_MiB(array: torch.Tensor) -> float:
    """
    Compute the memory size of a tensor in MiB, handling both dense and sparse tensors.

    Args:
        array (torch.Tensor): Input tensor (dense or sparse).

    Returns:
        float: Memory size in MiB.
    """
    assert isinstance(array, torch.Tensor), "Expect input to be a torch.Tensor"

    if array.is_sparse:
        # Sparse tensor: compute size using indices and values
        array = array.coalesce()
        indices = array.indices()
        values = array.values()
        total_bytes = (indices.numel() * indices.element_size()) + (
            values.numel() * values.element_size()
        )
    else:
        # Dense tensor: compute size based on total elements and element size
        total_bytes = array.numel() * array.element_size()

    return total_bytes / (2**20)  # Convert to MiB


@multimethod
def array_size_MiB(array: np.ndarray) -> float:
    """
    Compute the memory size of an array in MiB.

    Supports both dense numpy arrays and scipy sparse matrices.

    Args:
        array (np.ndarray): Input array.

    Returns:
        float: Memory size in MiB.
    """
    assert isinstance(array, np.ndarray), "Expect input to be a numpy array"

    total_bytes = array.nbytes

    return total_bytes / (2**20)


@multimethod
def array_size_MiB(array: sp.spmatrix) -> float:
    """
    Compute the memory size of a sparse matrix in MiB.

    Args:
        array (sp.spmatrix): Input sparse matrix.

    Returns:
        float: Memory size in MiB.
    """
    assert isinstance(array, sp.spmatrix), "Expect input to be a scipy sparse matrix"

    total_bytes = array.data.nbytes + array.indices.nbytes + array.indptr.nbytes

    return total_bytes / (2**20)


@multimethod
def to_torch(
    array: np.ndarray, device: torch.device = None, dtype: torch.dtype = None
) -> torch.Tensor:
    """Convert numpy array to torch tensor

    Args:
        array (np.ndarray)

    Returns:
        torch.Tensor
    """

    return torch.from_numpy(array).to(device=device, dtype=dtype)


@multimethod
def to_torch(
    array: torch.Tensor, device: torch.device = None, dtype: torch.dtype = None
) -> torch.Tensor:
    """Convert torch tensor to torch tensor

    Args:
        array (torch.Tensor)

    Returns:
        torch.Tensor
    """

    return array.to(device=device, dtype=dtype)


@multimethod
def to_numpy(array: np.ndarray) -> np.ndarray:
    """Convert numpy array to numpy array

    Args:
        array (np.ndarray)

    Returns:
        np.ndarray
    """

    return array


@multimethod
def to_numpy(array: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array

    Args:
        array (torch.Tensor)

    Returns:
        np.ndarray
    """

    return array.numpy(force=True)


@multimethod
def copy_or_clone(array: np.ndarray) -> np.ndarray:
    """Copy numpy array

    Args:
        array (np.ndarray)

    Returns:
        np.ndarray
    """

    return array.copy()


@multimethod
def copy_or_clone(array: torch.Tensor) -> torch.Tensor:
    """Clone torch tensor

    Args:
        array (torch.Tensor)

    Returns:
        torch.Tensor
    """

    return array.clone()


@multimethod
def stack(arrays: Union[tuple[np.ndarray], list[np.ndarray]], dim: int) -> np.ndarray:
    """Stack numpy array

    Args:
        arrays (Union[List[np.ndarray], Tuple[np.ndarray]])
        dim (int): dimension, translated into axis

    Returns:
        np.ndarray
    """
    return np.stack(arrays, axis=dim)


@multimethod
def stack(
    arrays: Union[tuple[torch.Tensor], list[torch.Tensor]], dim: int
) -> torch.Tensor:
    """Stack torch tensor

    Args:
        arrays (Union[List[torch.Tensor], Tuple[torch.Tensor]])
        dim (int): dimension

    Returns:
        torch.Tensor
    """
    return torch.stack(arrays, dim=dim)


@multimethod
def concatenate(
    arrays: Union[tuple[np.ndarray], list[np.ndarray]], dim: int
) -> np.ndarray:
    """Concatenate numpy array

    Args:
        arrays (Union[List[np.ndarray], Tuple[np.ndarray]])
        dim (int): dimension, translated into axis

    Returns:
        np.ndarray
    """
    return np.concatenate(arrays, axis=dim)


@multimethod
def concatenate(
    arrays: Union[tuple[torch.Tensor], list[torch.Tensor]], dim: int
) -> torch.Tensor:
    """Concatenate torch tensor

    Args:
        arrays (Union[List[torch.Tensor], Tuple[torch.Tensor]])
        dim (int): dimension

    Returns:
        torch.Tensor
    """
    return torch.cat(arrays, dim=dim)


@multimethod
def split(
    array: np.ndarray, indices_or_sections: Union[int, List[int]], dim: int
) -> List[np.ndarray]:
    """
    Split a numpy array along a specified dimension.

    Args:
        array (np.ndarray): The input array.
        indices_or_sections (Union[int, List[int]]): Either an integer indicating the number
            of equal splits or a list of indices where the split should occur.
        dim (int): The dimension along which to split (mapped to np.split's axis).

    Returns:
        List[np.ndarray]: A list of numpy arrays after splitting.
    """
    return np.split(array, indices_or_sections, axis=dim)


@multimethod
def split(
    array: torch.Tensor, indices_or_sections: Union[int, List[int]], dim: int
) -> List[torch.Tensor]:
    """
    Split a torch tensor along a specified dimension.

    Args:
        array (torch.Tensor): The input tensor.
        indices_or_sections (Union[int, List[int]]): Either an integer for uniform splits or
            a list of sizes for each split.
        dim (int): The dimension along which to split.

    Returns:
        List[torch.Tensor]: A list of torch tensors after splitting.
    """
    return list(torch.split(array, indices_or_sections, dim=dim))
