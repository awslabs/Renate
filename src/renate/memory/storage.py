# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import os
from typing import Any, Tuple

import torch

from renate.types import NestedTensors


def mmap_tensor(filename: str, size: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Creates or accesses a memory-mapped tensor."""
    t = torch.from_file(filename, shared=True, size=math.prod(size), dtype=dtype, device="cpu")
    return t.view(size)


class Storage(torch.utils.data.Dataset):
    """An abstract class for permanent storage of datasets."""

    def __init__(self, directory: str, data_point: Any, length: int) -> None:
        self._directory = directory
        self._data_point = data_point
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError()

    def __setitem__(self, idx: int, data_point: Any) -> None:
        raise NotImplementedError()


class MemoryMappedTensorStorage(Storage):
    """A class implementing permanent storage of nested tensor datasets.

    This implements storage for `length` data points consisting of nested tensors of fixed types
    and shapes. `Storage` implements `__len__` and `__getitem__` and therefore can be used as a
    torch `Datasets`. To populate the storage, it also implements `__setitem__`. It does _not_ keep
    track which slots have or have not been populated.

    `Storage` is given a path to a directory, where it creates (or accesses, if they already exist)
    memory-mapped tensor files.

    Args:
        directory: Path to a directory.
        data_point: Prototypical datapoint from which to infer shapes/dtypes.
        length: Number of items to be stored.
    """

    def __init__(self, directory: str, data_point: NestedTensors, length: int) -> None:
        super().__init__(directory, data_point, length)
        self._storage = self._create_mmap_tensors(directory, data_point, length)

    @staticmethod
    def _create_mmap_tensors(path: str, data_point: NestedTensors, length: int) -> NestedTensors:
        if isinstance(data_point, torch.Tensor):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            filename = f"{path}.pt"
            return mmap_tensor(filename, size=(length, *data_point.size()), dtype=data_point.dtype)
        elif isinstance(data_point, tuple):
            return tuple(
                MemoryMappedTensorStorage._create_mmap_tensors(
                    os.path.join(path, f"{i}"), data_point[i], length
                )
                for i in range(len(data_point))
            )
        elif isinstance(data_point, dict):
            return {
                key: MemoryMappedTensorStorage._create_mmap_tensors(
                    os.path.join(path, key), data_point[key], length
                )
                for key in data_point
            }
        else:
            raise TypeError(f"Expected nested tuple/dict of tensors, found {type(data_point)}.")

    @staticmethod
    def _get(storage: NestedTensors, idx: int) -> NestedTensors:
        if isinstance(storage, torch.Tensor):
            return storage[idx]
        elif isinstance(storage, tuple):
            return tuple(MemoryMappedTensorStorage._get(t, idx) for t in storage)
        elif isinstance(storage, dict):
            return {key: MemoryMappedTensorStorage._get(t, idx) for key, t in storage.items()}
        else:
            raise TypeError(f"Expected nested tuple/dict of tensors, found {type(storage)}.")

    def __getitem__(self, idx: int) -> NestedTensors:
        """Read the item stored at index `idx`."""
        return self._get(self._storage, idx)

    @staticmethod
    def _set(storage: NestedTensors, idx: int, data_point: NestedTensors) -> None:
        if isinstance(storage, torch.Tensor):
            assert isinstance(data_point, torch.Tensor)
            assert data_point.dtype is storage.dtype
            storage[idx] = data_point
        elif isinstance(storage, tuple):
            assert isinstance(data_point, tuple)
            assert len(data_point) == len(storage)
            for i in range(len(storage)):
                MemoryMappedTensorStorage._set(storage[i], idx, data_point[i])
        elif isinstance(storage, dict):
            assert isinstance(data_point, dict)
            assert set(data_point.keys()) == set(storage.keys())
            for key in storage:
                MemoryMappedTensorStorage._set(storage[key], idx, data_point[key])
        else:
            raise TypeError(f"Expected nested tuple/dict of tensors, found {type(storage)}.")

    def __setitem__(self, idx: int, data_point: NestedTensors) -> None:
        """Set the item stored at index `idx`."""
        self._set(self._storage, idx, data_point)
