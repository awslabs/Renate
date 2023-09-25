# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Union
from warnings import warn

import torch

from renate.types import NestedTensors


def mmap_tensor(
    filename: str, size: Union[int, Tuple[int, ...]], dtype: torch.dtype
) -> torch.Tensor:
    """Creates or accesses a memory-mapped tensor."""
    t = torch.from_file(
        filename,
        shared=True,
        size=math.prod(size) if isinstance(size, tuple) else size,
        dtype=dtype,
        device="cpu",
    )
    return t.view(size)


class Storage(torch.utils.data.Dataset):
    """An abstract class for permanent storage of datasets."""

    def __init__(self, directory: str) -> None:
        self._directory = directory

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError()

    def dump_dataset(self, ds: torch.utils.data.Dataset) -> None:
        raise NotImplementedError()

    def load_dataset(self, directory: Union[str, Path]):
        raise NotImplementedError()


class MemoryMappedTensorStorage(Storage):
    """A class implementing permanent storage of nested tensor datasets.

    This implements storage for `length` data points consisting of nested tensors of fixed types
    and shapes. `Storage` implements `__len__` and `__getitem__` and therefore can be used as a
    torch `Dataset`. To populate the storage, it also implements `dump_dataset`. It does _not_ keep
    track which slots have or have not been populated.

    `Storage` is given a path to a directory, where it creates (or accesses, if they already exist)
    memory-mapped tensor files.

    Args:
        directory: Path to a directory.
        data_point: Prototypical datapoint from which to infer shapes/dtypes.
        length: Number of items to be stored.
    """

    def __init__(self, directory: str) -> None:
        warn(
            f"""{self.__class__.__name__} will be deprecated very soon. Use FileTensorStorage
            instead. {self.__class__.__name__} is currently not fully functional, as some of the
            necessary parts of the interface have been modified and simplified. """,
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(directory)
        self._storage: Optional[NestedTensors] = None

    @staticmethod
    def _create_mmap_tensors(path: str, data_point: NestedTensors, length: int) -> NestedTensors:
        if isinstance(data_point, torch.Tensor):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            filename = f"{path}.pt"
            return mmap_tensor(filename, size=(length, *data_point.size()), dtype=data_point.dtype)
        elif isinstance(data_point, tuple):
            return tuple(
                MemoryMappedTensorStorage._create_mmap_tensors(
                    os.path.join(path, f"{i}.pt"), data_point[i], length
                )
                for i in range(len(data_point))
            )
        elif isinstance(data_point, dict):
            return {
                key: MemoryMappedTensorStorage._create_mmap_tensors(
                    os.path.join(path, f"{key}.pt"), data_point[key], length
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

    def dump_dataset(self, ds):
        self._length = len(ds)
        self._storage = self._create_mmap_tensors(self._directory, ds[0], self._length)
        for idx in range(len(self)):
            self._set(self._storage, idx, ds[idx])


class FileTensorStorage(Storage):
    """A class implementing permanent storage of nested tensor datasets to disk as pickle files.

    This implements storage for `length` data points consisting of nested tensors of fixed types
    and shapes. `Storage` implements `__len__` and `__getitem__` and therefore can be used as a
    torch `Dataset`. To populate the storage, it also implements `dump_dataset`. It does _not_ keep
    track which slots have or have not been populated.

    `Storage` is given a path to a directory, where it creates (or accesses, if they already exist)
    pickle files one for each point in the dataset.

    Args:
        directory: Path to a directory.
    """

    def __init__(self, directory: str) -> None:
        super().__init__(directory)

    def dump_dataset(self, ds: torch.utils.data.Dataset) -> None:
        for i in range(len(ds)):
            torch.save(ds[i], self._compose_file_path_from_index(i))

    def __getitem__(self, idx: int) -> Any:
        if not hasattr(self, "_length"):
            self.load_dataset(None)
        return torch.load(self._compose_file_path_from_index(idx))

    def load_dataset(self, directory: Union[str, Path]):
        self._length = len([x for x in os.listdir(self._directory) if x.endswith(".pt")])

    def _compose_file_path_from_index(self, idx: int) -> str:
        return os.path.join(self._directory, f"{idx}.pt")
