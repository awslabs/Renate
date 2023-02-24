# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import os
from typing import Tuple, Union

import torch
from renate.types import NestedTensors

Index = Union[str, slice]

TORCH_DTYPES_BY_STRING = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}


def mmap_tensor(filename: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Creates or accesses a memory-mapped tensor."""
    t = torch.from_file(filename, shared=True, size=math.prod(shape), dtype=dtype, device="cpu")
    return t.view(shape)


def _create_or_access_storage(path: str, length: int, shapes, dtypes) -> NestedTensors:
    """Creates or accesses a nested structure of memory-mapped tensors."""
    if isinstance(dtypes, str):
        assert isinstance(shapes, tuple)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        filename = f"{path}.pt"
        shape = (length,) + shapes
        dtype = TORCH_DTYPES_BY_STRING[dtypes]
        return mmap_tensor(filename, shape, dtype)
    elif isinstance(dtypes, tuple):
        assert isinstance(shapes, tuple)
        return tuple(
            _create_or_access_storage(os.path.join(path, f"{i}"), length, shapes[i], dtypes[i])
            for i in range(len(dtypes))
        )
    elif isinstance(dtypes, dict):
        assert isinstance(shapes, dict)
        return {
            key: _create_or_access_storage(
                os.path.join(path, key), length, shapes[key], dtypes[key]
            )
            for key in dtypes
        }


def _get_data_point(storage: NestedTensors, idx: Index) -> NestedTensors:
    """Retrieves a data point from a nested tensor storage.

    Args:
        storage: A storage for nested tensors, e.g., created using `_create_or_access_storage`.
        idx: An integer index or slice indicating the point(s) to extract.

    Returns:
        The extracted data point(s).
    """
    if isinstance(storage, torch.Tensor):
        return storage[idx]
    elif isinstance(storage, tuple):
        return tuple(_get_data_point(t, idx) for t in storage)
    elif isinstance(storage, dict):
        return {key: _get_data_point(t, idx) for key, t in storage.items()}
    else:
        raise TypeError(f"Expected nested tuple/dict of tensors, found {type(storage)}.")


def _insert_data_point(storage: NestedTensors, idx: Index, data_point: NestedTensors) -> None:
    """Inserts a data point into a nested tensor storage.

    Args:
        storage: A storage for nested tensors, e.g., created using `_create_or_access_storage`.
        idx: An integer index or slice indicating where to insert the data point(s).
        data_point: The data point(s) to insert, i.e., a nested tensor structure compatible with
            `storage`. If `idx` is a slice, the tensors in `data_point` need to have a batch
            dimension of corresponding length. If `idx` is an integer, they must not have a batch
            dimension.
    """
    if isinstance(storage, torch.Tensor):
        assert isinstance(data_point, torch.Tensor)
        assert data_point.dtype is storage.dtype
        storage[idx] = data_point
    elif isinstance(storage, tuple):
        assert isinstance(data_point, tuple)
        assert len(data_point) == len(storage)
        for i in range(len(storage)):
            _insert_data_point(storage[i], idx, data_point[i])
    elif isinstance(storage, dict):
        assert isinstance(data_point, dict)
        assert set(data_point.keys()) == set(storage.keys())
        for key in storage:
            _insert_data_point(storage[key], idx, data_point[key])


def get_nested_tensor_schema(nested_tensors: NestedTensors):
    """Extracts shapes and data types of nested tensors."""
    if isinstance(nested_tensors, torch.Tensor):
        shapes = tuple(nested_tensors.size())
        dtypes = str(nested_tensors.dtype)
        return shapes, dtypes
    elif isinstance(nested_tensors, tuple) or isinstance(nested_tensors, list):
        shapes = tuple(get_nested_tensor_schema(t)[0] for t in nested_tensors)
        dtypes = tuple(get_nested_tensor_schema(t)[1] for t in nested_tensors)
        return shapes, dtypes
    elif isinstance(nested_tensors, dict):
        shapes = {key: get_nested_tensor_schema(t)[0] for key, t in nested_tensors.items()}
        dtypes = {key: get_nested_tensor_schema(t)[1] for key, t in nested_tensors.items()}
        return shapes, dtypes
    else:
        raise TypeError(f"Expected nested tuple/dict of tensors; found {type(nested_tensors)}.")


class Storage(torch.utils.data.Dataset):
    """A class implementing permanent storage of nested tensor datasets.

    This implements storage for `length` data points consisting of nested tensors of fixed types
    and shapes. `Storage` implements `__len__` and `__getitem__` and therefore can be used as a
    torch `Datasets`. To populate the storage, it also implements `__setitem__`. It does _not_ keep
    track which slots have or have not been populated.

    `Storage` is given a path to a directory, where it creates (or accesses, if they already exist)
    memory-mapped tensor files.

    Args:
        directory: Path to a directory.
        length: Number of items to be stored.
        shapes: Shapes of the nested tensor structure. A nested tuple/dict of shape tuples.
        dtypes: Data types of the nested tensor structure. A nested tuple/dict of dtype strings.
    """

    def __init__(self, directory: str, length: int, shapes, dtypes) -> None:
        self._directory = directory
        self._length = length
        self._shapes = shapes
        self._dtypes = dtypes
        os.makedirs(directory, exist_ok=True)
        self._storage = _create_or_access_storage(directory, length, shapes, dtypes)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> NestedTensors:
        """Read the item stored at index `idx`."""
        return _get_data_point(self._storage, idx)

    def __setitem__(self, idx: int, item: NestedTensors) -> None:
        """Set the item stored at index `idx`."""
        _insert_data_point(self._storage, idx, item)
