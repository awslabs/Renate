# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import os
from typing import Any, Tuple, Union, Optional, List
from pathlib import Path

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

    def __init__(self, directory: str, data_point: Any, length: int) -> None:
        self._directory = directory
        self._data_point = data_point
        self._length = length

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
    torch `Datasets`. To populate the storage, it also implements `dump_dataset`. It does _not_ keep
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
        for idx in range(len(self)):
            self._set(self._storage, idx, ds[idx][0])  # Drop metadata


class FlattenedMemoryMappedTensorStorage(Storage):
    def __init__(self, directory: str, data_point: Any, length: int) -> None:
        super().__init__(directory, data_point, length)
        self._storage: Optional[NestedTensors] = None

    def _create_mmap_tensors(self, ds: torch.utils.data.Dataset):
        ref_data_point = ds[0][0]
        if isinstance(ref_data_point, torch.Tensor):
            # We are storing a tensor
            num_elements = sum(ds[i][0].numel() for i in range(len(ds)))
            os.makedirs(os.path.dirname(self._directory), exist_ok=True)
            filename = f"{self._directory}.pt"
            self._storage = mmap_tensor(filename, size=(num_elements,), dtype=ref_data_point.dtype)

        elif isinstance(ref_data_point, tuple):
            # each ds[i][0] is a tuple. So you we need to make one memmap per tuple.
            num_elements = [0] * len(ref_data_point)
            for i in range(len(ds)):
                for e_index, e in enumerate(ds[i][0]):
                    num_elements[e_index] += e.numel()

            storage = [None] * len(ref_data_point)
            for i, n_e in enumerate(num_elements):
                save_name = os.path.join(self._directory, f"{i}.pt")
                os.makedirs(os.path.dirname(save_name), exist_ok=True)
                storage[i] = mmap_tensor(save_name, size=(n_e,), dtype=ref_data_point[i].dtype)

            self._storage = tuple(storage)

        elif isinstance(ref_data_point, dict):
            num_elements = [0] * len(ref_data_point)
            for i in range(len(ds)):
                curr_element = ds[i][0]
                for v_index, v in enumerate(curr_element.values()):
                    num_elements[v_index] += v.numel()
            storage = {k: None for k in ref_data_point.keys()}
            for i, k in enumerate(ref_data_point.keys()):
                save_name = os.path.join(self._directory, f"{k}.pt")
                os.makedirs(os.path.dirname(save_name), exist_ok=True)
                storage[k] = mmap_tensor(
                    save_name, size=(num_elements[i],), dtype=ref_data_point[k].dtype
                )

            self._storage = storage
        else:
            raise TypeError(f"Expected nested tuple/dict of tensors, found {type(ref_data_point)}.")

    def _write_dataset(self, ds: torch.utils.data.Dataset) -> None:
        ref_data_point = ds[0][0]
        if isinstance(ref_data_point, torch.Tensor):
            start_index = 0
            tensor_shapes = []
            start_indices = []
            num_elements = []
            for ds_idx in range(len(ds)):
                curr_elem = ds[ds_idx][0]
                tensor_shapes.append(curr_elem.shape)
                start_indices.append(start_index)
                num_elements.append(curr_elem.numel())
                self._storage[start_index : start_index + curr_elem.numel()] = curr_elem.view(-1)
                start_index += curr_elem.numel()
            total_elements = sum(num_elements)
        elif isinstance(ref_data_point, tuple):
            start_index = [0] * len(ref_data_point)
            tensor_shapes = []
            start_indices = []
            num_elements = []
            for ds_idx in range(len(ds)):
                curr_elem = ds[ds_idx][0]
                tensor_shapes.append([e.shape for e in curr_elem])
                num_elements.append([e.numel() for e in curr_elem])
                start_indices.append(start_index[:])
                for e_index, e in enumerate(curr_elem):
                    self._storage[e_index][
                        start_index[e_index] : start_index[e_index] + e.numel()
                    ] = e.view(-1)
                    start_index[e_index] += e.numel()
            total_elements = list(map(sum, zip(*num_elements)))
        elif isinstance(ref_data_point, dict):
            start_index = {k: 0 for k in ref_data_point.keys()}
            tensor_shapes = {k: [] for k in ref_data_point.keys()}
            start_indices = {k: [] for k in ref_data_point.keys()}
            num_elements = {k: [] for k in ref_data_point.keys()}
            for ds_idx in range(len(ds)):
                curr_elem = ds[ds_idx][0]
                for k, v in curr_elem.items():
                    tensor_shapes[k].append(v.shape)
                    num_elements[k].append(v.numel())
                    start_indices[k].append(start_index[k])
                    self._storage[k][start_index[k] : start_index[k] + v.numel()] = v.view(-1)
                    start_index[k] += v.numel()
            total_elements = {k: sum(v) for k, v in num_elements.items()}
        ## save meta data
        torch.save(
            [tensor_shapes, num_elements, total_elements, start_indices],
            os.path.join(self._directory, "storage_metadata.pt"),
        )

    def dump_dataset(self, ds: torch.utils.data.Dataset) -> None:
        # There are two steps here:
        # 1. Count number of elemenets: We need to consider all possibilities of tensor, tuple, or
        # dictionary of tensors.
        # 2. Write it in so that retrieval becomes a indexing problem. This needs extra info that
        # stores sizes and possibly offset of starting index of the current image.
        self._create_mmap_tensors(ds)
        self._write_dataset(ds)

    def _read_mmap_tensors(self, num_elements: Union[int, list, dict]) -> None:
        if isinstance(self._data_point, torch.Tensor):
            filename = f"{self._directory}.pt"
            self._storage = mmap_tensor(
                filename, size=(num_elements,), dtype=self._data_point.dtype
            )
        elif isinstance(self._data_point, tuple):
            self._storage = tuple(
                mmap_tensor(
                    os.path.join(self._directory, f"{i}.pt"),
                    num_elements[i],
                    dtype=self._data_point[i].dtype,
                )
                for i in range(len(self._data_point))
            )
        elif isinstance(self._data_point, dict):
            self._storage = {
                key: mmap_tensor(
                    os.path.join(self._directory, f"{key}.pt"),
                    num_elements[key],
                    dtype=self._data_point[key].dtype,
                )
                for key in self._data_point
            }

    def load_dataset(self):
        # There are two steps here:
        # 1. Load the meta data object.
        # 2. Access meta data and index the storage memmap
        tensor_shapes, num_elements, total_elements, start_indices = torch.load(
            os.path.join(self._directory, "storage_metadata.pt")
        )
        self._read_mmap_tensors(num_elements=total_elements)
        self._tensor_shapes = tensor_shapes
        self._start_indices = start_indices
        self._num_elements = num_elements

    def __getitem__(self, idx: int) -> NestedTensors:
        if self._storage is None:
            self.load_dataset()

        if isinstance(self._data_point, torch.Tensor):
            shape = self._tensor_shapes[idx]
            start_idx = self._start_indices[idx]
            num_elem = self._num_elements[idx]
            ret_val = self._storage[start_idx : start_idx + num_elem].view(shape)
        elif isinstance(self._data_point, tuple):
            ret_val = []
            for i in range(len(self._data_point)):
                shape = self._tensor_shapes[idx][i]
                start_idx = self._start_indices[idx][i]
                num_elem = self._num_elements[idx][i]
                ret_val.append(self._storage[i][start_idx : start_idx + num_elem].view(shape))
            ret_val = tuple(ret_val)
        elif isinstance(self._data_point, dict):
            ret_val = {}
            for k in self._data_point.keys():
                shape = self._tensor_shapes[k][idx]
                start_idx = self._start_indices[k][idx]
                num_elem = self._num_elements[k][idx]
                ret_val[k] = self._storage[k][start_idx : start_idx + num_elem].view(shape)

        return ret_val
