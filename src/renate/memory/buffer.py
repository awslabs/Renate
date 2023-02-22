# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Hashable, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from renate import defaults
from renate.utils.pytorch import get_generator


DataTuple = Tuple[torch.Tensor, ...]
DataDict = Dict[Hashable, torch.Tensor]
NestedTensors = Union[torch.Tensor, DataTuple, DataDict]
Index = Union[int, slice]


def _make_storage(data_point: NestedTensors, length: int) -> NestedTensors:
    """Creates a nested tensor storage to store datapoints of sizes/shapes given by `data_point`.

    `data_point` can be any nested structure (using `tuple` and `dict`) containing `torch.Tensor`s.
    This function will return an equivalent nested structure with tensors of the same data type
    and shape with an additional "batch dimension" of size `length`.

    Args:
        data_point: A nested structure of tensors.
        length: Number of data points (like `data_point`) for which a storage will be allocated.

    Returns:
        A storage that can store `length` points like `data_point`.
    """
    if isinstance(data_point, torch.Tensor):
        return torch.empty(size=(length, *data_point.size()), dtype=data_point.dtype)
    elif isinstance(data_point, tuple):
        return tuple(_make_storage(t, length) for t in data_point)
    elif isinstance(data_point, dict):
        return {key: _make_storage(t, length) for key, t in data_point.items()}
    else:
        raise TypeError(f"Expected nested tuple/dict of tensors, found {type(data_point)}.")


def _get_data_point(storage: NestedTensors, idx: Index) -> NestedTensors:
    """Retrieves a data point from a nested tensor storage.

    Args:
        storage: A storage for nested tensors, e.g., created using `make_storage`.
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
        storage: A storage for nested tensors, e.g., created using `make_storage`.
        idx: An integer index or slice indicating where to insert the data point(s)
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


class DataBuffer(Dataset, ABC):
    """A memory buffer storing data points.

    The buffer functions as a torch dataset, i.e., it implements `__len__` and `__getitem__`.
    Pytorch data loaders can be used to sample from or iterate over the buffer.

    Data can be added to the buffer via `buffer.update(dataset, metadata)`. `dataset` is a
    pytorch dataset expected to return an arbitrary nested `tuple`/`dict` structure containing
    `torch.Tensor`s of _fixed_ size and data type. `metadata` is a dictionary mapping strings to
    tensors for associated metadata. The logic to decide which data points remain in the buffer is
    implemented by different subclasses.

    Extracting an element from the buffer will return a nested tuple of the form
    `data_point, metadata = buffer[i]`, where `data_point` is the raw data point  and `metadata` is
    a dictionary containing associated metadata as well as field `idx` containing the index of the
    data point in the buffer. Additional fields of metadata might be added by some buffering
    methods, e.g., instance weights in coreset methods.

    Note that the buffer does not change the device placement of data passed to it. Please ensure
    that the data passed to `DataBuffer.update` resides on the CPU.

    Note that, in order to apply transformations, the buffer assumes that the data points are tuples
    of the form `(x, y)`. We apply `transform` to `inputs` and `target_transform` to `y`. Ensure
    that the transforms accept the correct type, e.g., if `x` is a dictionary, `transform` needs to
    operate on a dictionary.

    Args:
        max_size: Maximal size of the buffer.
        storage_mode: How to store the data in the buffer. Currently, we only support `in_memory`.
        seed: Seed for the random number generator used in the buffer.
        transform: The transformation to be applied to the inputs "x" of points in the buffer.
        target_transform: The transformation to be applied to target "y" of points in the buffer.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        storage_mode: defaults.SUPPORTED_BUFFER_STORAGE_MODES = defaults.BUFFER_STORAGE_MODE,
        seed: int = defaults.SEED,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self._max_size = max_size
        self._storage_mode = storage_mode
        self._seed = seed

        self._rng = get_generator(self._seed)

        self._count = 0

        self._data_points = None
        self.metadata = None
        self._size = 0

        self._transform = transform
        self._target_transform = target_transform

    def __len__(self) -> int:
        """Returns the number of data points in the buffer."""
        return self._size

    def __getitem__(self, idx: int) -> Tuple[NestedTensors, DataDict]:
        """Retrieves a data point from the buffer."""
        metadata = _get_data_point(self.metadata, idx)
        data = _get_data_point(self._data_points, idx)
        if self._transform is None and self._target_transform is None:
            return data, metadata
        else:
            inputs, targets = data
            if self._transform is not None:
                inputs = self._transform(inputs)
            if self._target_transform is not None:
                targets = self._target_transform(targets)
            return (inputs, targets), metadata

    def __setitem__(self, idx: int, data_and_metadata: Tuple[NestedTensors, DataDict]) -> None:
        """Replaces a data point in the buffer."""
        data, metadata = data_and_metadata
        _insert_data_point(self._data_points, idx, data)
        _insert_data_point(self.metadata, idx, metadata)

    def _append(self, data: NestedTensors, metadata: DataDict) -> None:
        """Appends a data point to the internal storage."""
        if not len(self):
            self._data_points = _make_storage(data, self._max_size)
            self.metadata = _make_storage(metadata, self._max_size)
        self[self._size] = data, metadata
        self._size += 1

    def update(self, dataset: Dataset, metadata: Optional[DataDict] = None) -> None:
        """Updates the buffer with a new dataset.

        Args:
            dataset: A dataset containing a new chunk of data.
            metadata: A dictionary mapping to tensors, which are assumed to
                have size n in dimension 0, where `n = len(dataset)`.
        """
        metadata = metadata or {}
        return self._update(dataset, metadata)

    @abstractmethod
    def _update(self, dataset: Dataset, metadata: DataDict) -> None:
        pass

    def set_transforms(
        self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None
    ) -> None:
        """Update the transformations applied to the data."""
        self._transform = transform
        self._target_transform = target_transform

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the buffer as a dictionary."""
        return {
            "buffer_class_name": self.__class__.__name__,
            "max_size": self._max_size,
            "storage_mode": self._storage_mode,
            "seed": self._seed,
            "count": self._count,
            "size": self._size,
            "data_points": self._data_points,
            "metadata": self.metadata,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Assigns the values from `state_dict` into their respective attributes."""
        if self.__class__.__name__ != state_dict["buffer_class_name"]:
            raise RuntimeError(
                f"Buffer of class {self.__class__} was used to load a state dict created by class "
                f"{state_dict['buffer_class_name']}."
            )
        self._max_size = state_dict["max_size"]
        self._storage_mode = state_dict["storage_mode"]
        self._seed = state_dict["seed"]
        self._count = state_dict["count"]
        self._size = state_dict["size"]
        self._data_points = state_dict["data_points"]
        self.metadata = state_dict["metadata"]

        self._rng = get_generator(self._seed)

        for key in ["count", "size"]:
            if not isinstance(state_dict[key], int):
                raise TypeError(f"Invalid type for: {key}, should be int.")

        if not isinstance(state_dict["storage_mode"], str):
            raise TypeError("Invalid type for storage_mode, should be str.")

        if not (isinstance(state_dict["metadata"], dict) or state_dict["metadata"] is None):
            raise TypeError("Invalid container for metadata, should be a dictionary or None.")


class InfiniteBuffer(DataBuffer):
    """A data buffer that stores _all_ incoming data.

    Args:
        storage_mode: How to store the data in the buffer. Currently, we only support `in_memory`.
        transform: The transformation to be applied to the inputs "x" of points in the buffer.
        target_transform: The transformation to be applied to target "y" of points in the buffer.
    """

    def __init__(
        self,
        storage_mode: defaults.SUPPORTED_BUFFER_STORAGE_MODES = defaults.BUFFER_STORAGE_MODE,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            storage_mode=storage_mode, transform=transform, target_transform=target_transform
        )
        self._max_size = 16

    def _update(self, dataset: Dataset, metadata: DataDict) -> None:
        for i in range(len(dataset)):
            self._append(dataset[i], {key: value[i] for key, value in metadata.items()})

    def _append(self, data: NestedTensors, metadata: NestedTensors) -> None:
        """Appends a data point to the internal storage.

        Initializes a new buffer with twice the size if the current one is full.
        """
        super()._append(data, metadata)
        if self._size == self._max_size:
            # Extend the size of the storage by a factor of 2.
            current_data_points = self._data_points
            self._data_points = _make_storage(
                _get_data_point(current_data_points, 0), self._max_size * 2
            )
            _insert_data_point(self._data_points, slice(self._max_size), current_data_points)
            current_metadata = self.metadata
            self.metadata = _make_storage(_get_data_point(current_metadata, 0), self._max_size * 2)
            _insert_data_point(self.metadata, slice(self._max_size), current_metadata)
            self._max_size *= 2


class ReservoirBuffer(DataBuffer):
    """A buffer implementing reservoir sampling.

    Reservoir sampling maintains a uniform subset of the data seen so far.

    TODO: Adjust citation once we've agreed on a format.
    Jeffrey S. Vitter. 1985. Random sampling with a reservoir. ACM Trans.
    Math. Softw. 11, 1 (March 1985), 37–57. https://doi.org/10.1145/3147.3165
    """

    def _update(self, dataset: Dataset, metadata: DataDict) -> None:
        for i in range(len(dataset)):
            if len(self) < self._max_size:
                self._append(dataset[i], {key: value[i] for key, value in metadata.items()})
            else:
                rand = torch.randint(low=0, high=self._count, size=(), generator=self._rng).item()
                if rand < self._max_size:
                    self[rand] = (dataset[i], {key: value[i] for key, value in metadata.items()})
            self._count += 1


class SlidingWindowBuffer(DataBuffer):
    """A sliding window buffer, retaining the most recent data points."""

    def _update(self, dataset: Dataset, metadata: DataDict) -> None:
        for i in range(len(dataset)):
            if len(self) < self._max_size:
                self._append(dataset[i], {key: value[i] for key, value in metadata.items()})
            else:
                self[self._count % self._max_size] = (
                    dataset[i],
                    {key: value[i] for key, value in metadata.items()},
                )
            self._count += 1


class GreedyClassBalancingBuffer(DataBuffer):
    """A greedy class balancing buffer as proposed in:

    Prabhu, Ameya, Philip HS Torr, and Puneet K. Dokania. "GDumb: A simple
    approach that questions our progress in continual learning." ECCV, 2020.

    Note that, this implementation works only with for datasets returning `(x, y)` tuples, where we
    expect `y` to be an integer class label.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._class_to_index_map = defaultdict(list)

    def _update(self, dataset: Dataset, metadata: DataDict) -> None:
        for i in range(len(dataset)):
            if len(self) < self._max_size:
                self._append(dataset[i], {key: value[i] for key, value in metadata.items()})
                self._record_class_index(dataset[i], len(self) - 1)
            else:
                class_counts_max_value = max(
                    [len(self._class_to_index_map[key]) for key in self._class_to_index_map.keys()]
                )
                largest_classes = [
                    k
                    for k, v in self._class_to_index_map.items()
                    if len(v) == class_counts_max_value
                ]
                largest_class = largest_classes[
                    torch.randint(
                        low=0, high=len(largest_classes), size=(1,), generator=self._rng
                    ).item()
                ]
                memory_idx_to_replace = self._class_to_index_map[largest_class][
                    torch.randint(
                        low=0,
                        high=len(self._class_to_index_map[largest_class]),
                        size=(1,),
                        generator=self._rng,
                    ).item()
                ]
                self[memory_idx_to_replace] = (
                    dataset[i],
                    {key: value[i] for key, value in metadata.items()},
                )
                self._class_to_index_map[largest_class].remove(memory_idx_to_replace)
                self._record_class_index(dataset[i], memory_idx_to_replace)

            self._count += 1

    def _record_class_index(self, data: NestedTensors, target_index: int) -> None:
        """Helper function to record the class membership in the mapping."""
        _, y = data
        if isinstance(y, torch.Tensor):
            y = y.item()
        self._class_to_index_map[y].append(target_index)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the buffer as a dictionary."""
        state_dict = super().state_dict()
        state_dict["class_to_index_map"] = self._class_to_index_map
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._class_to_index_map = state_dict["class_to_index_map"]
