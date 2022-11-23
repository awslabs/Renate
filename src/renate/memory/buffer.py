# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from renate import defaults
from renate.utils.pytorch import get_generator

DataTuple = Tuple[torch.Tensor, ...]
DataDict = Dict[str, torch.Tensor]


class DataBuffer(Dataset, ABC):
    """A memory buffer storing data points.

    The buffer functions as a torch dataset, i.e., it implements `__len__` and
    `__getitem__`. Pytorch data loaders can be used to sample from or iterate
    over the buffer.

    Extracting an element from the buffer will return a tuple,
    `data_point, metadata = buffer[i]`, where `data_point` is the raw data point
    and `metadata` is a dictionary containing associated metadata (identical
    keys for all data points, possibly empty) and the index `idx` of the currently
    sampled data sample, referring to the ordering inside of the buffer.
    It is assumed that all passed in data are PyTorch tensors.
    Metadata can be passed externally when calling `Buffer.update` and
    additional fields of metadata might be added by some buffering methods, e.g.,
    instance weights in coreset methods.

    Note that the buffer does not change the device placement of data passed to it.
    Please ensure that the data passed to `DataBuffer.update` resides on the CPU.

    Note that, in order to apply transformations, the buffer assumes that the datapoints
    are tuples of exactly 2 tensors i.e. `(x, y)` where `x` is the input and `y` is some target.

    Args:
        max_size: Maximal size of the buffer.
        storage_mode: How to store the data in the buffer. Currently, we only
           support `in_memory`.
        seed: Seed for the random number generator used in the buffer.
        transform: The transformation to be applied to the memory buffer data samples.
        target_transform: The target transformation to be applied to the memory buffer target samples.
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

        self._data_points: DataDict = {}
        self.metadata: DataDict = {}
        self._size = 0

        self._transform = transform
        self._target_transform = target_transform

    def __len__(self) -> int:
        """Returns the number of data points in the buffer."""
        return self._size

    def __getitem__(self, idx: int) -> Tuple[DataTuple, DataDict]:
        """Retrieves a data point from the buffer."""
        metadata = {key: value[idx] for key, value in self.metadata.items()}
        metadata["idx"] = torch.tensor(idx, dtype=torch.long)
        data = tuple(self._data_points[f"{i}"][idx] for i in range(len(self._data_points)))
        if self._transform is not None or self._target_transform is not None:
            x, y = data
            if self._transform is not None:
                x = self._transform(x)
            if self._target_transform is not None:
                y = self._target_transform(y)
            return (x, y), metadata
        return data, metadata

    def _verify_metadata(self, metadata: DataDict, expected_length: int) -> None:
        """Verifies that passed metadata is compatible with internal metadata."""
        if len(self) == 0:
            return

        if set(self.metadata.keys()) != set(metadata.keys()):
            raise KeyError(
                f"Keys of provided metadata {list(metadata.keys())} do not match those present in ",
                f"the buffer {list(self.metadata.keys())}.",
            )

        for key in metadata:
            if not isinstance(metadata[key], torch.Tensor):
                raise TypeError(
                    f"Metadata needs to map to `torch.Tensor`, found {type(metadata[key])} at ",
                    f"key {key}.",
                )
            if metadata[key].dtype != self.metadata[key].dtype:
                raise TypeError(
                    f"Provided metadata at key {key} is of type {metadata[key].dtype}. This does "
                    f"not match type {self.metadata[key].dtype} already present in the buffer."
                )
            if metadata[key].size(0) != expected_length:
                raise ValueError(
                    f"Tensors in metadata dictionary need to be of size {expected_length} (size ",
                    f"of the associated dataset) in dimension 0. Found size {metadata[key].size()} ",
                    f"at key {key}.",
                )

    def __setitem__(self, idx: int, data_and_metadata: Tuple[DataTuple, DataDict]) -> None:
        """Replaces a data point in the buffer."""
        data, metadata = data_and_metadata
        for i, d in enumerate(data):
            self._data_points[f"{i}"][idx] = d
        for key in metadata:
            self.metadata[key][idx] = metadata[key]

    def _append(self, data: DataTuple, metadata: DataDict) -> None:
        """Appends a data point to the internal storage."""
        for i, d in enumerate(data):
            key = f"{i}"  # FIXME: choose a better naming for the keys of the data points coming from the dataset
            if key not in self._data_points:
                self._data_points[key] = torch.empty(
                    (self._max_size, *d.shape), dtype=d.dtype, device=d.device
                )

        for key in metadata:
            if key not in self.metadata:
                self.metadata[key] = torch.empty(
                    (self._max_size, *metadata[key].shape),
                    dtype=metadata[key].dtype,
                    device=metadata[key].device,
                )
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
        self._verify_metadata(metadata, expected_length=len(dataset))
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

        if not isinstance(state_dict["data_points"], dict):
            raise TypeError("Invalid container for data points, should be a dictionary.")

        if not isinstance(state_dict["metadata"], dict):
            raise TypeError("Invalid container for metadata, should be a dictionary.")


class InfiniteBuffer(DataBuffer):
    """A data buffer that supports infinite size.

    Args:
        storage_mode: How to store the data in the buffer. Currently, we only
           support `in_memory`.
        transform: The transformation to be applied to the memory buffer data samples.
        target_transform: The target transformation to be applied to the memory buffer target samples.
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
        self._max_size = 1

    def _update(self, dataset: Dataset, metadata: DataDict) -> None:
        for i in range(len(dataset)):
            self._append(dataset[i], {key: value[i] for key, value in metadata.items()})

    def _append(self, data: DataTuple, metadata: DataDict) -> None:
        """Appends a data point to the internal storage.

        Initializes a new buffer with twice the size if the current one is full.
        """
        super()._append(data, metadata)
        if self._size == self._max_size:
            data_containers = [self._data_points, metadata]
            for data_container in data_containers:
                for key in data_container:
                    data_container[key] = torch.cat(
                        [data_container[key], torch.empty_like(data_container[key])], dim=0
                    )
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

    Note that, this implementation works only with respect to classification problems
    and datasets, where we expect the data coming from the dataset to be organised
    as `x,y` tuples where `x` is the input and `y` is an integer class index.
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

    def _record_class_index(self, data: DataTuple, target_index: int) -> None:
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
