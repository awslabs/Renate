# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from renate import defaults
from renate.data.datasets import IndexedSubsetDataset
from renate.memory.storage import FileTensorStorage
from renate.types import NestedTensors
from renate.utils.pytorch import get_generator

DataDict = Dict[str, torch.Tensor]


class DataBuffer(Dataset):
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

    In addition to passing metadata to the `update` method, one also access and replace the metadata
    in the buffer via the `get_metadata` and `set_metadata` methods.

    Note that, in order to apply transformations, the buffer assumes that the data points are tuples
    of the form `(x, y)`. We apply `transform` to `inputs` and `target_transform` to `y`. Ensure
    that the transforms accept the correct type, e.g., if `x` is a dictionary, `transform` needs to
    operate on a dictionary.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        seed: int = defaults.SEED,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self._max_size = max_size
        self._seed = seed
        self._transform = transform
        self._target_transform = target_transform

        self._rng = get_generator(self._seed)

        self._count = 0
        self._datasets = []
        self._indices = {}
        self._data_point_prototype = None
        self._metadata = {}

    def __len__(self) -> int:
        """Returns the current length of the buffer."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[NestedTensors, Dict[str, Any]]:
        """Reads the item at index `idx` of the buffer."""
        i, j = self._indices[idx]
        data = self._datasets[i][j]
        metadata = {key: value[idx] for key, value in self._metadata.items()}
        if self._transform is None and self._target_transform is None:
            return data, metadata
        else:
            inputs, targets = data
            if self._transform is not None:
                inputs = self._transform(inputs)
            if self._target_transform is not None:
                targets = self._target_transform(targets)
            return (inputs, targets), metadata

    def update(self, dataset: Dataset, metadata: Optional[Dict] = None) -> None:
        """Updates the buffer with a new dataset."""
        metadata = metadata or {}
        self._check_metadata_internal_consistency(metadata, expected_length=len(dataset))
        if not len(self):
            self._data_point_prototype = copy.deepcopy(dataset[0])
            self._add_metadata_like(metadata)
        else:
            self._check_metadata_compatibility(metadata)
        assignments = self._update(dataset, metadata)
        self._check_assignments_are_contiguous(assignments)

        # Perform assignments.
        d = len(self._datasets)
        self._datasets.append(dataset)
        for buffer_idx, dataset_idx in assignments.items():
            self._indices[buffer_idx] = (d, dataset_idx)
            for key in self._metadata.keys():
                self._metadata[key][buffer_idx] = metadata[key][dataset_idx]

    def _update(self, dataset: Dataset, metadata: Dict) -> Dict[int, int]:
        """Returns the updates.

        This method is used to implement different buffering methods. Given a dataset, it returns a
        dictionary of `assignments` indicating the updates to be performed to the buffer.

        Args:
            dataset: The dataset used to update the buffer.

        Returns:
            assignments: A dictionary mapping int to int. The presence of the pair `(i, j)` means
                that the `i`-th buffer slot will be filled with the `j`-th element of `dataset`.
        """
        raise NotImplementedError()

    def get_metadata(self, key: str) -> Dict[str, torch.Tensor]:
        return self._metadata[key][: len(self)]

    def set_metadata(self, key: str, values: torch.Tensor) -> None:
        if values.size(0) != len(self):
            raise ValueError()
        if key not in self._metadata:
            self._add_metadata_like({key: values})
        self._metadata[key][: len(self)] = values.cpu()

    def state_dict(self) -> Dict:
        return {
            "buffer_class_name": self.__class__.__name__,
            "max_size": self._max_size,
            "seed": self._seed,
            "count": self._count,
            "indices": self._indices,
            "data_point_prototype": self._data_point_prototype,
            "metadata": self._metadata,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        if self.__class__.__name__ != state_dict["buffer_class_name"]:
            raise RuntimeError(
                f"Buffer of class {self.__class__} was used to load a state dict created by class "
                f"{state_dict['buffer_class_name']}."
            )
        self._max_size = state_dict["max_size"]
        self._seed = state_dict["seed"]
        self._count = state_dict["count"]
        self._indices = state_dict["indices"]
        self._data_point_prototype = state_dict["data_point_prototype"]
        self._metadata = state_dict["metadata"]
        self._rng = get_generator(self._seed)

    def save(self, target_dir: str) -> None:
        if not len(self):
            return

        transforms = self._transform, self._target_transform
        self._transform, self._target_transform = None, None
        storage = FileTensorStorage(target_dir)
        storage.dump_dataset(IndexedSubsetDataset(self, [0]))
        self._datasets = [storage]
        self._indices = {i: (0, i) for i in range(len(self))}
        self._transform, self._target_transform = transforms

    def load(self, source_dir: str) -> None:
        if not len(self):
            return

        storage = FileTensorStorage(source_dir)
        self._datasets = [storage]

    def _add_metadata_like(self, metadata: DataDict):
        for key, value in metadata.items():
            self._metadata[key] = torch.empty(
                (self._max_size, *value.size()[1:]), dtype=value.dtype, device="cpu"
            )

    def _check_assignments_are_contiguous(self, assignments):
        # The subclass _update method is expected to only return assignments that populate the
        # buffer contiguously. We check this here. The invariant we expect to hold is that, at any
        # time, the indices 0:len(buffer) are filled.
        current_length = len(self)
        new_idx = [i for i in assignments.keys() if i > current_length]
        if new_idx:
            assert max(new_idx) == current_length + len(new_idx)

    @staticmethod
    def _check_metadata_internal_consistency(metadata: DataDict, expected_length: int):
        """Checks metadata for internal consistency and an expected length."""
        for key in metadata:
            if not isinstance(metadata[key], torch.Tensor):
                raise TypeError(
                    f"Expected tensors in `metadata`, found {type(metadata[key])} at key {key}."
                )
            if not metadata[key].size(0) == expected_length:
                raise ValueError(
                    f"Tensors in metadata dictionary need to be of size {expected_length} (size "
                    f"of the associated dataset) in dimension 0. Found size {metadata[key].size()} "
                    f"at key {key}."
                )

    def _check_metadata_compatibility(self, metadata: DataDict) -> None:
        """Verifies that passed metadata is compatible with internal metadata."""
        if not len(self):
            return
        if set(self._metadata.keys()) != set(metadata.keys()):
            raise KeyError(
                f"Keys of provided metadata {list(metadata.keys())} do not match those present in ",
                f"the buffer {list(self._metadata.keys())}.",
            )
        for key in metadata:
            if metadata[key].size()[1:] != self._metadata[key].size()[1:]:
                raise ValueError()
            if metadata[key].dtype != self._metadata[key].dtype:
                raise TypeError(
                    f"Provided metadata at key {key} is of type {metadata[key].dtype}. This does "
                    f"not match type {self._metadata[key].dtype} already present in the buffer."
                )


class ReservoirBuffer(DataBuffer):
    """A buffer implementing reservoir sampling."""

    def _update(self, dataset: Dataset, metadata: Dict) -> Dict[int, int]:
        assignments = {}
        for i in range(len(dataset)):
            if self._count < self._max_size:
                assignments[self._count] = i
            else:
                rand = torch.randint(low=0, high=self._count, size=(), generator=self._rng).item()
                if rand < self._max_size:
                    assignments[rand] = i
            self._count += 1
        return assignments


class SlidingWindowBuffer(DataBuffer):
    """A buffer implementing a sliding window."""

    def _update(self, dataset: Dataset, metadata) -> Dict[int, int]:
        assignments = {}
        for i in range(len(dataset)):
            assignments[self._count % self._max_size] = i
            self._count += 1
        return assignments


class GreedyClassBalancingBuffer(DataBuffer):
    """A buffer implementing a greedy class-balancing approach."""

    def __init__(
        self,
        max_size: Optional[int] = None,
        seed: int = defaults.SEED,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(max_size, seed, transform, target_transform)
        self._indices_by_class = defaultdict(list)

    def state_dict(self) -> Dict:
        state_dict = super().state_dict()
        state_dict["indices_by_class"] = self._indices_by_class
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        super().load_state_dict(state_dict)
        self._indices_by_class = defaultdict(list)
        self._indices_by_class.update(state_dict["indices_by_class"])

    def _get_largest_class(self):
        largest_classes = []
        max_length = 0
        for y, indices in self._indices_by_class.items():
            if len(indices) > max_length:
                largest_classes = [y]
                max_length = len(indices)
            if len(indices) == max_length:
                largest_classes.append(y)
        rand = torch.randint(low=0, high=len(largest_classes), size=(1,), generator=self._rng)
        largest_class = largest_classes[rand.item()]
        return largest_class, max_length

    def _update(self, dataset: Dataset, metadata: Dict) -> Dict[int, int]:
        assignments = {}
        for i in range(len(dataset)):
            y = int(dataset[i][1])
            if self._count < self._max_size:
                assignments[self._count] = i
                self._indices_by_class[y].append(self._count)
            else:
                largest_class, max_length = self._get_largest_class()
                rand = torch.randint(low=0, high=max_length, size=(1,), generator=self._rng).item()
                idx_to_replace = self._indices_by_class[largest_class].pop(rand)
                assignments[idx_to_replace] = i
                self._indices_by_class[y].append(idx_to_replace)
            self._count += 1
        return assignments


class InfiniteBuffer(DataBuffer):
    """A data buffer that stores _all_ incoming data."""

    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(max_size=1, transform=transform, target_transform=target_transform)

    def _update(self, dataset: Dataset, metadata: Dict) -> Dict[int, int]:
        # Resize metadata dictionary.
        if self._count + len(dataset) > self._max_size:
            while self._count + len(dataset) > self._max_size:
                self._max_size *= 2
            current_metadata = {key: self.get_metadata(key).clone() for key in self._metadata}
            self._add_metadata_like(current_metadata)
            for key, values in current_metadata.items():
                self.set_metadata(key, values)

        assignments = {self._count + i: i for i in range(len(dataset))}
        self._count += len(dataset)
        return assignments
