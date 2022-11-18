# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Lambda, RandomRotation

from renate import defaults
from renate.data.data_module import RenateDataModule
from renate.data.datasets import _TransformedDataset
from renate.utils.pytorch import get_generator, randomly_split_data


class Scenario(abc.ABC):
    """Creates a continual learning scenario from a RenateDataModule.

    This class can be extended to modify the returned training/validation/test sets
    to implement different experimentation settings.

    Note that many scenarios implemented here perform randomized operations, e.g., to split a base
    dataset into chunks. The scenario is only reproducible if the _same_ seed is provided in
    subsequent instantiatons. The seed argument is required for these scenarios.

    Args:
        data_module: The source RenateDataModule for the the user data.
        num_tasks: The total number of expected tasks for experimentation.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        num_tasks: int,
        chunk_id: int,
        seed: int = defaults.SEED,
    ) -> None:
        super().__init__()
        self._data_module = data_module
        self._num_tasks = num_tasks
        self._verify_chunk_id(chunk_id)
        self._chunk_id = chunk_id
        self._seed = seed
        self._train_data: Dataset = None
        self._val_data: Dataset = None

    def prepare_data(self) -> None:
        """Downloads datasets."""
        self._data_module.prepare_data()

    @abc.abstractmethod
    def setup(self, stage: Optional[str] = None, chunk_id: Optional[int] = None) -> None:
        """Sets up the scenario."""
        pass

    def train_data(self) -> Dataset:
        """Returns training dataset with respect to current `chunk_id`."""
        return self._train_data

    def val_data(self) -> Dataset:
        """Returns validation dataset with respect to current `chunk_id`."""
        return self._val_data

    def test_data(self) -> List[Dataset]:
        """Returns the test data with respect to all tasks in `num_tasks`."""
        datasets = []
        for chunk_id in range(self._num_tasks):
            self.setup(stage="test", chunk_id=chunk_id)
            datasets.append(self._data_module.test_data())
        return datasets

    def _verify_chunk_id(self, chunk_id: int) -> None:
        """A helper function to verify that the `chunk_id` is valid."""
        assert 0 <= chunk_id < self._num_tasks

    def _split_and_assign_train_and_val_data(
        self, stage: Optional[str] = None, chunk_id: Optional[int] = None
    ) -> None:
        """Performs train/val split and assigns the `train_data` and `val_data` attributes."""
        if chunk_id is None:
            chunk_id = self._chunk_id
        self._verify_chunk_id(chunk_id)
        proportions = [1 / self._num_tasks for _ in range(self._num_tasks)]
        if stage == "train" or stage is None:
            train_data = self._data_module.train_data()
            self._train_data = randomly_split_data(train_data, proportions, self._seed)[chunk_id]
        if (stage == "val" or stage is None) and self._data_module.val_data():
            val_data = self._data_module.val_data()
            self._val_data = randomly_split_data(val_data, proportions, self._seed)[chunk_id]


class BenchmarkScenario(Scenario):
    """This is a scenario to concatenate test data of a data module, which by definition has different chunks."""

    def setup(self, stage: Optional[str] = None, chunk_id: Optional[int] = None) -> None:
        self._data_module.setup(stage=stage, chunk_id=chunk_id)
        self._train_data = self._data_module.train_data()
        self._val_data = self._data_module.val_data()


class ClassIncrementalScenario(Scenario):
    """A scenario that creates data chunks from datasamples with specific classes from a data module.

    This class, upon giving a list describing the separation of the dataset separates the dataset
    with respect to classification labels.

    Note that, in order to apply this scenario, the scenario assumes that the datapoints in the data module
    are organised into tuples of exactly 2 tensors i.e. `(x, y)` where `x` is the input and `y` is the class id.

    Args:
        data_module: The base data module.
        num_tasks: The total number of expected tasks for experimentation.
        chunk_id: The data chunk to load in for the training or validation data.
        class_groupings: List of lists, describing the division of the classes for respective tasks.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        num_tasks: int,
        chunk_id: int,
        class_groupings: List[List[int]],
    ) -> None:
        super().__init__(data_module, num_tasks, chunk_id)
        assert len(class_groupings) == num_tasks
        self._class_groupings = class_groupings

    def setup(self, stage: Optional[str] = None, chunk_id: Optional[int] = None) -> None:
        """Make assignments: val/train/test splits."""
        self._data_module.setup(stage)
        if chunk_id is None:
            chunk_id = self._chunk_id

        self._verify_chunk_id(chunk_id)

        if stage == "train" or stage is None:
            self._train_data = self._get_task_subset(
                self._data_module.train_data(), chunk_id=chunk_id
            )

        if (stage == "val" or stage is None) and self._data_module.val_data():
            self._val_data = self._get_task_subset(self._data_module.val_data(), chunk_id=chunk_id)

    def test_data(self) -> List[Dataset]:
        """Returns the test data with respect to all tasks with respect to `num_tasks`."""
        self._data_module.setup(stage="test")
        datasets = []
        dataset = self._data_module.test_data()
        for chunk_id in range(self._num_tasks):
            datasets.append(self._get_task_subset(dataset, chunk_id))
        return datasets

    def _get_task_subset(self, dataset: Dataset, chunk_id: int) -> Dataset:
        """A helper function identifying indices corresponding to given classes."""
        self._verify_chunk_id(chunk_id)
        class_group = self._class_groupings[chunk_id]
        indices = torch.tensor(
            [i for i in range(len(dataset)) if dataset[i][1] in class_group],
            dtype=torch.long,
        )
        return Subset(dataset, indices)


class TransformScenario(Scenario):
    """A scenario that applies a different transformation to each chunk.

    The base ``data_module`` is split into ``len(transforms)`` random chunks. Then ``transforms[i]``
    is applied to chunk ``i``.

    Args:
        data_module: The base data module.
        transforms: A list of transformations.
        chunk_id: The id of the chunk to retrieve.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        transforms: List[Callable],
        chunk_id: int,
        seed: int = defaults.SEED,
    ) -> None:
        num_tasks = len(transforms)
        super().__init__(data_module, num_tasks, chunk_id, seed)
        self._transforms = transforms

    def setup(self, stage: Optional[str] = None, chunk_id: Optional[int] = None) -> None:
        if chunk_id is None:
            chunk_id = self._chunk_id
        self._verify_chunk_id(chunk_id)
        self._data_module.setup(stage)
        self._split_and_assign_train_and_val_data(stage, chunk_id)
        if stage == "train" or stage is None:
            self._train_data = _TransformedDataset(
                self._train_data, transform=self._transforms[chunk_id]
            )
        if (stage == "val" or stage is None) and self._val_data:
            self._val_data = _TransformedDataset(
                self._val_data, transform=self._transforms[chunk_id]
            )

    def test_data(self) -> List[Dataset]:
        """Returns the test data for all tasks."""
        self._data_module.setup(stage="test")
        dataset = self._data_module.test_data()
        datasets = []
        for chunk_id in range(self._num_tasks):
            datasets.append(_TransformedDataset(dataset, transform=self._transforms[chunk_id]))
        return datasets


class ImageRotationScenario(TransformScenario):
    """A scenario that rotates the images in the dataset by a different angle for each chunk.

    Args:
        data_module: The base data module.
        degrees: List of degrees corresponding to different tasks.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        degrees: List[int],
        chunk_id: int,
        seed: int,
    ) -> None:
        transforms = [RandomRotation(degrees=(deg, deg)) for deg in degrees]
        super().__init__(data_module, transforms, chunk_id, seed)


class PermutationScenario(TransformScenario):
    """A scenario that applies a different random permutation of features for each chunk.

    Args:
        data_module: The base data module.
        num_tasks: The total number of expected tasks for experimentation.
        input_dim: Dimension of the inputs. Can be a shape tuple or the total number of features.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: A random seed to fix the random number generation for permutations.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        num_tasks: int,
        input_dim: Union[List[int], Tuple[int], int],
        chunk_id: int,
        seed: int,
    ) -> None:
        input_dim = np.prod(input_dim)
        rng = get_generator(seed)
        transforms = [torch.nn.Identity()]
        for _ in range(num_tasks - 1):
            permutation = torch.randperm(input_dim, generator=rng)
            transform = Lambda(lambda x: x.flatten()[permutation].view(x.size()))
            transforms.append(transform)
        super().__init__(data_module, transforms, chunk_id, seed)
