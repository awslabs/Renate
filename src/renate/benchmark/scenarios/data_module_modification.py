# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import copy
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, RandomRotation
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from renate import defaults
from renate.data.data_module import RenateDataModule
from renate.utils.pytorch import get_generator, randomly_split_data


class Scenario(abc.ABC):
    """An abstract class modifying a RenateDataModule to provide scenarios for continual learning experiments.

    This class can be extended to modify the returned training/validation/test sets
    to implement different experimentation settings.

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
        """Sets up the scenario for training/validation/test and perform the splitting of the data for train and validation."""
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

    def _insert_data_module_transform(self, transform: Callable, position: int) -> None:
        """A helper function to insert a datasample transformation at a chosen position with respect to `_transform`."""
        if self._data_module._transform is None:
            self._data_module._transform = transform
        elif isinstance(self._data_module._transform, Compose):
            self._data_module._transform.transforms.insert(position, transform)
        else:
            raise ValueError("Unable to insert the Scenario modification transformation.")

    def _verify_chunk_id(self, chunk_id: int) -> None:
        """A helper function to verify that the `chunk_id` is valid."""
        assert 0 <= chunk_id < self._num_tasks

    def _split_and_assign_train_and_val_data(
        self, stage: Optional[str] = None, chunk_id: Optional[int] = None
    ) -> None:
        """A helper function to split the data into train and validation sets and assign them to the `train_data` and `val_data` attributes."""
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


class ImageRotationScenario(Scenario):
    """A scenario that rotates the images in the dataset.

    This class rotates the entire training/validation/test set with respect to specified degree,
    depending on the provided `task_index` in the `.setup()` method.

    Args:
        data_module: The source RenateDataModule for the the user data.
        num_tasks: The total number of expected tasks for experimentation.
        chunk_id: The data chunk to load in for the training or validation data.
        degrees: List of degrees corresponding to different tasks.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        num_tasks: int,
        chunk_id: int,
        degrees: List[int],
        seed: int = defaults.SEED,
    ) -> None:
        super().__init__(data_module, num_tasks, chunk_id, seed)
        assert len(degrees) == num_tasks
        self._degrees = degrees

    def setup(self, stage: Optional[str] = None, chunk_id: Optional[int] = None) -> None:
        """Make assignments: val/train/test splits.

        Adjusts the transformation in the original RenateDataModule to put the rotation as the first
        augmentation applied in series. If the transformation is `None` it sets it to rotation.
        """
        if chunk_id is None:
            chunk_id = self._chunk_id
        self._verify_chunk_id(chunk_id)
        original_transform = copy.deepcopy(self._data_module._transform)
        rotation = RandomRotation(degrees=(self._degrees[chunk_id], self._degrees[chunk_id]))
        self._insert_data_module_transform(transform=rotation, position=0)
        self._data_module.setup(stage)
        self._split_and_assign_train_and_val_data(stage, chunk_id)
        self._data_module._transform = original_transform


class ClassIncrementalScenario(Scenario):
    """A scenario that creates data chunks from datasamples with specific classes from a data module.

    This class, upon giving a list describing the separation of the dataset separates the dataset
    with respect to classification labels.

    Note that, in order to apply this scenario, the scenario assumes that the datapoints in the data module
    are organised into tuples of exactly 2 tensors i.e. `(x, y)` where `x` is the input and `y` is the class id.

    Args:
        data_module: The source RenateDataModule for the the user data.
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


class Permutation:
    """Permute the input with respect to the indices provided.

    It makes the assumption that the input can be linearized and its entries can be
    indexed.

    If the input is a PIL image, the return of the call is also a PIL image. If the input is torch tensor,
    the output is a torch tensor.

    Args:
        indices: A one dimensional tensor specifying the indices for permutation.
    """

    def __init__(self, indices: torch.Tensor) -> None:
        self._indices = indices

    def __call__(
        self, sample: Union[torch.Tensor, Image.Image]
    ) -> Union[torch.Tensor, Image.Image]:
        pil_image = False
        pil_image_mode = None
        if isinstance(sample, Image.Image):
            pil_image_mode = sample.mode
            sample = pil_to_tensor(sample)
            pil_image = True
        original_shape = sample.shape
        sample = sample.reshape(-1)
        if len(sample) != len(self._indices):
            raise ValueError(
                f"The length of the indices does not match the size of the data: {original_shape}!={len(self._indices)}."
            )

        permuted_sample = sample[self._indices]
        permuted_sample = permuted_sample.reshape(*original_shape)
        if pil_image:
            permuted_sample = to_pil_image(permuted_sample, pil_image_mode)
        return permuted_sample


class PermutationScenario(Scenario):
    """A scenario that creates data chunks from datasamples with specific classes from a data module.

    This class, given the input feature size and a seed, permutes the datasample features.
    The permutations are precomputed depending on the total number of tasks. The first permutation
    is the original datasample without modification.

    Args:
        data_module: The source RenateDataModule for the the user data.
        num_tasks: The total number of expected tasks for experimentation.
        chunk_id: The data chunk to load in for the training or validation data.
        input_dim: List of input dimensions for the input or the overall input dimensionality as an integer.
        seed: A random seed to fix the random number generation for permutations.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        num_tasks: int,
        chunk_id: int,
        input_dim: Union[List[int], Tuple[int], int],
        seed: int = defaults.SEED,
    ) -> None:
        super().__init__(data_module, num_tasks, chunk_id, seed)
        input_dim = np.prod(input_dim)
        rng = get_generator(seed)
        self._indices = [
            torch.randperm(input_dim, generator=rng).long() for _ in range(num_tasks - 1)
        ]

    def setup(self, stage: Optional[str] = None, chunk_id: Optional[int] = None) -> None:
        """Make assignments: val/train/test splits.

        Adjusts the transformation in the original RenateDataModule to put the rotation as the first
        augmentation applied in series. If the transformation is `None` it sets it to rotation.
        """
        if chunk_id is None:
            chunk_id = self._chunk_id
        self._verify_chunk_id(chunk_id)
        original_transform = copy.deepcopy(self._data_module._transform)
        if chunk_id != 0:
            permutation = Permutation(indices=self._indices[chunk_id - 1])
            self._insert_data_module_transform(transform=permutation, position=0)
        self._data_module.setup(stage)
        self._split_and_assign_train_and_val_data(stage, chunk_id)
        self._data_module._transform = original_transform
