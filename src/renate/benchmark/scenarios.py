# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
from typing import Callable, List, Tuple, Union

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
    subsequent instantiations. The seed argument is required for these scenarios.

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
        self._test_data: List[Dataset] = None

    def prepare_data(self) -> None:
        """Downloads datasets."""
        self._data_module.prepare_data()

    @abc.abstractmethod
    def setup(self) -> None:
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
        return self._test_data

    def _verify_chunk_id(self, chunk_id: int) -> None:
        """A helper function to verify that the `chunk_id` is valid."""
        assert 0 <= chunk_id < self._num_tasks

    def _split_and_assign_train_and_val_data(self) -> None:
        """Performs train/val split and assigns the `train_data` and `val_data` attributes."""
        proportions = [1 / self._num_tasks for _ in range(self._num_tasks)]
        train_data = self._data_module.train_data()
        self._train_data = randomly_split_data(train_data, proportions, self._seed)[self._chunk_id]
        if self._data_module.val_data():
            val_data = self._data_module.val_data()
            self._val_data = randomly_split_data(val_data, proportions, self._seed)[self._chunk_id]


class BenchmarkScenario(Scenario):
    """This is a scenario to concatenate test data of a data module, which by definition has
    different chunks.
    """

    def setup(self) -> None:
        self._data_module.setup()
        self._train_data = self._data_module.train_data()
        self._val_data = self._data_module.val_data()
        self._test_data = self._data_module._test_data


class ClassIncrementalScenario(Scenario):
    """A scenario that creates data chunks from data samples with specific classes from a data
    module.

    This class, upon giving a list describing the separation of the dataset separates the dataset
    with respect to classification labels.

    Note that, in order to apply this scenario, the scenario assumes that the data points in the
    data module are organised into tuples of exactly 2 tensors i.e. `(x, y)` where `x` is the input
    and `y` is the class id.

    Args:
        data_module: The source RenateDataModule for the the user data.
        chunk_id: The data chunk to load in for the training or validation data.
        class_groupings: List of lists, describing the division of the classes for respective tasks.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        chunk_id: int,
        class_groupings: List[List[int]],
    ) -> None:
        super().__init__(data_module, len(class_groupings), chunk_id)
        self._class_groupings = class_groupings

    def setup(self) -> None:
        """Make assignments: val/train/test splits."""
        self._data_module.setup()
        self._train_data = self._get_task_subset(
            self._data_module.train_data(), chunk_id=self._chunk_id
        )
        if self._data_module.val_data():
            self._val_data = self._get_task_subset(
                self._data_module.val_data(), chunk_id=self._chunk_id
            )
        self._test_data = [
            self._get_task_subset(self._data_module.test_data(), i) for i in range(self._num_tasks)
        ]

    def _get_task_subset(self, dataset: Dataset, chunk_id: int) -> Dataset:
        """A helper function identifying indices corresponding to given classes."""
        class_group = self._class_groupings[chunk_id]
        indices = torch.tensor(
            [i for i in range(len(dataset)) if dataset[i][1] in class_group],
            dtype=torch.long,
        )
        subset = Subset(dataset, indices)
        targets = set(np.unique([subset[i][1].item() for i in range(len(subset))]))
        expected_targets = set(self._class_groupings[chunk_id])
        if targets != expected_targets:
            raise ValueError(
                f"Chunk {chunk_id} does not contain classes "
                f"{sorted(list(expected_targets - targets))}."
            )
        return subset


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
        super().__init__(data_module, len(transforms), chunk_id, seed)
        self._transforms = transforms

    def setup(self) -> None:
        self._data_module.setup()
        self._split_and_assign_train_and_val_data()
        self._train_data = _TransformedDataset(
            self._train_data, transform=self._transforms[self._chunk_id]
        )
        if self._val_data:
            self._val_data = _TransformedDataset(
                self._val_data, transform=self._transforms[self._chunk_id]
            )
        self._test_data = []
        for i in range(self._num_tasks):
            self._test_data.append(
                _TransformedDataset(self._data_module.test_data(), transform=self._transforms[i])
            )


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
            transform = Lambda(lambda x, p=permutation: x.flatten()[p].view(x.size()))
            transforms.append(transform)
        super().__init__(data_module, transforms, chunk_id, seed)


class IIDScenario(Scenario):
    """A scenario splitting datasets into random equally-sized chunks.

    Args:
        data_module: The source RenateDataModule for the the user data.
        num_tasks: The total number of expected tasks for experimentation.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: Seed used to fix random number generation.
    """

    def setup(self) -> None:
        """Make assignments: val/train/test splits."""
        self._data_module.setup()
        proportions = [1 / self._num_tasks for _ in range(self._num_tasks)]
        self._train_data = randomly_split_data(
            self._data_module.train_data(), proportions, self._seed
        )[self._chunk_id]
        val_data = self._data_module.val_data()
        if val_data:
            self._val_data = randomly_split_data(val_data, proportions, self._seed)[self._chunk_id]
        self._test_data = [self._data_module.test_data() for _ in range(self._num_tasks)]


class SoftSortingScenario(Scenario):
    """A scenario that _softly_ sorts a dataset by the value of a feature, then creates chunks.

    This extracts a feature value for each data point (see `feature_idx`) and computes sampling
    probabilities proportional to `feature_value ** exponent`. An ordering of the data points is
    then sampled from a corresponding multinomial distribution without replacement. This results in
    a "soft" sorting, where data points with large feature values tend to appear earlier. The
    ordered dataset is then split into `num_tasks` chunks.

    The larger `exponent`, the more pronounced the sorting effect. Very large values will result in
    (deterministic) sorting.

    This scenario assumes that `dataset[i]` returns a tuple `(x, y)` with a tensor `x` containing
    the features.

    Args:
        data_module: The source RenateDataModule for the the user data.
        num_tasks: The total number of expected tasks for experimentation.
        feature_idx: Index of the feature by which to sort. This index refers to the input features
            `x` of a single data point, i.e., no batch dimension. If the tensor `x` has more than
            one dimension, this indexes along the 0-dim while additional dimensions will be averaged
            out. Hence, for images, `feature_idx` refers to a color channel and we sort by mean
            color channel value.
        exponent: Exponent used when computing sampling probabilties. The larger `exponent`, the
            more pronounced the sorting effect.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        num_tasks: int,
        feature_idx: int,
        exponent: int,
        chunk_id: int,
        seed: int = defaults.SEED,
    ) -> None:
        super().__init__(data_module, num_tasks, chunk_id, seed)
        self._feature_idx = feature_idx
        self._exponent = exponent

    def _get_features(self, dataset: Dataset) -> torch.Tensor:
        features = torch.empty(len(dataset))
        for i in range(len(dataset)):
            features[i] = dataset[i][self._feature_idx].mean()
        return features

    def _compute_sampling_probs(self, features: torch.Tensor) -> torch.Tensor:
        # Use log-sum-exp trick.
        tmp = self._exponent * torch.log(features)
        log_probs = tmp - torch.logsumexp(tmp, dim=0)
        return torch.exp(log_probs)

    def _split(self, dataset: Dataset) -> Dataset:
        features = self._get_features(dataset)
        sampling_probs = self._compute_sampling_probs(features)
        rng = get_generator(self._seed)
        idx_ordered = torch.multinomial(
            sampling_probs, num_samples=len(dataset), replacement=False, generator=rng
        )
        split = torch.tensor_split(idx_ordered, self._num_tasks)
        return [Subset(dataset, idx) for idx in split]

    def setup(self) -> None:
        self._data_module.setup()
        train_data = self._data_module.train_data()
        self._train_data = self._split(train_data)[self._chunk_id]
        val_data = self._data_module.val_data()
        if val_data:
            self._val_data = self._split(val_data)[self._chunk_id]
        test_data = self._data_module.test_data()
        self._test_data = self._split(test_data)
