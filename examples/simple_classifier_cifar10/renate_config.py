# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torchvision import transforms

import renate.defaults as defaults
from renate.benchmark.datasets.vision_datasets import TorchVisionDataModule
from renate.benchmark.models import ResNet18CIFAR
from renate.benchmark.scenarios import ClassIncrementalScenario, Scenario
from renate.models import RenateModule


def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    """Returns a model instance."""
    if model_state_url is None:
        model = ResNet18CIFAR()
    else:
        state_dict = torch.load(str(model_state_url))
        model = ResNet18CIFAR.from_state_dict(state_dict)
    return model


def data_module_fn(
    data_path: Union[Path, str], chunk_id: int, seed: int = defaults.SEED
) -> Scenario:
    """Returns a class-incremental scenario instance.

    The transformations passed to prepare the input data are required to convert the data to
    PyTorch tensors.
    """
    data_module = TorchVisionDataModule(
        str(data_path),
        dataset_name="CIFAR10",
        val_size=0.2,
        seed=seed,
    )
    class_incremental_scenario = ClassIncrementalScenario(
        data_module=data_module,
        class_groupings=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        chunk_id=chunk_id,
    )
    return class_incremental_scenario


def train_transform() -> Callable:
    """Returns a transform function to be used in the training."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
        ]
    )


def test_transform() -> Callable:
    """Returns a transform function to be used for validation or testing."""
    return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))


def buffer_transform() -> Callable:
    """Returns a transform function to be used in the Memory Buffer."""
    return train_transform()
