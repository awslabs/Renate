# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Callable, Dict, Generator, Optional, Tuple

import torch
import torchvision
from torch.nn import Parameter
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torchmetrics import Accuracy
from torchvision.transforms import transforms

from renate.data.data_module import RenateDataModule
from renate.models import RenateModule


class MyMNISTMLP(RenateModule):
    def __init__(self, num_hidden: int) -> None:
        # Model hyperparameters need to registered via RenateModule's
        # constructor, see documentation. Otherwise, this is a standard torch model.
        super().__init__(constructor_arguments={"num_hidden": num_hidden})
        self._fc1 = torch.nn.Linear(28 * 28, num_hidden)
        self._fc2 = torch.nn.Linear(num_hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._fc1(x)
        x = torch.nn.functional.relu(x)
        return self._fc2(x)


def model_fn(model_state_url: Optional[str] = None) -> RenateModule:
    if model_state_url is None:
        # If no model state is given, we create the model from scratch with initial model
        # hyperparameters.
        model = MyMNISTMLP(num_hidden=100)
    else:
        # If a model state is passed, we reload the model using PyTorch's load_state_dict.
        # In this case, model hyperparameters are restored from the saved state.
        state_dict = torch.load(model_state_url)
        model = MyMNISTMLP.from_state_dict(state_dict)
    return model


class MyMNISTDataModule(RenateDataModule):
    def __init__(self, data_path: str, val_size: float, seed: int = 42) -> None:
        super().__init__(data_path, val_size=val_size, seed=seed)

    def prepare_data(self) -> None:
        # This is only to download the data. We separate downloading from the remaining set-up to
        # streamline data loading when using multiple training jobs during HPO.
        torchvision.datasets.MNIST(self._data_path, download=True)

    def setup(self) -> None:
        # This sets up train/val/test datasets, assuming data has already been downloaded.
        train_data = torchvision.datasets.MNIST(
            self._data_path,
            train=True,
            transform=transforms.ToTensor(),
            target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)),
        )
        self._train_data, self._val_data = self._split_train_val_data(train_data)
        self._test_data = torchvision.datasets.MNIST(
            self._data_path,
            train=False,
            transform=transforms.ToTensor(),
            target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)),
        )


def data_module_fn(data_path: str, seed: int) -> RenateDataModule:
    return MyMNISTDataModule(val_size=0.2, seed=seed)


def train_transform() -> Callable:
    return torchvision.transforms.Compose(
        [torchvision.transforms.RandomCrop((28, 28), padding=4), torch.nn.Flatten()]
    )


def test_transform() -> Callable:
    return torch.nn.Flatten()


def buffer_transform() -> Callable:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop((28, 28), padding=4),
            torchvision.transforms.RandomRotation(degrees=15),
            torch.nn.Flatten(),
        ]
    )


def metrics_fn() -> Dict:
    return {"accuracy": Accuracy(task="multiclass", num_classes=10)}


def loss_fn() -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss(reduction="none")


def optimizer_fn() -> Callable[[Generator[Parameter]], Optimizer]:
    return partial(AdamW, lr=0.01, weight_decay=0.0)


def lr_scheduler_fn() -> Tuple[Callable[[Optimizer], _LRScheduler], str]:
    return partial(StepLR, step_size=10, gamma=0.1), "epoch"
