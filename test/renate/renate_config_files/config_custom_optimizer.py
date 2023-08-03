# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import Parameter
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torchmetrics import Accuracy

from dummy_datasets import DummyTorchVisionDataModule
from renate.benchmark.models.mlp import MultiLayerPerceptron
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule


def model_fn(model_state_url: Optional[str] = None) -> RenateModule:
    if model_state_url is None:
        return MultiLayerPerceptron(5 * 5, 10, 0, 64)
    state_dict = torch.load(model_state_url)
    return MultiLayerPerceptron.from_state_dict(state_dict)


def data_module_fn(
    data_path: str,
    val_size: float = 0.0,
    seed: int = 0,
) -> RenateDataModule:
    return DummyTorchVisionDataModule(transform=None, val_size=val_size, seed=seed)


def loss_fn(updater: Optional[str] = None) -> torch.nn.Module:
    if updater.startswith("Avalanche-"):
        return torch.nn.CrossEntropyLoss()
    return torch.nn.CrossEntropyLoss(reduction="none")


def optimizer_fn() -> Callable[[List[Parameter]], Optimizer]:
    return partial(SGD, lr=0.01)


def lr_scheduler_fn() -> Tuple[Callable[[Optimizer], _LRScheduler], str]:
    return partial(StepLR, step_size=10, gamma=0.1), "epoch"


def metrics_fn() -> Dict:
    return {"accuracy": Accuracy(task="multiclass", num_classes=10)}
