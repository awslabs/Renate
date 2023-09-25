# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional, Tuple

import torch
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
    groupings: Tuple[Tuple[int]] = ((0, 1), (2, 3, 4)),
    optional_tuple: Optional[Tuple[float]] = None,
    optional_float: Optional[float] = None,
    list_param: list = [1, 2],
    bool_param: bool = False,
) -> RenateDataModule:
    return DummyTorchVisionDataModule(transform=None, val_size=val_size, seed=seed)


def loss_fn(updater: Optional[str] = None) -> torch.nn.Module:
    if updater.startswith("Avalanche-"):
        return torch.nn.CrossEntropyLoss()
    return torch.nn.CrossEntropyLoss(reduction="none")


def metrics_fn(num_outputs: int = 10) -> Dict:
    return {"accuracy": Accuracy(task="multiclass", num_classes=num_outputs)}
