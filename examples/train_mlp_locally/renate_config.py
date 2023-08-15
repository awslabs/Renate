# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Dict, Optional

import torch
from torchmetrics import Accuracy
from torchvision.transforms import transforms

from renate import defaults
from renate.benchmark.datasets.vision_datasets import TorchVisionDataModule
from renate.benchmark.models.mlp import MultiLayerPerceptron
from renate.benchmark.scenarios import ClassIncrementalScenario, Scenario
from renate.models import RenateModule


def data_module_fn(data_path: str, chunk_id: int, seed: int = defaults.SEED) -> Scenario:
    """Returns a class-incremental scenario instance.

    The transformations passed to prepare the input data are required to convert the data to
    PyTorch tensors.
    """
    data_module = TorchVisionDataModule(
        data_path,
        dataset_name="MNIST",
        val_size=0.1,
        seed=seed,
    )

    class_incremental_scenario = ClassIncrementalScenario(
        data_module=data_module,
        groupings=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
        chunk_id=chunk_id,
    )
    return class_incremental_scenario


def model_fn(model_state_url: Optional[str] = None) -> RenateModule:
    """Returns a model instance."""
    if model_state_url is None:
        model = MultiLayerPerceptron(
            num_inputs=784, num_outputs=10, num_hidden_layers=2, hidden_size=128
        )
    else:
        state_dict = torch.load(model_state_url)
        model = MultiLayerPerceptron.from_state_dict(state_dict)
    return model


def train_transform() -> Callable:
    """Returns a transform function to be used in the training."""
    return transforms.Lambda(lambda x: torch.flatten(x))


def loss_fn() -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss(reduction="none")


def metrics_fn() -> Dict:
    return {"accuracy": Accuracy(task="multiclass", num_classes=10)}
