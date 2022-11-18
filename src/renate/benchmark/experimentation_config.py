# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchmetrics
from syne_tune.config_space import loguniform

from renate.benchmark.datasets.nlp_datasets import TorchTextDataModule
from renate.benchmark.datasets.vision_datasets import CLEARDataModule, TorchVisionDataModule
from renate.benchmark.models.mlp import MultiLayerPerceptron
from renate.benchmark.models.resnet import (
    ResNet18,
    ResNet18CIFAR,
    ResNet34,
    ResNet34CIFAR,
    ResNet50,
    ResNet50CIFAR,
)
from renate.benchmark.models.vision_transformer import (
    VisionTransformerB16,
    VisionTransformerB32,
    VisionTransformerCIFAR,
    VisionTransformerH14,
    VisionTransformerL16,
    VisionTransformerL32,
)
from renate.benchmark.scenarios import BenchmarkScenario, ClassIncrementalScenario, Scenario
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule


def model_fn(
    model_state_url: Optional[Union[Path, str]] = None, model_fn_model_name: Optional[str] = None
) -> RenateModule:
    """Returns a model instance."""
    models = {
        "MultiLayerPerceptron": MultiLayerPerceptron,
        "ResNet18CIFAR": ResNet18CIFAR,
        "ResNet34CIFAR": ResNet34CIFAR,
        "ResNet50CIFAR": ResNet50CIFAR,
        "ResNet18": ResNet18,
        "ResNet34": ResNet34,
        "ResNet50": ResNet50,
        "VisionTransformerCIFAR": VisionTransformerCIFAR,
        "VisionTransformerB16": VisionTransformerB16,
        "VisionTransformerB32": VisionTransformerB32,
        "VisionTransformerB16": VisionTransformerL16,
        "VisionTransformerB32": VisionTransformerL32,
        "VisionTransformerH14": VisionTransformerH14,
    }
    assert model_fn_model_name in models, f"Unknown model: {model_fn_model_name}"
    model_class = models[model_fn_model_name]
    if model_state_url is None:
        model = model_class()
    else:
        state_dict = torch.load(str(model_state_url))
        model = model_class.from_state_dict(state_dict)
    return model


def get_data_module(
    data_path: str, dataset_name: str, val_size: float, seed: int
) -> RenateDataModule:
    if dataset_name in TorchVisionDataModule.dataset_dict:
        return TorchVisionDataModule(
            data_path, dataset_name=dataset_name, download=True, val_size=val_size, seed=seed
        )
    if dataset_name in ["CLEAR10", "CLEAR100"]:
        return CLEARDataModule(data_path, dataset_name=dataset_name, val_size=val_size, seed=seed)
    if dataset_name in TorchTextDataModule.dataset_dict:
        return TorchTextDataModule(
            data_path, dataset_name=dataset_name, val_size=val_size, seed=seed
        )
    raise ValueError(f"Unknown dataset `{dataset_name}`.")


def get_scenario(
    scenario_name: str,
    data_module: RenateDataModule,
    chunk_id: int,
    seed: int,
    num_tasks: Optional[int] = None,
    class_groupings: Optional[List[List[int]]] = None,
) -> Scenario:
    if scenario_name == "class_incremental":
        assert (
            class_groupings is not None
        ), "Provide `class_groupings` for the class-incremental scenario."
        return ClassIncrementalScenario(
            data_module=data_module,
            class_groupings=class_groupings,
            chunk_id=chunk_id,
        )
    if scenario_name == "benchmark":
        return BenchmarkScenario(
            data_module=data_module, num_tasks=num_tasks, chunk_id=chunk_id, seed=seed
        )
    # TODO:
    raise ValueError(f"Unknown scenario `{scenario_name}`.")


def data_module_fn(
    data_path: Union[Path, str],
    chunk_id: Optional[int] = None,
    seed: int = 0,
    data_module_fn_scenario_name: Optional[str] = None,
    data_module_fn_dataset_name: Optional[str] = None,
    data_module_fn_val_size: str = "0.0",
    data_module_fn_class_groupings: Optional[str] = None,
):
    data_module = get_data_module(
        data_path=str(data_path),
        dataset_name=data_module_fn_dataset_name,
        val_size=float(data_module_fn_val_size),
        seed=seed,
    )
    return get_scenario(
        scenario_name=data_module_fn_scenario_name,
        data_module=data_module,
        chunk_id=chunk_id,
        seed=seed,
        class_groupings=ast.literal_eval(data_module_fn_class_groupings),
    )


def config_space_fn():
    return {
        "updater": "ER",
        "learning_rate": loguniform(10e-5, 0.1),
        "loss_weight": 0.3,
        "early_stopping": False,
        "max_epochs": 1,  # TODO:
        "model_fn_model_name": "ResNet18CIFAR",
        "data_module_fn_scenario_name": "class_incremental",
        "data_module_fn_dataset_name": "CIFAR10",
        "data_module_fn_val_size": 0.995,
        "data_module_fn_class_groupings": "[[0,1]]",
    }


def metrics_fn() -> Dict[str, torchmetrics.Metric]:
    return {"acc": torchmetrics.Accuracy()}
