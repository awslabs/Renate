# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torchvision.transforms import transforms

from renate.benchmark.datasets.vision_datasets import CLEARDataModule, TorchVisionDataModule
from renate.benchmark.models import (
    MultiLayerPerceptron,
    ResNet18,
    ResNet18CIFAR,
    ResNet34,
    ResNet34CIFAR,
    ResNet50,
    ResNet50CIFAR,
    VisionTransformerB16,
    VisionTransformerB32,
    VisionTransformerCIFAR,
    VisionTransformerH14,
    VisionTransformerL16,
    VisionTransformerL32,
)
from renate.benchmark.scenarios import (
    BenchmarkScenario,
    ClassIncrementalScenario,
    IIDScenario,
    ImageRotationScenario,
    PermutationScenario,
    Scenario,
    SoftSortingScenario,
)
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule

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
    "VisionTransformerL16": VisionTransformerL16,
    "VisionTransformerL32": VisionTransformerL32,
    "VisionTransformerH14": VisionTransformerH14,
}


def model_fn(
    model_state_url: Optional[Union[Path, str]] = None,
    model_fn_model_name: Optional[str] = None,
    model_fn_num_inputs: Optional[str] = None,
    model_fn_num_outputs: Optional[str] = None,
    model_fn_num_hidden_layers: Optional[str] = None,
    model_fn_hidden_size: Optional[str] = None,
) -> RenateModule:
    """Returns a model instance."""
    if model_fn_model_name not in models:
        raise ValueError(f"Unknown model `{model_fn_model_name}`")
    model_class = models[model_fn_model_name]
    model_kwargs = {}
    if model_fn_model_name == "MultiLayerPerceptron":
        model_kwargs = {
            "num_inputs": int(model_fn_num_inputs),
            "num_hidden_layers": int(model_fn_num_hidden_layers),
            "hidden_size": ast.literal_eval(model_fn_hidden_size),
        }
    if model_fn_num_outputs is not None:
        model_kwargs["num_outputs"] = int(model_fn_num_outputs)
    if model_state_url is None:
        model = model_class(**model_kwargs)
    else:
        state_dict = torch.load(str(model_state_url))
        model = model_class.from_state_dict(state_dict)
    return model


def get_data_module(
    data_path: str, dataset_name: str, val_size: float, seed: int
) -> RenateDataModule:
    if dataset_name in TorchVisionDataModule.dataset_dict:
        return TorchVisionDataModule(
            data_path, dataset_name=dataset_name, val_size=val_size, seed=seed
        )
    if dataset_name in ["CLEAR10", "CLEAR100"]:
        return CLEARDataModule(data_path, dataset_name=dataset_name, val_size=val_size, seed=seed)
    raise ValueError(f"Unknown dataset `{dataset_name}`.")


def get_scenario(
    scenario_name: str,
    data_module: RenateDataModule,
    chunk_id: int,
    seed: int,
    num_tasks: Optional[int] = None,
    class_groupings: Optional[List[List[int]]] = None,
    degrees: Optional[List[int]] = None,
    input_dim: Optional[Union[List[int], Tuple[int], int]] = None,
    feature_idx: Optional[int] = None,
    exponent: Optional[int] = None,
) -> Scenario:
    """Function to create scenario based on name and arguments.

    Args:
        scenario_name: Name to identify scenario.
        data_module: The base data module.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: A random seed to fix the created scenario.
        num_tasks: The total number of expected tasks for experimentation.
        class_groupings: Used for scenario `ClassIncrementalScenario`. Partitions classes into
            different chunks.
        degrees: Used for scenario `ImageRotationScenario`. Rotations applied for each chunk.
        input_dim: Used for scenario `PermutationScenario`. Input dimensionality.
        feature_idx: Used for scenario `SoftSortingScenario`. Index of feature to sort by.
        exponent: Used for secnario `SoftSortingScenario`. Exponent for soft sorting.

    Returns:
        An instance of the requested scenario.

    Raises:
        ValueError: If scenario name is unknown.
    """
    if scenario_name == "ClassIncrementalScenario":
        assert (
            class_groupings is not None
        ), "Provide `class_groupings` for the class-incremental scenario."
        return ClassIncrementalScenario(
            data_module=data_module,
            class_groupings=class_groupings,
            chunk_id=chunk_id,
        )
    if scenario_name == "BenchmarkScenario":
        return BenchmarkScenario(
            data_module=data_module, num_tasks=num_tasks, chunk_id=chunk_id, seed=seed
        )
    if scenario_name == "IIDScenario":
        return IIDScenario(
            data_module=data_module, num_tasks=num_tasks, chunk_id=chunk_id, seed=seed
        )
    if scenario_name == "ImageRotationScenario":
        return ImageRotationScenario(
            data_module=data_module, degrees=degrees, chunk_id=chunk_id, seed=seed
        )
    if scenario_name == "PermutationScenario":
        return PermutationScenario(
            data_module=data_module,
            num_tasks=num_tasks,
            input_dim=input_dim,
            chunk_id=chunk_id,
            seed=seed,
        )
    if scenario_name == "SoftSortingScenario":
        return SoftSortingScenario(
            data_module=data_module,
            num_tasks=num_tasks,
            feature_idx=feature_idx,
            exponent=exponent,
            chunk_id=chunk_id,
            seed=seed,
        )
    raise ValueError(f"Unknown scenario `{scenario_name}`.")


def data_module_fn(
    data_path: Union[Path, str],
    chunk_id: int,
    seed: int,
    data_module_fn_scenario_name: str,
    data_module_fn_dataset_name: str,
    data_module_fn_val_size: str = "0.0",
    data_module_fn_num_tasks: Optional[str] = None,
    data_module_fn_class_groupings: Optional[str] = None,
    data_module_fn_degrees: Optional[str] = None,
    data_module_fn_input_dim: Optional[str] = None,
    data_module_fn_feature_idx: Optional[str] = None,
    data_module_fn_exponent: Optional[str] = None,
):
    data_module = get_data_module(
        data_path=str(data_path),
        dataset_name=data_module_fn_dataset_name,
        val_size=float(data_module_fn_val_size),
        seed=seed,
    )
    if data_module_fn_num_tasks is not None:
        data_module_fn_num_tasks = int(data_module_fn_num_tasks)
    if data_module_fn_class_groupings is not None:
        data_module_fn_class_groupings = ast.literal_eval(data_module_fn_class_groupings)
    if data_module_fn_degrees is not None:
        data_module_fn_degrees = ast.literal_eval(data_module_fn_degrees)
    if data_module_fn_input_dim is not None:
        data_module_fn_input_dim = ast.literal_eval(data_module_fn_input_dim)
    if data_module_fn_feature_idx is not None:
        data_module_fn_feature_idx = ast.literal_eval(data_module_fn_feature_idx)
    if data_module_fn_exponent is not None:
        data_module_fn_exponent = ast.literal_eval(data_module_fn_exponent)
    return get_scenario(
        scenario_name=data_module_fn_scenario_name,
        data_module=data_module,
        chunk_id=chunk_id,
        seed=seed,
        num_tasks=data_module_fn_num_tasks,
        class_groupings=data_module_fn_class_groupings,
        degrees=data_module_fn_degrees,
        input_dim=data_module_fn_input_dim,
        feature_idx=data_module_fn_feature_idx,
        exponent=data_module_fn_exponent,
    )


def _get_normalize_transform(dataset_name):
    if dataset_name in TorchVisionDataModule.dataset_stats:
        return transforms.Normalize(
            TorchVisionDataModule.dataset_stats[dataset_name]["mean"],
            TorchVisionDataModule.dataset_stats[dataset_name]["std"],
        )


def train_transform(transform_dataset_name: str) -> Optional[transforms.Compose]:
    """Returns a transform function to be used in the training."""
    if transform_dataset_name in ["MNIST", "FashionMNIST"]:
        return None
    elif transform_dataset_name in ["CIFAR10", "CIFAR100"]:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                _get_normalize_transform(transform_dataset_name),
            ]
        )
    raise ValueError(f"Unknown dataset `{transform_dataset_name}`.")


def test_transform(transform_dataset_name: str) -> Optional[transforms.Normalize]:
    """Returns a transform function to be used for validation or testing."""
    if transform_dataset_name in ["MNIST", "FashionMNIST"]:
        return None
    elif transform_dataset_name in ["CIFAR10", "CIFAR100"]:
        return _get_normalize_transform(transform_dataset_name)
    raise ValueError(f"Unknown dataset `{transform_dataset_name}`.")
