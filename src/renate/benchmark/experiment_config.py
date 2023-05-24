# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple, Union

import torch
import wild_time_data
from torchvision.transforms import transforms
from transformers import AutoTokenizer

from renate.benchmark.datasets.nlp_datasets import HuggingFaceTextDataModule
from renate.benchmark.datasets.vision_datasets import CLEARDataModule, TorchVisionDataModule
from renate.benchmark.datasets.wild_time_data import WildTimeDataModule
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
from renate.benchmark.models.transformer import HuggingFaceSequenceClassificationTransformer
from renate.benchmark.scenarios import (
    BenchmarkScenario,
    ClassIncrementalScenario,
    FeatureSortingScenario,
    HueShiftScenario,
    IIDScenario,
    ImageRotationScenario,
    PermutationScenario,
    Scenario,
    WildTimeScenario,
)
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule
from renate.models.prediction_strategies import ICaRLClassificationStrategy

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
    "HuggingFaceTransformer": HuggingFaceSequenceClassificationTransformer,
}


def model_fn(
    model_state_url: Optional[str] = None,
    updater: Optional[str] = None,
    model_name: Optional[str] = None,
    num_inputs: Optional[int] = None,
    num_outputs: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
    hidden_size: Optional[Tuple[int]] = None,
    dataset_name: Optional[str] = None,
    pretrained_model_name: Optional[str] = None,
) -> RenateModule:
    """Returns a model instance."""
    if model_name not in models:
        raise ValueError(f"Unknown model `{model_name}`")
    model_class = models[model_name]
    model_kwargs = {}
    if updater == "Avalanche-iCaRL":
        model_kwargs["prediction_strategy"] = ICaRLClassificationStrategy()
    if model_name == "MultiLayerPerceptron":
        model_kwargs.update(
            {
                "num_inputs": num_inputs,
                "num_hidden_layers": num_hidden_layers,
                "hidden_size": hidden_size,
            }
        )
    elif model_name.startswith("ResNet") and dataset_name in ["FashionMNIST", "MNIST", "yearbook"]:
        model_kwargs["gray_scale"] = True
    elif model_name == "HuggingFaceTransformer":
        if updater == "Avalanche-iCaRL":
            raise ValueError("Transformers do not support iCaRL.")
        model_kwargs["pretrained_model_name"] = pretrained_model_name
    if num_outputs is not None:
        model_kwargs["num_outputs"] = num_outputs
    if model_state_url is None:
        model = model_class(**model_kwargs)
    else:
        state_dict = torch.load(model_state_url)
        model = model_class.from_state_dict(state_dict)
    return model


def get_data_module(
    data_path: str,
    src_bucket: Optional[str],
    src_object_name: Optional[str],
    dataset_name: str,
    val_size: float,
    seed: int,
    pretrained_model_name: Optional[str],
    input_column: Optional[str],
    target_column: Optional[str],
) -> RenateDataModule:
    if dataset_name in TorchVisionDataModule.dataset_dict:
        return TorchVisionDataModule(
            data_path, dataset_name=dataset_name, val_size=val_size, seed=seed
        )
    if dataset_name in ["CLEAR10", "CLEAR100"]:
        return CLEARDataModule(data_path, dataset_name=dataset_name, val_size=val_size, seed=seed)
    if dataset_name in wild_time_data.list_datasets():
        data_module_kwargs = {
            "data_path": data_path,
            "src_bucket": src_bucket,
            "src_object_name": src_object_name,
            "dataset_name": dataset_name,
            "val_size": val_size,
            "seed": seed,
        }
        if pretrained_model_name is not None:
            data_module_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(pretrained_model_name)
        return WildTimeDataModule(**data_module_kwargs)

    if dataset_name.startswith("hfd-"):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        return HuggingFaceTextDataModule(
            data_path=data_path,
            dataset_name=dataset_name[4:],
            input_column=input_column,
            target_column=target_column,
            tokenizer=tokenizer,
            val_size=val_size,
            seed=seed,
        )
    raise ValueError(f"Unknown dataset `{dataset_name}`.")


def get_scenario(
    scenario_name: str,
    data_module: RenateDataModule,
    chunk_id: int,
    seed: int,
    num_tasks: Optional[int] = None,
    class_groupings: Optional[Tuple[Tuple[int]]] = None,
    degrees: Optional[List[int]] = None,
    input_dim: Optional[Union[List[int], Tuple[int], int]] = None,
    feature_idx: Optional[int] = None,
    randomness: Optional[float] = None,
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
        randomness: Used for all `_SortingScenario`. Randomness strength in [0, 1].

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
    if scenario_name == "FeatureSortingScenario":
        return FeatureSortingScenario(
            data_module=data_module,
            num_tasks=num_tasks,
            feature_idx=feature_idx,
            randomness=randomness,
            chunk_id=chunk_id,
            seed=seed,
        )
    if scenario_name == "HueShiftScenario":
        return HueShiftScenario(
            data_module=data_module,
            num_tasks=num_tasks,
            randomness=randomness,
            chunk_id=chunk_id,
            seed=seed,
        )
    if scenario_name == "WildTimeScenario":
        return WildTimeScenario(
            data_module=data_module, num_tasks=num_tasks, chunk_id=chunk_id, seed=seed
        )
    raise ValueError(f"Unknown scenario `{scenario_name}`.")


def data_module_fn(
    data_path: str,
    chunk_id: int,
    seed: int,
    scenario_name: str,
    dataset_name: str,
    val_size: float = 0.0,
    num_tasks: Optional[int] = None,
    class_groupings: Optional[Tuple[Tuple[int]]] = None,
    degrees: Optional[Tuple[int]] = None,
    input_dim: Optional[Tuple[int]] = None,
    feature_idx: Optional[int] = None,
    randomness: Optional[float] = None,
    src_bucket: Optional[str] = None,
    src_object_name: Optional[str] = None,
    pretrained_model_name: Optional[str] = None,
    input_column: Optional[str] = None,
    target_column: Optional[str] = None,
):
    data_module = get_data_module(
        data_path=data_path,
        src_bucket=src_bucket,
        src_object_name=src_object_name,
        dataset_name=dataset_name,
        val_size=val_size,
        seed=seed,
        pretrained_model_name=pretrained_model_name,
        input_column=input_column,
        target_column=target_column,
    )
    if dataset_name in wild_time_data.list_datasets() and num_tasks is None:
        num_tasks = len(wild_time_data.available_time_steps(dataset_name))
    return get_scenario(
        scenario_name=scenario_name,
        data_module=data_module,
        chunk_id=chunk_id,
        seed=seed,
        num_tasks=num_tasks,
        class_groupings=class_groupings,
        degrees=degrees,
        input_dim=input_dim,
        feature_idx=feature_idx,
        randomness=randomness,
    )


def _get_normalize_transform(dataset_name):
    if dataset_name in TorchVisionDataModule.dataset_stats:
        return transforms.Normalize(
            TorchVisionDataModule.dataset_stats[dataset_name]["mean"],
            TorchVisionDataModule.dataset_stats[dataset_name]["std"],
        )


def train_transform(dataset_name: str) -> Optional[transforms.Compose]:
    """Returns a transform function to be used in the training."""
    if dataset_name in [
        "MNIST",
        "FashionMNIST",
    ] + wild_time_data.list_datasets() or dataset_name.startswith("hfd-"):
        return None
    if dataset_name in ["CIFAR10", "CIFAR100"]:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                _get_normalize_transform(dataset_name),
            ]
        )
    raise ValueError(f"Unknown dataset `{dataset_name}`.")


def test_transform(dataset_name: str) -> Optional[transforms.Normalize]:
    """Returns a transform function to be used for validation or testing."""
    if dataset_name in [
        "MNIST",
        "FashionMNIST",
    ] + wild_time_data.list_datasets() or dataset_name.startswith("hfd-"):
        return None
    if dataset_name in ["CIFAR10", "CIFAR100"]:
        return _get_normalize_transform(dataset_name)
    raise ValueError(f"Unknown dataset `{dataset_name}`.")
