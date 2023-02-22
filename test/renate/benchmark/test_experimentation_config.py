# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from torchvision.transforms import Compose, Normalize

from renate.benchmark import experiment_config
from renate.benchmark.datasets.vision_datasets import CLEARDataModule, TorchVisionDataModule
from renate.benchmark.experiment_config import (
    data_module_fn,
    get_data_module,
    get_scenario,
    model_fn,
    models,
    train_transform,
)
from renate.benchmark.scenarios import (
    BenchmarkScenario,
    ClassIncrementalScenario,
    IIDScenario,
    ImageRotationScenario,
    PermutationScenario,
)


@pytest.mark.parametrize(
    "model_name,expected_model_class",
    [(model_name, model_class) for model_name, model_class in zip(models.keys(), models.values())],
)
def test_model_fn(model_name, expected_model_class):
    model = model_fn(
        model_state_url=None,
        model_fn_model_name=model_name,
        model_fn_num_inputs=1 if model_name == "MultiLayerPerceptron" else None,
        model_fn_num_outputs=1 if model_name == "MultiLayerPerceptron" else None,
        model_fn_num_hidden_layers=1 if model_name == "MultiLayerPerceptron" else None,
        model_fn_hidden_size="1" if model_name == "MultiLayerPerceptron" else None,
    )
    assert isinstance(model, expected_model_class)


def test_model_fn_fails_for_unknown_model():
    unknown_model_name = "UNKNOWN_MODEL_NAME"
    with pytest.raises(ValueError, match=f"Unknown model `{unknown_model_name}`"):
        model_fn(model_fn_model_name=unknown_model_name)


@pytest.mark.parametrize(
    "dataset_name,data_module_class",
    (("CIFAR10", TorchVisionDataModule), ("CLEAR10", CLEARDataModule)),
)
def test_get_data_module(tmpdir, dataset_name, data_module_class):
    data_module = get_data_module(data_path=tmpdir, dataset_name=dataset_name, val_size=0.5, seed=0)
    assert isinstance(data_module, data_module_class)


def test_get_data_module_fails_for_unknown_dataset(tmpdir):
    unknown_dataset_name = "UNKNOWN_DATASET_NAME"
    with pytest.raises(ValueError, match=f"Unknown dataset `{unknown_dataset_name}`"):
        get_data_module(data_path=tmpdir, dataset_name=unknown_dataset_name, val_size=0.5, seed=0)


def test_get_scenario_fails_for_unknown_scenario(tmpdir):
    data_module = get_data_module(data_path=tmpdir, dataset_name="MNIST", val_size=0.5, seed=0)
    unknown_scenario_name = "UNKNOWN_SCENARIO_NAME"
    with pytest.raises(ValueError, match=f"Unknown scenario `{unknown_scenario_name}`"):
        get_scenario(
            scenario_name=unknown_scenario_name, data_module=data_module, chunk_id=0, seed=0
        )


@pytest.mark.parametrize(
    "scenario_name,dataset_name,scenario_kwargs,expected_scenario_class,expected_num_tasks",
    (
        (
            "ClassIncrementalScenario",
            "CIFAR10",
            {"data_module_fn_class_groupings": "[[0,1],[2,3,4],[5,6]]"},
            ClassIncrementalScenario,
            3,
        ),
        (
            "IIDScenario",
            "MNIST",
            {"data_module_fn_num_tasks": 3},
            IIDScenario,
            3,
        ),
        (
            "ImageRotationScenario",
            "MNIST",
            {"data_module_fn_degrees": "[0,90,180]"},
            ImageRotationScenario,
            3,
        ),
        ("BenchmarkScenario", "CLEAR10", {"data_module_fn_num_tasks": "5"}, BenchmarkScenario, 5),
        (
            "PermutationScenario",
            "MNIST",
            {"data_module_fn_num_tasks": "3", "data_module_fn_input_dim": "(1,28,28)"},
            PermutationScenario,
            3,
        ),
    ),
    ids=[
        "class_incremental_image",
        "iid",
        "rotation",
        "benchmark",
        "permutation",
    ],
)
@pytest.mark.parametrize("val_size", (0, 0.5), ids=["no_val", "val"])
def test_data_module_fn(
    tmpdir,
    scenario_name,
    dataset_name,
    scenario_kwargs,
    expected_scenario_class,
    expected_num_tasks,
    val_size,
):
    scenario = data_module_fn(
        data_path=tmpdir,
        chunk_id=0,
        seed=0,
        data_module_fn_scenario_name=scenario_name,
        data_module_fn_dataset_name=dataset_name,
        data_module_fn_val_size=val_size,
        **scenario_kwargs,
    )
    assert isinstance(scenario, expected_scenario_class)
    if expected_scenario_class == ClassIncrementalScenario:
        assert (
            str(scenario._class_groupings).replace(" ", "")
            == scenario_kwargs["data_module_fn_class_groupings"]
        )
    assert scenario._num_tasks == expected_num_tasks


@pytest.mark.parametrize(
    "dataset_name,use_transforms",
    (("MNIST", False), ("FashionMNIST", False), ("CIFAR10", True), ("CIFAR100", True)),
)
def test_transforms(dataset_name, use_transforms):
    train_preprocessing = train_transform(dataset_name)
    test_preprocessing = experiment_config.test_transform(dataset_name)
    if use_transforms:
        assert isinstance(train_preprocessing, Compose)
        assert isinstance(test_preprocessing, Normalize)
    else:
        assert train_preprocessing is None
        assert test_preprocessing is None


def test_transforms_fails_for_unknown_dataset():
    unknown_dataset_set = "UNKNOWN_DATASET_NAME"
    for transform_function in [train_transform, experiment_config.test_transform]:
        with pytest.raises(ValueError, match=f"Unknown dataset `{unknown_dataset_set}`"):
            transform_function(unknown_dataset_set)
