# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from torch.nn import Linear
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms import Compose, Normalize, ToTensor

from renate.benchmark import experiment_config
from renate.benchmark.datasets.nlp_datasets import HuggingFaceTextDataModule, MultiTextDataModule
from renate.benchmark.datasets.vision_datasets import CLEARDataModule, TorchVisionDataModule
from renate.benchmark.experiment_config import (
    data_module_fn,
    get_data_module,
    get_scenario,
    loss_fn,
    lr_scheduler_fn,
    metrics_fn,
    model_fn,
    models,
    optimizer_fn,
    train_transform,
)
from renate.benchmark.scenarios import (
    ClassIncrementalScenario,
    DataIncrementalScenario,
    FeatureSortingScenario,
    HueShiftScenario,
    IIDScenario,
    ImageRotationScenario,
    PermutationScenario,
)
from renate.models.prediction_strategies import ICaRLClassificationStrategy


@pytest.mark.parametrize(
    "model_name,expected_model_class",
    [(model_name, model_class) for model_name, model_class in models.items()],
)
def test_model_fn(model_name, expected_model_class):
    model = model_fn(
        model_state_url=None,
        model_name=model_name,
        num_inputs=1 if model_name == "MultiLayerPerceptron" else None,
        num_outputs=2,
        num_hidden_layers=1 if model_name == "MultiLayerPerceptron" else None,
        hidden_size=1 if model_name == "MultiLayerPerceptron" else None,
        pretrained_model_name_or_path="distilbert-base-uncased"
        if model_name == "HuggingFaceTransformer"
        else None,
    )
    assert isinstance(model, expected_model_class)


@pytest.mark.parametrize(
    "dataset_name,expected_in_channels",
    (
        ("FashionMNIST", 1),
        ("MNIST", 1),
        ("yearbook", 1),
        ("CIFAR10", 3),
        ("CIFAR100", 3),
        ("fmow", 3),
    ),
)
def test_model_fn_automatic_input_channel_detection_resnet(dataset_name, expected_in_channels):
    """Tests if ResNet architectures input channels are correctly adapted to the dataset."""
    model = model_fn(
        num_outputs=10,
        model_state_url=None,
        model_name="ResNet18",
        dataset_name=dataset_name,
    )
    assert model.get_backbone().conv1.in_channels == expected_in_channels


def test_model_fn_fails_for_unknown_model():
    unknown_model_name = "UNKNOWN_MODEL_NAME"
    with pytest.raises(ValueError, match=f"Unknown model `{unknown_model_name}`"):
        model_fn(num_outputs=10, model_name=unknown_model_name)


@pytest.mark.parametrize(
    "dataset_name,data_module_class,pretrained_model_name_or_path,input_column,target_column",
    (
        ("CIFAR10", TorchVisionDataModule, None, None, None),
        ("CLEAR10", CLEARDataModule, None, None, None),
        (
            "hfd-rotten-tomatoes",
            HuggingFaceTextDataModule,
            "distilbert-base-uncased",
            "text",
            "label",
        ),
        ("MultiText", MultiTextDataModule, "distilbert-base-uncased", None, None),
    ),
)
def test_get_data_module(
    tmpdir,
    dataset_name,
    data_module_class,
    pretrained_model_name_or_path,
    input_column,
    target_column,
):
    data_module = get_data_module(
        data_path=tmpdir,
        dataset_name=dataset_name,
        val_size=0.5,
        seed=0,
        src_bucket=None,
        src_object_name=None,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        input_column=input_column,
        target_column=target_column,
    )
    assert isinstance(data_module, data_module_class)


def test_get_data_module_fails_for_unknown_dataset(tmpdir):
    unknown_dataset_name = "UNKNOWN_DATASET_NAME"
    with pytest.raises(ValueError, match=f"Unknown dataset `{unknown_dataset_name}`"):
        get_data_module(
            data_path=tmpdir,
            dataset_name=unknown_dataset_name,
            val_size=0.5,
            seed=0,
            src_bucket=None,
            src_object_name=None,
            pretrained_model_name_or_path=None,
            input_column=None,
            target_column=None,
        )


def test_get_scenario_fails_for_unknown_scenario(tmpdir):
    data_module = get_data_module(
        data_path=tmpdir,
        dataset_name="MNIST",
        val_size=0.5,
        seed=0,
        src_bucket=None,
        src_object_name=None,
        pretrained_model_name_or_path=None,
        input_column=None,
        target_column=None,
    )
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
            "hfd-trec",
            {
                "pretrained_model_name_or_path": "distilbert-base-uncased",
                "input_column": "text",
                "target_column": "coarse_label",
                "groupings": ((0, 1), (2, 3), (4, 5)),
            },
            ClassIncrementalScenario,
            3,
        ),
        (
            "IIDScenario",
            "MNIST",
            {"num_tasks": 3},
            IIDScenario,
            3,
        ),
        (
            "ImageRotationScenario",
            "MNIST",
            {"degrees": (0, 90, 180)},
            ImageRotationScenario,
            3,
        ),
        (
            "PermutationScenario",
            "MNIST",
            {"num_tasks": 3, "input_dim": (1, 28, 28)},
            PermutationScenario,
            3,
        ),
        (
            "FeatureSortingScenario",
            "MNIST",
            {
                "num_tasks": 5,
                "feature_idx": 0,
                "randomness": 0.3,
            },
            FeatureSortingScenario,
            5,
        ),
        (
            "HueShiftScenario",
            "CIFAR10",
            {"num_tasks": 3, "randomness": 0.5},
            HueShiftScenario,
            3,
        ),
        ("DataIncrementalScenario", "CLEAR10", {"num_tasks": 5}, DataIncrementalScenario, 5),
        (
            "DataIncrementalScenario",
            "arxiv",
            {"num_tasks": 3, "pretrained_model_name_or_path": "distilbert-base-uncased"},
            DataIncrementalScenario,
            3,
        ),
        (
            "DataIncrementalScenario",
            "fmow",
            {},
            DataIncrementalScenario,
            16,
        ),
        (
            "DataIncrementalScenario",
            "DomainNet",
            {"data_ids": ("clipart", "infograph")},
            DataIncrementalScenario,
            2,
        ),
        (
            "DataIncrementalScenario",
            "DomainNet",
            {"groupings": (("clipart", "infograph"), "painting")},
            DataIncrementalScenario,
            2,
        ),
    ),
    ids=[
        "class_incremental",
        "iid",
        "rotation",
        "permutation",
        "feature_sorting",
        "hue_shift",
        "data_incremental with CLEAR",
        "wild_time_text_with_tokenizer",
        "wild_time_image_all_tasks",
        "domainnet_by data_id",
        "domainnet by groupings",
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
        scenario_name=scenario_name,
        dataset_name=dataset_name,
        val_size=val_size,
        **scenario_kwargs,
    )
    assert isinstance(scenario, expected_scenario_class)
    if expected_scenario_class == ClassIncrementalScenario:
        assert scenario._class_groupings == scenario_kwargs["groupings"]
    elif expected_scenario_class == FeatureSortingScenario:
        assert scenario._feature_idx == scenario_kwargs["feature_idx"]
        assert scenario._randomness == scenario_kwargs["randomness"]
    elif expected_scenario_class == HueShiftScenario:
        assert scenario._randomness == scenario_kwargs["randomness"]
    elif expected_scenario_class == DataIncrementalScenario:
        if "pretrained_model_name_or_path" in scenario_kwargs:
            assert scenario._data_module._tokenizer is not None
        elif dataset_name not in ["CLEAR10", "CLEAR100", "DomainNet"]:
            assert scenario._data_module._tokenizer is None
    assert scenario._num_tasks == expected_num_tasks


@pytest.mark.parametrize(
    "dataset_name,expected_train_transform_class, expected_test_transform_class,model_name",
    (
        ("MNIST", type(None), type(None), "ResNet18CIFAR"),
        ("FashionMNIST", type(None), type(None), "ResNet18CIFAR"),
        ("CIFAR10", Compose, Normalize, "ResNet18CIFAR"),
        ("CIFAR100", Compose, Normalize, "ResNet18CIFAR"),
        ("CLEAR10", Compose, Compose, "ResNet18"),
        ("DomainNet", Compose, Compose, "VisionTransformerB16"),
        ("hfd-rotten_tomatoes", type(None), type(None), "HuggingFaceTransformer"),
        ("fmow", Compose, Compose, "ResNet18"),
        ("yearbook", ToTensor, ToTensor, "ResNet18CIFAR"),
        ("yearbook", Compose, Compose, "VisionTransformerB16"),
    ),
)
def test_transforms(
    dataset_name, expected_train_transform_class, expected_test_transform_class, model_name
):
    train_preprocessing = train_transform(dataset_name, model_name)
    test_preprocessing = experiment_config.test_transform(dataset_name, model_name)
    assert isinstance(train_preprocessing, expected_train_transform_class)
    assert isinstance(test_preprocessing, expected_test_transform_class)


def test_transforms_fails_for_unknown_dataset():
    unknown_dataset_set = "UNKNOWN_DATASET_NAME"
    for transform_function in [train_transform, experiment_config.test_transform]:
        with pytest.raises(ValueError, match=f"Unknown dataset `{unknown_dataset_set}`"):
            transform_function(unknown_dataset_set)


@pytest.mark.parametrize(
    "learning_rate_scheduler,expected_lr_class,expected_interval",
    (
        ("StepLR", StepLR, "epoch"),
        ("CosineAnnealingLR", CosineAnnealingLR, "step"),
        (None, None, "epoch"),
    ),
)
def test_lr_scheduler_fn(learning_rate_scheduler, expected_lr_class, expected_interval):
    scheduler, interval = lr_scheduler_fn(
        learning_rate_scheduler=learning_rate_scheduler,
        learning_rate_scheduler_interval=expected_interval,
    )
    assert interval == expected_interval
    if learning_rate_scheduler is None:
        assert scheduler is None
    else:
        scheduler = scheduler(SGD(Linear(1, 1).parameters(), lr=0.1))
        assert isinstance(scheduler, expected_lr_class)


def test_lr_scheduler_fn_fails_for_unknown_scheduler():
    unknown_lr_scheduler = "UNKNOWN_SCHEDULER_NAME"
    with pytest.raises(ValueError, match=f"Unknown scheduler `{unknown_lr_scheduler}`."):
        lr_scheduler_fn(unknown_lr_scheduler)


@pytest.mark.parametrize("model_name", [model_name for model_name in models])
@pytest.mark.parametrize("updater", ("ER", "Avalanche-iCaRL"))
def test_prediction_strategy_is_correctly_set(model_name, updater):
    """If iCaRL is used, the model prediction strategy must be changed."""
    model_kwargs = {
        "model_name": model_name,
        "updater": updater,
        "num_outputs": 2,
    }
    if model_name == "MultiLayerPerceptron":
        model_kwargs.update({"num_inputs": 10, "hidden_size": 10, "num_hidden_layers": 2})
    elif model_name == "HuggingFaceTransformer":
        model_kwargs["pretrained_model_name_or_path"] = "distilbert-base-uncased"
    if model_name == "HuggingFaceTransformer" and updater == "Avalanche-iCaRL":
        with pytest.raises(ValueError, match="Transformers do not support iCaRL."):
            model_fn(**model_kwargs)
    else:
        model = model_fn(**model_kwargs)
        if updater == "ER":
            assert not hasattr(model, "_prediction_strategy") or model._prediction_strategy is None
        else:
            assert isinstance(model._prediction_strategy, ICaRLClassificationStrategy)


def test_loss_fn_returns_correct_reduction_type():
    assert loss_fn("ER").reduction == "none"
    assert loss_fn("Avalanche-ER").reduction == "mean"


def test_metrics_fn_contains_accuracy():
    assert isinstance(metrics_fn(num_outputs=2)["accuracy"], MulticlassAccuracy)
    assert isinstance(metrics_fn(num_outputs=10)["accuracy"], MulticlassAccuracy)


def test_optimizer_fn():
    expected_learning_rate = 0.12
    partial_optimizer = optimizer_fn(
        optimizer="AdamW", learning_rate=expected_learning_rate, weight_decay=0.1
    )
    optimizer: AdamW = partial_optimizer(Linear(10, 10).parameters())
    assert isinstance(optimizer, AdamW)
    assert optimizer.defaults["lr"] == expected_learning_rate


def test_optimizer_fn_unknown_optimizer():
    assert optimizer_fn(optimizer="UNKNOWN_OPTIMIZER", learning_rate=0.1, weight_decay=1) is None
