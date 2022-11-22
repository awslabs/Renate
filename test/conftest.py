# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
from typing import Callable, Dict

import pytest
import torch
from pytorch_lightning.loggers import TensorBoardLogger

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
from renate.models.renate_module import RenateModule
from renate.updaters.experimental.repeated_distill import RepeatedDistillationLearner
from renate.updaters.experimental.er import ExperienceReplayLearner
from renate.updaters.experimental.gdumb import GDumbLearner
from renate.updaters.experimental.joint import JointLearner
from renate.updaters.experimental.offline_er import OfflineExperienceReplayLearner
from renate.updaters.learner import Learner, ReplayLearner
from renate.updaters.model_updater import SimpleModelUpdater

pytest_plugins = ["helpers_namespace"]


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests e.g. testing data modules.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="Need --runslow option to run.")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


LEARNER_KWARGS = {
    ExperienceReplayLearner: {
        "memory_size": 30,
        "memory_batch_size": 20,
        "optimizer": "SGD",
        "learning_rate": 2.5,
        "momentum": 1.3,
        "weight_decay": 0.5,
        "batch_size": 50,
        "seed": 1,
    },
    Learner: {
        "optimizer": "SGD",
        "learning_rate": 1.23,
        "momentum": 0.9,
        "weight_decay": 0.005,
        "batch_size": 10,
        "seed": 42,
    },
    GDumbLearner: {
        "optimizer": "SGD",
        "learning_rate": 1.23,
        "momentum": 0.9,
        "weight_decay": 0.005,
        "batch_size": 10,
        "seed": 42,
        "memory_size": 30,
    },
    JointLearner: {
        "optimizer": "SGD",
        "learning_rate": 1.11,
        "momentum": 0.4,
        "weight_decay": 0.001,
        "batch_size": 10,
        "seed": 3,
    },
    RepeatedDistillationLearner: {
        "optimizer": "SGD",
        "learning_rate": 1.23,
        "momentum": 0.9,
        "weight_decay": 0.005,
        "batch_size": 10,
        "seed": 42,
        "memory_size": 30,
    },
    OfflineExperienceReplayLearner: {
        "memory_size": 30,
        "memory_batch_size": 20,
        "loss_weight_new_data": 0.5,
        "optimizer": "SGD",
        "learning_rate": 2.5,
        "momentum": 1.3,
        "weight_decay": 0.5,
        "batch_size": 50,
        "seed": 1,
    },
}
LEARNER_HYPERPARAMETER_UPDATES = {
    ExperienceReplayLearner: {
        "optimizer": "Adam",
        "learning_rate": 3.0,
        "momentum": 0.5,
        "weight_decay": 0.01,
        "batch_size": 128,
    },
    Learner: {
        "optimizer": "Adam",
        "learning_rate": 3.0,
        "weight_decay": 0.01,
        "batch_size": 128,
    },
    GDumbLearner: {
        "optimizer": "Adam",
        "learning_rate": 2.0,
        "momentum": 0.5,
        "weight_decay": 0.03,
        "batch_size": 128,
        "memory_size": 50,
    },
    JointLearner: {
        "optimizer": "Adam",
        "learning_rate": 2.0,
        "weight_decay": 0.01,
        "batch_size": 128,
    },
    RepeatedDistillationLearner: {
        "optimizer": "Adam",
        "learning_rate": 2.0,
        "weight_decay": 0.01,
        "batch_size": 128,
    },
    OfflineExperienceReplayLearner: {
        "optimizer": "Adam",
        "learning_rate": 3.0,
        "momentum": 0.5,
        "weight_decay": 0.01,
        "batch_size": 128,
    },
}
LEARNERS = list(LEARNER_KWARGS)
LEARNERS_USING_SIMPLE_UPDATER = [
    ExperienceReplayLearner,
    Learner,
    GDumbLearner,
    JointLearner,
    OfflineExperienceReplayLearner,
]

SAMPLE_CLASSIFICATION_RESULTS = {
    "accuracy": [
        [0.9362000226974487, 0.6093000173568726, 0.3325999975204468],
        [0.8284000158309937, 0.9506999850273132, 0.3382999897003174],
        [0.4377000033855438, 0.48260000348091125, 0.9438999891281128],
    ],
    "accuracy_init": [[0.2, 0.1, 0.09]],
}

TEST_WORKING_DIRECTORY = "./test_renate_working_dir/"
TEST_LOGGER = TensorBoardLogger
TEST_LOGGER_KWARGS = {"save_dir": TEST_WORKING_DIRECTORY, "version": 1, "name": "lightning_logs"}


@pytest.helpers.register
def get_renate_module_mlp(num_inputs, num_outputs, num_hidden_layers, hidden_size) -> RenateModule:
    return MultiLayerPerceptron(num_inputs, num_outputs, num_hidden_layers, hidden_size)


@pytest.helpers.register
def get_renate_module_resnet(sub_class="resnet18cifar", **kwargs) -> RenateModule:
    if sub_class == "resnet18cifar":
        return ResNet18CIFAR(**kwargs)
    elif sub_class == "resnet34cifar":
        return ResNet34CIFAR(**kwargs)
    elif sub_class == "resnet50cifar":
        return ResNet50CIFAR(**kwargs)
    elif sub_class == "resnet18":
        return ResNet18(**kwargs)
    elif sub_class == "resnet34":
        return ResNet34(**kwargs)
    elif sub_class == "resnet50":
        return ResNet50(**kwargs)
    else:
        raise ValueError("Invalid ResNet called.")


@pytest.helpers.register
def get_renate_module_vision_transformer(
    sub_class="visiontransformerb16", **kwargs
) -> RenateModule:
    if sub_class == "visiontransformercifar":
        return VisionTransformerCIFAR(**kwargs)
    elif sub_class == "visiontransformerb16":
        return VisionTransformerB16(**kwargs)
    elif sub_class == "visiontransformerb32":
        return VisionTransformerB32(**kwargs)
    elif sub_class == "visiontransformerl16":
        return VisionTransformerL16(**kwargs)
    elif sub_class == "visiontransformerl32":
        return VisionTransformerL32(**kwargs)
    elif sub_class == "visiontransformerh14":
        return VisionTransformerH14(**kwargs)
    else:
        raise ValueError("Invalid Vision Transformer called.")


@pytest.helpers.register
def get_renate_vision_module(model, sub_class="resnet18cifar", **kwargs):
    if model == "resnet":
        return get_renate_module_resnet(sub_class, **kwargs)
    elif model == "visiontransformer":
        return get_renate_module_vision_transformer(sub_class, **kwargs)
    else:
        raise ValueError("Invalid vision model called.")


@pytest.helpers.register
def get_renate_module_mlp_and_data(
    num_inputs,
    num_outputs,
    num_hidden_layers,
    hidden_size,
    train_num_samples,
    test_num_samples,
    val_num_samples=0,
):
    model = get_renate_module_mlp(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.rand(train_num_samples, num_inputs),
        torch.randint(num_outputs, (train_num_samples,)),
    )
    test_data = torch.rand(test_num_samples, num_inputs)

    if val_num_samples > 0:
        val_dataset = torch.utils.data.TensorDataset(
            torch.rand(val_num_samples, num_inputs),
            torch.randint(num_outputs, (val_num_samples,)),
        )
        return model, train_dataset, val_dataset

    return model, train_dataset, test_data


@pytest.helpers.register
def get_renate_vision_module_and_data(
    input_size,
    num_outputs,
    train_num_samples,
    test_num_samples,
    model="resnet",
    sub_class="reduced18",
    **kwargs,
):
    model = get_renate_vision_module(model, sub_class, **kwargs)
    train_dataset = torch.utils.data.TensorDataset(
        torch.rand(train_num_samples, *input_size),
        torch.randint(num_outputs, (train_num_samples,)),
    )
    test_data = torch.rand(test_num_samples, *input_size)
    return model, train_dataset, test_data


@pytest.helpers.register
def get_simple_updater(
    model,
    current_state_folder=None,
    next_state_folder=None,
    learner_class=ExperienceReplayLearner,
    learner_kwargs={"memory_size": 10},
    max_epochs=5,
    train_transform=None,
    train_target_transform=None,
    test_transform=None,
    test_target_transform=None,
    buffer_transform=None,
    buffer_target_transform=None,
    early_stopping_enabled=False,
    metric=None,
):
    transforms_kwargs = {
        "train_transform": train_transform,
        "train_target_transform": train_target_transform,
        "test_transform": test_transform,
        "test_target_transform": test_target_transform,
    }
    if issubclass(learner_class, ReplayLearner):
        transforms_kwargs["buffer_transform"] = buffer_transform
        transforms_kwargs["buffer_target_transform"] = buffer_target_transform
    return SimpleModelUpdater(
        model=model,
        learner_class=learner_class,
        learner_kwargs=learner_kwargs,
        current_state_folder=current_state_folder,
        next_state_folder=next_state_folder,
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=TEST_LOGGER(**TEST_LOGGER_KWARGS),
        early_stopping_enabled=early_stopping_enabled,
        metric=metric,
        **transforms_kwargs,
    )


@pytest.helpers.register
def check_learner_transforms(learner: Learner, expected_transforms: Dict[str, Callable]):
    """Checks if the learner transforms match to expected ones.

    Args:
        learner: The learner which transforms will be checked.
        expected_transforms: Dictionairy mapping from transform name to transform. These are the expected transforms
            for the learner.
    """
    assert learner._train_transform is expected_transforms.get(
        "train_transform"
    ) and learner._train_target_transform is expected_transforms.get("train_target_transform")
    if isinstance(learner, ReplayLearner):
        assert learner._memory_buffer._transform is expected_transforms.get(
            "buffer_transform"
        ) and learner._memory_buffer._target_transform is expected_transforms.get(
            "buffer_target_transform"
        )


def pytest_sessionstart(session):
    if not os.path.exists(TEST_WORKING_DIRECTORY):
        os.mkdir(TEST_WORKING_DIRECTORY)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(TEST_WORKING_DIRECTORY)
