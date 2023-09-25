# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
from typing import Callable, Dict, Literal

import pytest
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from renate import defaults
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
from renate.models.renate_module import RenateModule
from renate.updaters.avalanche.learner import (
    AvalancheEWCLearner,
    AvalancheICaRLLearner,
    AvalancheLwFLearner,
    AvalancheReplayLearner,
)
from renate.updaters.avalanche.model_updater import AvalancheModelUpdater
from renate.updaters.experimental.er import ExperienceReplayLearner
from renate.updaters.experimental.gdumb import GDumbLearner
from renate.updaters.experimental.joint import JointLearner
from renate.updaters.experimental.l2p import LearningToPromptLearner, LearningToPromptReplayLearner
from renate.updaters.experimental.offline_er import OfflineExperienceReplayLearner
from renate.updaters.experimental.repeated_distill import RepeatedDistillationLearner
from renate.updaters.learner import Learner, ReplayLearner
from renate.updaters.model_updater import SingleTrainingLoopUpdater
from renate.utils.optimizer import create_partial_optimizer

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
        "batch_memory_frac": 0.4,
        "batch_size": 50,
        "seed": 1,
    },
    Learner: {"batch_size": 10, "seed": 42},
    LearningToPromptLearner: {"batch_size": 10, "seed": 42, "prompt_sim_loss_weight": 1},
    LearningToPromptReplayLearner: {
        "batch_size": 10,
        "seed": 42,
        "prompt_sim_loss_weight": 1,
        "loss_weight_new_data": 0.5,
        "memory_size": 30,
        "batch_memory_frac": 0.3,
    },
    GDumbLearner: {
        "batch_size": 10,
        "seed": 42,
        "memory_size": 30,
    },
    JointLearner: {"batch_size": 10, "seed": 3},
    RepeatedDistillationLearner: {"batch_size": 10, "seed": 42, "memory_size": 30},
    OfflineExperienceReplayLearner: {
        "memory_size": 30,
        "batch_memory_frac": 0.4,
        "loss_weight_new_data": 0.5,
        "batch_size": 50,
        "seed": 1,
    },
}
AVALANCHE_LEARNER_KWARGS = {
    AvalancheReplayLearner: {
        "memory_size": 30,
        "batch_memory_frac": 0.4,
        "batch_size": 50,
        "seed": 1,
    },
    AvalancheEWCLearner: {
        "ewc_lambda": 0.1,
        "batch_size": 50,
        "seed": 1,
    },
    AvalancheLwFLearner: {
        "alpha": 0.1,
        "temperature": 2,
        "batch_size": 50,
        "seed": 1,
    },
    AvalancheICaRLLearner: {
        "memory_size": 30,
        "batch_size": 50,
        "seed": 1,
    },
}
AVALANCHE_LEARNER_HYPERPARAMETER_UPDATES = {
    AvalancheEWCLearner: {"ewc_lambda": 0.3},
    AvalancheLwFLearner: {"alpha": 0.2, "temperature": 3},
    AvalancheICaRLLearner: {},
    AvalancheReplayLearner: {},
}
LEARNERS = list(LEARNER_KWARGS)
AVALANCHE_LEARNERS = list(AVALANCHE_LEARNER_KWARGS)
LEARNERS_USING_SIMPLE_UPDATER = [
    ExperienceReplayLearner,
    Learner,
    GDumbLearner,
    JointLearner,
    OfflineExperienceReplayLearner,
]
L2P_LEARNERS = [
    LearningToPromptLearner,
    LearningToPromptReplayLearner,
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
def get_renate_module_mlp(
    num_inputs, num_outputs, num_hidden_layers, hidden_size, add_icarl_class_means=False
) -> RenateModule:
    return MultiLayerPerceptron(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_size,
        add_icarl_class_means=add_icarl_class_means,
    )


@pytest.helpers.register
def get_loss_fn(reduction="none") -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss(reduction=reduction)


@pytest.helpers.register
def get_renate_module_resnet(sub_class="resnet18cifar", **kwargs) -> RenateModule:
    kwargs["add_icarl_class_means"] = False
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
    kwargs["add_icarl_class_means"] = False
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
    add_icarl_class_means=False,
):
    model = get_renate_module_mlp(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        add_icarl_class_means=add_icarl_class_means,
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
    partial_optimizer=None,
    input_state_folder=None,
    output_state_folder=None,
    learner_class=ExperienceReplayLearner,
    learner_kwargs=None,
    max_epochs=5,
    train_transform=None,
    train_target_transform=None,
    test_transform=None,
    test_target_transform=None,
    buffer_transform=None,
    buffer_target_transform=None,
    early_stopping_enabled=False,
    metric=None,
    deterministic_trainer=False,
):
    if learner_kwargs is None:
        learner_kwargs = {"memory_size": 10}
    transforms_kwargs = {
        "train_transform": train_transform,
        "train_target_transform": train_target_transform,
        "test_transform": test_transform,
        "test_target_transform": test_target_transform,
    }
    if issubclass(learner_class, ReplayLearner):
        transforms_kwargs["buffer_transform"] = buffer_transform
        transforms_kwargs["buffer_target_transform"] = buffer_target_transform
    return SingleTrainingLoopUpdater(
        model=model,
        loss_fn=get_loss_fn(),
        optimizer=partial_optimizer or get_partial_optimizer(),
        learner_class=learner_class,
        learner_kwargs=learner_kwargs,
        input_state_folder=input_state_folder,
        output_state_folder=output_state_folder,
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=TEST_LOGGER(**TEST_LOGGER_KWARGS),
        early_stopping_enabled=early_stopping_enabled,
        metric=metric,
        deterministic_trainer=deterministic_trainer,
        **transforms_kwargs,
    )


@pytest.helpers.register
def get_avalanche_updater(
    model,
    input_state_folder=None,
    output_state_folder=None,
    learner_class=AvalancheReplayLearner,
    learner_kwargs=None,
    max_epochs=5,
    train_transform=None,
    train_target_transform=None,
    test_transform=None,
    test_target_transform=None,
    early_stopping_enabled=False,
    metric=None,
):
    if learner_kwargs is None:
        learner_kwargs = {"memory_size": 10}
    transforms_kwargs = {
        "train_transform": train_transform,
        "train_target_transform": train_target_transform,
        "test_transform": test_transform,
        "test_target_transform": test_target_transform,
    }
    return AvalancheModelUpdater(
        model=model,
        loss_fn=get_loss_fn("mean"),
        optimizer=get_partial_optimizer(),
        learner_class=learner_class,
        learner_kwargs=learner_kwargs,
        input_state_folder=input_state_folder,
        output_state_folder=output_state_folder,
        max_epochs=max_epochs,
        accelerator="cpu",
        early_stopping_enabled=early_stopping_enabled,
        metric=metric,
        **transforms_kwargs,
    )


@pytest.helpers.register
def get_partial_optimizer(
    optimizer: Literal["Adam", "SGD"] = defaults.OPTIMIZER,
    lr: float = defaults.LEARNING_RATE,
    momentum: float = defaults.MOMENTUM,
    weight_decay: float = defaults.WEIGHT_DECAY,
):
    return create_partial_optimizer(
        optimizer=optimizer, lr=lr, momentum=momentum, weight_decay=weight_decay
    )


@pytest.helpers.register
def check_learner_transforms(learner: Learner, expected_transforms: Dict[str, Callable]):
    """Checks if the learner transforms match to expected ones.

    Args:
        learner: The learner which transforms will be checked.
        expected_transforms: Dictionairy mapping from transform name to transform. These are the
            expected transforms for the learner.
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
