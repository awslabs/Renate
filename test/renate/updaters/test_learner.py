# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Dict, Tuple, Type

import pytest
import torch
from torchvision.transforms import ToTensor

from conftest import (
    LEARNERS,
    LEARNER_HYPERPARAMETER_UPDATES,
    LEARNER_KWARGS,
    check_learner_transforms,
)
from renate.models import RenateModule
from renate.updaters.learner import Learner, ReplayLearner


def get_model_and_learner_and_learner_kwargs(
    learner_class: Type[Learner],
) -> Tuple[RenateModule, Learner, Dict[str, Any]]:
    learner_kwargs = LEARNER_KWARGS[learner_class]
    model = pytest.helpers.get_renate_module_mlp(
        num_inputs=1, num_outputs=1, hidden_size=1, num_hidden_layers=1
    )
    learner = learner_class(model=model, **learner_kwargs)
    return model, learner, learner_kwargs


def check_learner_variables(learner: Learner, expected_variable_values: Dict[str, Any]):
    for attribute_name, attribute_value in expected_variable_values.items():
        if attribute_name == "memory_size":
            continue
        assert getattr(learner, f"_{attribute_name}") == attribute_value


@pytest.mark.parametrize("learner_class", LEARNERS)
def test_save_and_load_learner(tmpdir, learner_class):
    model, learner, learner_kwargs = get_model_and_learner_and_learner_kwargs(learner_class)
    filename = os.path.join(tmpdir, "learner.pkl")
    torch.save(learner.state_dict(), filename)
    learner = learner_class.__new__(learner_class)
    learner.load_state_dict(model, torch.load(filename))
    check_learner_variables(learner, learner_kwargs)


@pytest.mark.parametrize("learner_class", LEARNERS)
def test_update_hyperparameters(learner_class):
    model, learner, learner_kwargs = get_model_and_learner_and_learner_kwargs(learner_class)
    check_learner_variables(learner, learner_kwargs)
    learner.update_hyperparameters({})
    check_learner_variables(learner, learner_kwargs)
    learner.update_hyperparameters(LEARNER_HYPERPARAMETER_UPDATES[learner_class])
    learner_kwargs = dict(learner_kwargs)
    learner_kwargs.update(LEARNER_HYPERPARAMETER_UPDATES[learner_class])
    check_learner_variables(learner, learner_kwargs)


@pytest.mark.parametrize("learner_class", LEARNERS)
def test_set_transforms(learner_class):
    """Tests if set_transforms function correctly sets transforms in Learner and MemoryBuffer."""
    model, learner, learner_kwargs = get_model_and_learner_and_learner_kwargs(learner_class)
    check_learner_transforms(learner, {})
    transforms_kwargs = {
        "train_transform": ToTensor(),
        "train_target_transform": ToTensor(),
        "test_transform": ToTensor(),
        "test_target_transform": ToTensor(),
    }
    if issubclass(learner_class, ReplayLearner):
        transforms_kwargs["buffer_transform"] = ToTensor()
        transforms_kwargs["buffer_target_transform"] = ToTensor()
    learner.set_transforms(**transforms_kwargs)
    check_learner_transforms(learner, transforms_kwargs)
