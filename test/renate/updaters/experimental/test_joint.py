# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
import torch

import renate.defaults as defaults
from renate.updaters.experimental.joint import JointLearner


def get_model_and_dataset():
    model = pytest.helpers.get_renate_module_mlp(
        num_outputs=10, num_inputs=10, hidden_size=32, num_hidden_layers=3
    )
    dataset = torch.utils.data.TensorDataset(
        torch.rand((100, 10)),
        torch.randint(10, (100,)),
    )
    return model, dataset


def test_joint_learner_memory_append():
    """This test checks that the memory buffer is updated correctly."""
    model, dataset = get_model_and_dataset()
    dataset_len = len(dataset)
    model_updater = pytest.helpers.get_simple_updater(
        model=model,
        learner_class=JointLearner,
        learner_kwargs={},
        max_epochs=1,
    )
    model_updater.update(train_dataset=dataset, task_id=defaults.TASK_ID)
    assert len(model_updater._learner._memory_buffer) == dataset_len
    model_updater.update(train_dataset=dataset, task_id=defaults.TASK_ID)
    assert len(model_updater._learner._memory_buffer) == 2 * dataset_len


def test_joint_learner_model_reset():
    """This test checks that the model is reinitialized correctly before an update."""
    model, dataset = get_model_and_dataset()
    model_updater = pytest.helpers.get_simple_updater(
        model=model,
        learner_class=JointLearner,
        learner_kwargs={"learning_rate": 0.0},
        max_epochs=1,
    )
    model = model_updater.update(train_dataset=dataset, task_id=defaults.TASK_ID)
    model_copy = copy.deepcopy(model)
    model = model_updater.update(train_dataset=dataset, task_id=defaults.TASK_ID)
    for (name, param), (name_copy, param_copy) in zip(
        model.named_parameters(), model_copy.named_parameters()
    ):
        assert name == name_copy
        assert not torch.allclose(param, param_copy)
