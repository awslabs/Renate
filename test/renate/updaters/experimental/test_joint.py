# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import renate.defaults as defaults
from renate.updaters.experimental.joint import JointLearner


def test_joint_learner_memory_append():
    """This test checks that the memory buffer is updated correctly."""
    model, dataset, _ = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        num_hidden_layers=3,
        hidden_size=32,
        train_num_samples=100,
        test_num_samples=100,
    )
    dataset_len = len(dataset)
    model_updater = pytest.helpers.get_simple_updater(
        model=model,
        partial_optimizer=pytest.helpers.get_partial_optimizer(),
        learner_class=JointLearner,
        learner_kwargs={},
        max_epochs=1,
    )
    model_updater.update(train_dataset=dataset, task_id=defaults.TASK_ID)
    assert len(model_updater._learner._memory_buffer) == dataset_len
    model_updater.update(train_dataset=dataset, task_id=defaults.TASK_ID)
    assert len(model_updater._learner._memory_buffer) == 2 * dataset_len
