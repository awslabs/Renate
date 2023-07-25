# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from renate import defaults
from renate.updaters.experimental.er import (
    CLSExperienceReplayLearner,
    DarkExperienceReplayLearner,
    ExperienceReplayLearner,
    PooledOutputDistillationExperienceReplayLearner,
    SuperExperienceReplayLearner,
)


def get_model_and_dataset():
    model = pytest.helpers.get_renate_module_mlp(
        num_outputs=10, num_inputs=10, hidden_size=32, num_hidden_layers=3
    )
    dataset = torch.utils.data.TensorDataset(
        torch.rand((100, 10)),
        torch.randint(10, (100,)),
    )
    return model, dataset


@pytest.mark.parametrize(
    "batch_size,memory_size,memory_batch_size",
    [[10, 10, 10], [20, 10, 10], [10, 100, 10], [10, 30, 1], [100, 10, 3]],
)
def test_er_overall_memory_size_after_update(batch_size, memory_size, memory_batch_size):
    model, dataset = get_model_and_dataset()
    learner_kwargs = {
        "memory_size": memory_size,
        "memory_batch_size": memory_batch_size,
        "batch_size": batch_size,
    }
    model_updater = pytest.helpers.get_simple_updater(
        model=model,
        partial_optimizer=pytest.helpers.get_partial_optimizer(),
        learner_class=ExperienceReplayLearner,
        learner_kwargs=learner_kwargs,
        max_epochs=1,
    )
    model_updater.update(train_dataset=dataset, task_id=defaults.TASK_ID)
    memory, _ = next(iter(model_updater._learner._memory_loader))
    x, y = memory
    assert x.shape[0] == memory_batch_size and y.shape[0] == memory_batch_size
    assert len(model_updater._learner._memory_buffer) == memory_size


def test_er_validation_buffer(tmpdir):
    model, dataset_train, _ = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=32,
        num_hidden_layers=3,
        train_num_samples=10,
        test_num_samples=5,
    )
    datasets_val = []
    state_folder = None
    next_state_folder = defaults.input_state_folder(tmpdir)
    for i in range(3):
        dataset_val = torch.utils.data.TensorDataset(
            torch.rand((100, 10)),
            torch.randint(10, (100,)),
        )
        model_updater = pytest.helpers.get_simple_updater(
            model, input_state_folder=state_folder, output_state_folder=next_state_folder
        )
        model_updater.update(
            train_dataset=dataset_train, val_dataset=dataset_val, task_id=defaults.TASK_ID
        )
        datasets_val.append(dataset_val)
        state_folder = next_state_folder

    model_updater = pytest.helpers.get_simple_updater(
        model, input_state_folder=state_folder, output_state_folder=next_state_folder
    )
    for i in range(3):
        for j in range(100):
            assert torch.allclose(
                datasets_val[i][j][0],
                model_updater._learner._val_memory_buffer[i * 100 + j][0][0],
                rtol=1e-3,
            )
