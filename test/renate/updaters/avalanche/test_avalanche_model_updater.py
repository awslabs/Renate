# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import torch
from avalanche.training.plugins import ReplayPlugin

from conftest import AVALANCHE_LEARNER_KWARGS
from renate import defaults
from renate.updaters.avalanche.learner import AvalancheICaRLLearner, plugin_by_class
from renate.updaters.avalanche.model_updater import ExperienceReplayAvalancheModelUpdater


def get_model_and_dataset(dataset_size: int = 100):  # TODO: remove
    model, dataset, _ = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=32,
        num_hidden_layers=3,
        train_num_samples=dataset_size,
        test_num_samples=5,
    )
    return model, dataset


@pytest.mark.parametrize("provide_folder", [True, False])
def test_avalanche_model_updater(tmpdir, provide_folder):
    model, train_dataset, test_data = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=32,
        num_hidden_layers=3,
        train_num_samples=10,
        test_num_samples=5,
    )
    model_updater = pytest.helpers.get_avalanche_updater(
        model, output_state_folder=defaults.output_state_folder(tmpdir) if provide_folder else None
    )
    y_hat_before_train = model(test_data, task_id=defaults.TASK_ID)
    model_updater.update(train_dataset, task_id=defaults.TASK_ID)
    y_hat_after_train = model(test_data, task_id=defaults.TASK_ID)
    assert y_hat_before_train.shape[0] == y_hat_after_train.shape[0]
    assert not torch.allclose(y_hat_before_train, y_hat_after_train)


@pytest.mark.parametrize("learner_class", list(AVALANCHE_LEARNER_KWARGS.keys()))
def test_continuation_of_training_with_avalanche_model_updater(tmpdir, learner_class):
    if learner_class == AvalancheICaRLLearner:
        return
    model, train_dataset, _ = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=32,
        num_hidden_layers=1,
        train_num_samples=10,
        test_num_samples=5,
        add_icarl_class_means=learner_class == AvalancheICaRLLearner,
    )
    state_url = defaults.input_state_folder(tmpdir)
    model_updater = pytest.helpers.get_avalanche_updater(
        model,
        learner_class=learner_class,
        learner_kwargs=AVALANCHE_LEARNER_KWARGS[learner_class],
        output_state_folder=state_url,
        max_epochs=2,
    )
    model = model_updater.update(train_dataset, task_id=defaults.TASK_ID)
    model_updater = pytest.helpers.get_avalanche_updater(
        model,
        learner_class=learner_class,
        learner_kwargs=AVALANCHE_LEARNER_KWARGS[learner_class],
        input_state_folder=state_url,
        max_epochs=2,
    )
    model_updater.update(train_dataset, task_id=defaults.TASK_ID)


@pytest.mark.parametrize(
    "batch_size,memory_size,batch_memory_frac",
    [[20, 10, 0.5], [30, 10, 0.34], [20, 100, 0.5], [10, 30, 0.1], [100, 10, 0.03]],
)
def test_experience_replay_buffer_size(tmpdir, batch_size, memory_size, batch_memory_frac):
    expected_memory_batch_size = int(batch_memory_frac * batch_size)
    expected_batch_size = batch_size - expected_memory_batch_size
    dataset_size = 100
    model, dataset = get_model_and_dataset(dataset_size)
    learner_kwargs = {
        "memory_size": memory_size,
        "batch_memory_frac": batch_memory_frac,
        "batch_size": batch_size,
    }
    model_updater = ExperienceReplayAvalancheModelUpdater(
        output_state_folder=str(Path(tmpdir) / "0"),
        model=model,
        loss_fn=pytest.helpers.get_loss_fn("mean"),
        optimizer=pytest.helpers.get_partial_optimizer(),
        **learner_kwargs,
        max_epochs=1,
        accelerator="cpu",
    )
    model_updater.update(train_dataset=dataset)
    replay_plugin = plugin_by_class(ReplayPlugin, model_updater._learner.plugins)
    assert replay_plugin.batch_size == expected_batch_size
    assert replay_plugin.mem_size == memory_size
    assert replay_plugin.batch_size_mem == expected_memory_batch_size
    assert len(replay_plugin.storage_policy.buffer) == min(
        memory_size, dataset_size, len(replay_plugin.storage_policy.buffer)
    )
    assert len(model_updater._learner.dataloader.data) == dataset_size
    for i in range(6):
        del model_updater
        del replay_plugin
        _, dataset = get_model_and_dataset(dataset_size)
        model_updater = ExperienceReplayAvalancheModelUpdater(
            input_state_folder=str(Path(tmpdir) / str(i)),
            output_state_folder=str(Path(tmpdir) / str(i + 1)),
            model=model,
            loss_fn=pytest.helpers.get_loss_fn("mean"),
            optimizer=pytest.helpers.get_partial_optimizer(),
            **learner_kwargs,
            max_epochs=1,
        )
        replay_plugin = plugin_by_class(ReplayPlugin, model_updater._learner.plugins)

        assert replay_plugin.batch_size == expected_batch_size
        assert replay_plugin.mem_size == memory_size
        assert replay_plugin.batch_size_mem == expected_memory_batch_size
        model_updater.update(train_dataset=dataset)
        assert len(model_updater._learner.dataloader.data) == dataset_size
        assert len(model_updater._learner.dataloader.memory) == min(
            memory_size, (i + 1) * dataset_size
        )
    assert len(replay_plugin.storage_policy.buffer) == min(memory_size, 2 * dataset_size)
