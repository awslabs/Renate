# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.plugins import ReplayPlugin

from datasets import DummyTorchVisionDataModule
from renate.updaters.avalanche.learner import plugin_by_class
from renate.updaters.avalanche.model_updater import ExperienceReplayAvalancheModelUpdater


def get_model_and_dataset(dataset_size: int = 100):  # TODO: remove
    model = pytest.helpers.get_renate_module_mlp(
        num_outputs=10, num_inputs=10, hidden_size=32, num_hidden_layers=3
    )
    dataset = torch.utils.data.TensorDataset(
        torch.rand((dataset_size, 10)),
        torch.randint(10, (dataset_size,)),
    )
    return model, dataset


@pytest.mark.parametrize(
    "batch_size,memory_size,memory_batch_size",
    [[10, 10, 10], [20, 10, 10], [10, 100, 10], [10, 30, 1], [100, 10, 3]],
)
@pytest.mark.parametrize("dataset_size", (50, 100))
def test_er(tmpdir, batch_size, memory_size, memory_batch_size, dataset_size):
    model, dataset = get_model_and_dataset(dataset_size)
    learner_kwargs = {
        "memory_size": memory_size,
        "memory_batch_size": memory_batch_size,
        "batch_size": batch_size,
    }
    model_updater = ExperienceReplayAvalancheModelUpdater(
        next_state_folder=tmpdir,
        model=model,
        **learner_kwargs,
        max_epochs=1,
    )
    print("a1", model_updater._learner.clock.train_exp_counter)
    model_updater.update(train_dataset=dataset)
    print("a2", model_updater._learner.clock.train_exp_counter)
    replay_plugin = plugin_by_class(ReplayPlugin, model_updater._learner.plugins)
    assert replay_plugin.batch_size == batch_size
    assert replay_plugin.mem_size == memory_size
    assert replay_plugin.batch_size_mem == memory_batch_size
    assert len(replay_plugin.storage_policy.buffer) == min(
        memory_size, dataset_size, len(replay_plugin.storage_policy.buffer)
    )
    # assert len(model_updater._learner.dataloader.data) == dataset_size
    # assert len(model_updater._learner.dataloader.memory) == 0
    for i in range(6):
        del model_updater
        del replay_plugin
        _, dataset = get_model_and_dataset(dataset_size)
        model_updater = ExperienceReplayAvalancheModelUpdater(
            current_state_folder=tmpdir,
            next_state_folder=tmpdir,
            model=model,
            **learner_kwargs,
            max_epochs=1,
        )
        replay_plugin = plugin_by_class(ReplayPlugin, model_updater._learner.plugins)

        assert replay_plugin.batch_size == batch_size
        assert replay_plugin.mem_size == memory_size
        assert replay_plugin.batch_size_mem == memory_batch_size
        model_updater.update(train_dataset=dataset)
        assert len(model_updater._learner.dataloader.data) == dataset_size
        assert len(model_updater._learner.dataloader.memory) == min(
            memory_size, (i + 1) * dataset_size
        )
    # assert len(replay_plugin.storage_policy.buffer) == min(memory_size, 2 * dataset_size)
