# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from renate import defaults
from renate.updaters.experimental.repeated_distill import (
    RepeatedDistillationModelUpdater,
    double_distillation_loss,
)


def test_dmc_runs_end_to_end():
    data = []
    classes_available = [0, 1, 2]

    mlp = pytest.helpers.get_renate_module_mlp(
        num_inputs=10, num_outputs=len(classes_available), hidden_size=256, num_hidden_layers=2
    )

    for i in classes_available:
        ds = torch.utils.data.TensorDataset(
            torch.mul(torch.ones(200, 10), i),
            torch.randint(low=i, high=i + 1, size=(200,)),
        )

        data.append(ds)

    val = ConcatDataset(data)
    updater = RepeatedDistillationModelUpdater(
        loss_fn=pytest.helpers.get_loss_fn(),
        optimizer=pytest.helpers.get_partial_optimizer(),
        model=mlp,
        memory_size=300,
        batch_size=20,
        max_epochs=5,
        accelerator="cpu",
    )

    for i in range(len(data)):
        updater.update(data[i], val_dataset=val, task_id=defaults.TASK_ID)


@pytest.mark.parametrize("dataset_size", [50, 200])
@pytest.mark.parametrize("memory_size", [10, 100])
def test_dmc_memory_size_after_update(memory_size, dataset_size):
    model = pytest.helpers.get_renate_module_mlp(
        num_inputs=10, num_outputs=3, hidden_size=20, num_hidden_layers=1
    )
    model_updater = RepeatedDistillationModelUpdater(
        loss_fn=pytest.helpers.get_loss_fn(),
        optimizer=pytest.helpers.get_partial_optimizer(),
        model=model,
        memory_size=memory_size,
        max_epochs=1,
        accelerator="cpu",
    )
    datasets = [
        TensorDataset(
            torch.randn(dataset_size, 10), torch.randint(low=0, high=3, size=(dataset_size,))
        )
        for _ in range(3)
    ]
    assert len(model_updater._learner._memory_buffer) == 0
    for i, dataset in enumerate(datasets):
        model_updater.update(dataset, task_id=defaults.TASK_ID)
        assert len(model_updater._learner._memory_buffer) == min(
            memory_size, (i + 1) * dataset_size
        )


@pytest.mark.parametrize("provide_folder", [True, False])
def test_dmc_model_updater(tmpdir, provide_folder):
    model, train_dataset, test_data = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=32,
        num_hidden_layers=3,
        train_num_samples=10,
        test_num_samples=5,
    )
    model_updater = RepeatedDistillationModelUpdater(
        model,
        loss_fn=pytest.helpers.get_loss_fn(),
        optimizer=pytest.helpers.get_partial_optimizer(),
        memory_size=50,
        max_epochs=1,
        output_state_folder=defaults.output_state_folder(tmpdir) if provide_folder else None,
        accelerator="cpu",
    )
    y_hat_before_train = model(test_data, task_id=defaults.TASK_ID)
    model_updater.update(train_dataset, task_id=defaults.TASK_ID)
    y_hat_after_train = model(test_data, task_id=defaults.TASK_ID)
    assert y_hat_before_train.shape[0] == y_hat_after_train.shape[0]
    assert not torch.allclose(y_hat_before_train, y_hat_after_train)


def test_continuation_of_training_with_dmc_model_updater(tmpdir):
    model, train_dataset, _ = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=32,
        num_hidden_layers=3,
        train_num_samples=10,
        test_num_samples=5,
    )
    state_url = defaults.input_state_folder(tmpdir)
    model_updater = RepeatedDistillationModelUpdater(
        model,
        loss_fn=pytest.helpers.get_loss_fn(),
        optimizer=pytest.helpers.get_partial_optimizer(),
        memory_size=50,
        max_epochs=1,
        output_state_folder=state_url,
        accelerator="cpu",
    )
    model = model_updater.update(train_dataset, task_id=defaults.TASK_ID)
    model_updater = RepeatedDistillationModelUpdater(
        model,
        loss_fn=pytest.helpers.get_loss_fn(),
        optimizer=pytest.helpers.get_partial_optimizer(),
        memory_size=50,
        max_epochs=1,
        input_state_folder=state_url,
        accelerator="cpu",
    )
    model_updater.update(train_dataset, task_id=defaults.TASK_ID)


def test_dmc_loss():
    a = torch.ones(10, 3)
    b = torch.zeros(10, 3)

    loss = double_distillation_loss(a, b)
    expected_loss = torch.mul(
        torch.ones(
            10,
        ),
        0.5,
    )

    assert torch.allclose(loss, expected_loss)
