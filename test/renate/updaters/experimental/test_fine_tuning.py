# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
import torch

import renate.defaults as defaults
from renate.updaters.experimental.fine_tuning import FineTuningModelUpdater


def get_model_and_dataset():
    model = pytest.helpers.get_renate_module_mlp(
        num_outputs=10, num_inputs=10, hidden_size=32, num_hidden_layers=3
    )
    dataset = torch.utils.data.TensorDataset(
        torch.rand((100, 10)),
        torch.randint(10, (100,)),
    )
    return model, dataset


def test_fine_tuning_updater():
    """This test checks that the memory buffer is updated correctly."""
    init_model, dataset = get_model_and_dataset()

    model = copy.deepcopy(init_model)

    model_updater = FineTuningModelUpdater(model, max_epochs=1)
    model_updater.update(train_dataset=dataset, task_id=defaults.TASK_ID)

    for p1, p2 in zip(init_model.parameters(), model.parameters()):
        assert p1.data.ne(p2.data).sum().item() != 0
