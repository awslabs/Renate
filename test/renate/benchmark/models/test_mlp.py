# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.defaults import TASK_ID


def test_renate_mlp_init():
    pytest.helpers.get_renate_module_mlp(
        num_inputs=10, num_outputs=10, hidden_size=32, num_hidden_layers=3
    )


def test_renate_mlp_fwd():
    mlp = pytest.helpers.get_renate_module_mlp(
        num_inputs=10, num_outputs=10, hidden_size=32, num_hidden_layers=3
    )
    x = torch.rand(5, 10)
    y_hat = mlp(x)

    assert y_hat.shape[0] == 5
    assert y_hat.shape[1] == 10


@pytest.mark.parametrize("num_hidden_layers,expected_num_params", [[1, 4], [2, 6], [3, 8]])
def test_renate_mlp_get_params(num_hidden_layers, expected_num_params):
    mlp = pytest.helpers.get_renate_module_mlp(
        num_inputs=10, num_outputs=10, hidden_size=5, num_hidden_layers=num_hidden_layers
    )
    mlp.add_task_params(TASK_ID)

    first_task_params = mlp.get_params(TASK_ID)

    mlp.add_task_params("second_task")

    second_task_params = mlp.get_params("second_task")
    assert len(first_task_params) == len(second_task_params)
    # +2 for the output layer (weight and bias)
    # +4 for the 2 output layers  (weight and bias)
    assert len(first_task_params) == expected_num_params + 2
    assert len(second_task_params) == expected_num_params + 2
    assert len(list(mlp.parameters())) == expected_num_params + 4
    for i in range(len(first_task_params) - 2):
        # -2 because the last two parameters are weight and bias of a task specific linear layer
        assert torch.equal(first_task_params[i], second_task_params[i])

    for i in range(len(first_task_params) - 2, len(first_task_params)):
        # -2 because the last two parameters are weight and bias of a task specific linear layer
        assert not torch.equal(first_task_params[i], second_task_params[i])
