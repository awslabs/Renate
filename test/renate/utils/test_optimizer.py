# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.defaults import TASK_ID
from renate.utils.optimizer import create_partial_optimizer


@pytest.mark.parametrize(
    "optimizer,kwargs",
    [
        ("SGD", {"lr": 0.01, "momentum": 0.0, "weight_decay": 0.5}),
        ("Adam", {"lr": 0.01, "weight_decay": 0.5}),
    ],
)
def test_make_optimizer_with_different_configurations(optimizer, kwargs):
    model = pytest.helpers.get_renate_module_mlp(
        num_outputs=10, num_inputs=10, hidden_size=32, num_hidden_layers=3
    )
    params = model.get_params(task_id=TASK_ID)

    opt = create_partial_optimizer(optimizer=optimizer, **kwargs)(params)
    assert isinstance(opt, torch.optim.Optimizer)


def test_unknown_optimizer_raises_error():
    optimizer_name = "Unknown Optimizer"
    with pytest.raises(ValueError, match=f"Unknown optimizer: {optimizer_name}."):
        create_partial_optimizer(optimizer=optimizer_name, lr=0.01, momentum=0.0, weight_decay=0.5)
