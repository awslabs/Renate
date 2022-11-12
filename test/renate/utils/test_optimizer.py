# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.defaults import TASK_ID
from renate.utils.optimizer import create_optimizer, create_scheduler


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

    opt = create_optimizer(params, optimizer=optimizer, **kwargs)
    assert isinstance(opt, torch.optim.Optimizer)


def test_unknown_optimizer_raises_error():
    model = pytest.helpers.get_renate_module_mlp(
        num_outputs=10, num_inputs=10, hidden_size=32, num_hidden_layers=3
    )
    params = model.get_params(task_id=TASK_ID)

    with pytest.raises(ValueError):
        create_optimizer(
            params, optimizer="UNKNOWN_OPTIMIZER", lr=0.01, momentum=0.0, weight_decay=0.5
        )


@pytest.mark.parametrize(
    "scheduler,kwargs",
    [
        ("ConstantLR", {}),
        ("ExponentialLR", {"gamma": 0.5}),
        ("StepLR", {"step_size": 10, "gamma": 0.5}),
    ],
)
def test_make_scheduler_with_different_configurations(scheduler, kwargs):
    model = pytest.helpers.get_renate_module_mlp(
        num_outputs=10, num_inputs=10, hidden_size=32, num_hidden_layers=3
    )
    params = model.get_params(task_id=TASK_ID)
    opt = create_optimizer(params, optimizer="SGD", lr=0.01, momentum=0.0, weight_decay=0.5)

    sch = create_scheduler(opt, scheduler=scheduler, **kwargs)
    assert isinstance(sch, torch.optim.lr_scheduler._LRScheduler)


def test_unknown_scheduler_raises_error():
    model = pytest.helpers.get_renate_module_mlp(
        num_outputs=10, num_inputs=10, hidden_size=32, num_hidden_layers=3
    )
    params = model.get_params(task_id=TASK_ID)
    opt = create_optimizer(params, optimizer="SGD", lr=0.01, momentum=0.0, weight_decay=0.5)

    with pytest.raises(ValueError):
        create_scheduler(opt, scheduler="UNKNOWN_SCHEDULER", gamma=0.5)
