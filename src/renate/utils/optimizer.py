# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List

import torch

import renate.defaults as defaults


def create_optimizer(
    params: List[torch.nn.Parameter],
    optimizer: defaults.SUPPORTED_OPTIMIZERS_TYPE = defaults.OPTIMIZER,
    lr: float = defaults.LEARNING_RATE,
    momentum: float = defaults.MOMENTUM,
    weight_decay: float = defaults.WEIGHT_DECAY,
) -> torch.optim.Optimizer:
    """Creates optimizer used to train the model.

    Args:
        params: The list of parameters to be updated.
        optimizer: The name of the optimizer to be used. Currently 'Adam' and 'SGD' are supported.
        lr: Learning rate to be used.
        momentum: Value for the momentum hyperparameter (if relevant).
        weight_decay: Value for the weight_decay hyperparameter (if relevant).
    """

    if optimizer == "SGD":
        return torch.optim.SGD(params, lr, momentum, weight_decay)
    elif optimizer == "Adam":
        return torch.optim.Adam(params, lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}.")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: defaults.SUPPORTED_LEARNING_RATE_SCHEDULERS_TYPE = defaults.LEARNING_RATE_SCHEDULER,
    step_size: int = defaults.LEARNING_RATE_SCHEDULER_STEP_SIZE,
    gamma: float = defaults.LEARNING_RATE_SCHEDULER_GAMMA,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Creates a learning rate scheduler used to train the model.

    Args:
        optimizer: The optimizer to be used.
        scheduler: The name of the scheduler to be used.
        step_size: Period of learning rate decay.
        gamma: Value for the gamma hyperparameter (if relevant).
    """

    if scheduler == "ConstantLR":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif scheduler == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}.")
