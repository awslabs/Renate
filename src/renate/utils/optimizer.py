# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Callable, List

import torch
from torch.nn import Parameter
from torch.optim import Optimizer

import renate.defaults as defaults


def create_partial_optimizer(
    optimizer: defaults.SUPPORTED_OPTIMIZERS_TYPE = defaults.OPTIMIZER,
    lr: float = defaults.LEARNING_RATE,
    momentum: float = defaults.MOMENTUM,
    weight_decay: float = defaults.WEIGHT_DECAY,
) -> Callable[[List[Parameter]], Optimizer]:
    """Creates a partial optimizer object.

    Args:
        optimizer: The name of the optimizer to be used. Options: `Adam` or `SGD`.
        lr: Learning rate to be used.
        momentum: Value for the momentum hyperparameter (if relevant).
        weight_decay: Value for the weight_decay hyperparameter (if relevant).
    """

    if optimizer == "SGD":
        return partial(torch.optim.SGD, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "Adam":
        return partial(torch.optim.Adam, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}.")
