# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from renate.defaults import TASK_ID
from renate.models.renate_module import RenateModule


class MultiLayerPerceptron(RenateModule):
    """A simple Multi Layer Perceptron with hidden layers, activation and Batch Normalization if enabled.

    Args:
        num_inputs: Number of input nodes.
        num_outputs: Number of output nodes.
        num_hidden_layers: Number of hidden layers.
        hidden_size: Uniform hidden size or the list or tuple of hidden sizes for individual hidden layers.
        loss: Loss function to be used for training.
        activation: Activation name, matching activation name in `torch.nn` to be used between the hidden layers.
        batch_normalization: Whether to use Batch Normalization after the activation. By default the Batch Normalization
            tracks the running statistics.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_hidden_layers: int,
        hidden_size: Union[int, List[int], Tuple[int]],
        loss: nn.Module = nn.CrossEntropyLoss(),
        activation: str = "ReLU",
        batch_normalization: bool = False,
    ) -> None:
        super().__init__(
            constructor_arguments={
                "num_inputs": num_inputs,
                "num_outputs": num_outputs,
                "num_hidden_layers": num_hidden_layers,
                "hidden_size": hidden_size,
                "activation": activation,
                "batch_normalization": batch_normalization,
            },
            loss_fn=loss,
        )
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size for _ in range(num_hidden_layers + 1)]
        assert len(hidden_size) == num_hidden_layers + 1

        activation = getattr(nn, activation)

        hidden_size = [num_inputs] + hidden_size
        layers = [nn.Flatten()]
        for i in range(num_hidden_layers + 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(activation())
            if batch_normalization:
                layers.append(nn.BatchNorm1d(hidden_size[i + 1]))

        self._last_hidden_size = hidden_size[-1]
        self._num_outputs = num_outputs

        self._model = nn.Sequential(*layers)
        self._tasks_params: nn.ModuleDict = nn.ModuleDict()
        self.add_task_params(TASK_ID)

    def forward(self, x: torch.Tensor, task_id: str = TASK_ID) -> torch.Tensor:
        """Performs a forward pass on the inputs and returns the predictions."""
        return self._tasks_params[task_id](self._model(x))

    def _add_task_params(self, task_id: str = TASK_ID) -> None:
        """Adds new parameters associated to a specific task to the model."""
        self._tasks_params[task_id] = nn.Linear(
            self._last_hidden_size,
            self._num_outputs,
        )

    def get_params(self, task_id: str = TASK_ID) -> List[nn.Parameter]:
        """Returns the list of parameters for the core model and a specific `task_id`."""
        return list(self._model.parameters()) + list(self._tasks_params[task_id].parameters())
