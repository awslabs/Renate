# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC
from typing import List, Optional

import torch

from renate import defaults
from renate.models import RenateModule
from renate.models.classification_strategies import (
    ClassificationStrategy,
    ICaRLClassificationStrategy,
)


class RenateBenchmarkingModule(RenateModule, ABC):
    def __init__(
        self,
        embedding_size: int,
        num_outputs: int,
        constructor_arguments: dict,
        loss_fn: torch.nn.Module,
        classification_strategy: Optional[ClassificationStrategy] = None,
    ):
        constructor_arguments["num_outputs"] = num_outputs
        super().__init__(
            constructor_arguments=constructor_arguments,
            loss_fn=loss_fn,
            classification_strategy=classification_strategy,
        )
        self._embedding_size = embedding_size
        self._num_outputs = num_outputs
        self._tasks_params: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.add_task_params(defaults.TASK_ID)
        self.class_means = torch.nn.Parameter(
            torch.zeros((embedding_size, num_outputs)), requires_grad=False
        )
        """Required for the ICaRLClassificationStrategy."""

    def forward(self, x: torch.Tensor, task_id: str = defaults.TASK_ID) -> torch.Tensor:
        """Performs a forward pass on the inputs and returns the predictions."""
        x = self._model(x)
        if isinstance(self._classification_strategy, ICaRLClassificationStrategy):
            return self._classification_strategy(x, self.training, class_means=self.class_means)
        else:
            assert (
                self._classification_strategy is None
            ), f"Unknown classification strategy of type {type(self._classification_strategy)}."
        return self._tasks_params[task_id](x)

    def _add_task_params(self, task_id: str = defaults.TASK_ID) -> None:
        """Adds new parameters associated to a specific task to the model."""
        self._tasks_params[task_id] = torch.nn.Linear(self._embedding_size, self._num_outputs)

    def get_params(self, task_id: str = defaults.TASK_ID) -> List[torch.nn.Parameter]:
        """Returns the list of parameters for the core model and a specific `task_id`."""
        return list(self._model.parameters()) + list(self._tasks_params[task_id].parameters())
