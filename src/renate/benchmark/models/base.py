# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC
from typing import List, Optional

import torch
from torch import nn

from renate import defaults
from renate.models import RenateModule
from renate.models.prediction_strategies import ICaRLClassificationStrategy, PredictionStrategy

# TODO: merge unit tests for the submodules
class RenateBenchmarkingModule(RenateModule, ABC):
    def __init__(
        self,
        embedding_size: int,
        num_outputs: int,
        constructor_arguments: dict,
        loss_fn: torch.nn.Module,
        prediction_strategy: Optional[PredictionStrategy] = None,
        add_icarl_class_means: bool = True,
    ):
        constructor_arguments["num_outputs"] = num_outputs
        constructor_arguments["add_icarl_class_means"] = add_icarl_class_means
        super().__init__(
            constructor_arguments=constructor_arguments,
            loss_fn=loss_fn,
            prediction_strategy=prediction_strategy,
        )
        self._embedding_size = embedding_size
        self._num_outputs = num_outputs
        self._tasks_params: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.add_task_params(defaults.TASK_ID)
        if add_icarl_class_means:
            self.class_means = torch.nn.Parameter(
                torch.zeros((embedding_size, num_outputs)), requires_grad=False
            )

    def forward(self, x: torch.Tensor, task_id: str = defaults.TASK_ID) -> torch.Tensor:
        x = self.get_backbone(task_id=task_id)(x)
        if isinstance(self._prediction_strategy, ICaRLClassificationStrategy):
            return self._prediction_strategy(x, self.training, class_means=self.class_means)
        else:
            assert (
                self._prediction_strategy is None
            ), f"Unknown prediction strategy of type {type(self._prediction_strategy)}."
        return self.get_predictor(task_id)(x)

    def get_backbone(self, task_id: str = defaults.TASK_ID) -> nn.Module:
        return self._model

    def get_predictor(self, task_id: str = defaults.TASK_ID) -> nn.Module:
        return self._tasks_params[task_id]

    def _add_task_params(self, task_id: str = defaults.TASK_ID) -> None:
        """Adds new parameters associated to a specific task to the model."""
        self._tasks_params[task_id] = torch.nn.Linear(self._embedding_size, self._num_outputs)

    def get_params(self, task_id: str = defaults.TASK_ID) -> List[torch.nn.Parameter]:
        """Returns the list of parameters for the core model and a specific `task_id`."""
        return list(self._model.parameters()) + list(self._tasks_params[task_id].parameters())
