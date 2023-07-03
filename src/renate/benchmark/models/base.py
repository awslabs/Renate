# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC
from typing import Any, List, Optional

import torch
from torch import nn

from renate import defaults
from renate.models import RenateModule
from renate.models.prediction_strategies import ICaRLClassificationStrategy, PredictionStrategy
from renate.utils.deepspeed import convert_to_tensor, recover_object_from_tensor


# TODO: merge unit tests for the submodules
class RenateBenchmarkingModule(RenateModule, ABC):
    """Base class for all models provided by Renate.

    This class ensures that each models works with all ModelUpdaters when using the benchmarking
    feature of Renate. New models can extend this class or alternatively extend the RenateModule
    and make sure they are compatible with the considered ModelUpdater.

    Args:
        embedding_size: Representation size of the model after the backbone.
        num_outputs: The number of outputs of the model.
        constructor_arguments: Arguments needed to instantiate the model.
        prediction_strategy: By default a forward pass through the model. Some ModelUpdater must
            be combined with specific prediction strategies to work as intended.
        add_icarl_class_means: Specific parameters for iCaRL. Can be set to ``False`` if any other
            ModelUpdater is used.
    """

    def __init__(
        self,
        embedding_size: int,
        num_outputs: int,
        constructor_arguments: dict,
        prediction_strategy: Optional[PredictionStrategy] = None,
        add_icarl_class_means: bool = True,
    ):
        constructor_arguments["num_outputs"] = num_outputs
        constructor_arguments["add_icarl_class_means"] = add_icarl_class_means
        super().__init__(
            constructor_arguments=constructor_arguments,
        )
        self._embedding_size = embedding_size
        self._num_outputs = num_outputs
        self._prediction_strategy = prediction_strategy
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
        """Returns the model without the prediction head."""
        return self._backbone

    def get_predictor(self, task_id: str = defaults.TASK_ID) -> nn.Module:
        """Returns the model without the backbone."""
        return self._tasks_params[task_id]

    def _add_task_params(self, task_id: str = defaults.TASK_ID) -> None:
        """Adds new parameters associated to a specific task to the model."""
        self._tasks_params[task_id] = torch.nn.Linear(self._embedding_size, self._num_outputs)

    def get_params(self, task_id: str = defaults.TASK_ID) -> List[torch.nn.Parameter]:
        """Returns the list of parameters for the core model and a specific `task_id`."""
        return list(self.get_backbone(task_id=task_id).parameters()) + list(
            self.get_predictor(task_id=task_id).parameters()
        )

    def get_extra_state(self, encode=True) -> Any:
        """Get the constructor_arguments and task ids necessary to reconstruct the model.

        Encode converts the state into a torch tensor so that Deepspeed serialization works.
        We don't encode any of the super() calls, but encode only the final dict.
        """
        extra_state = super().get_extra_state(encode=False)
        extra_state["prediction_strategy"] = self._prediction_strategy
        return convert_to_tensor(extra_state) if encode else extra_state

    def set_extra_state(self, state: Any, decode=True):
        """Extract the content of the ``_extra_state`` and set the related values in the module.

        decode flag is to decode the tensor of pkl bytes."""
        super().set_extra_state(state, decode=decode)
        self._prediction_strategy = (
            recover_object_from_tensor(state)["prediction_strategy"]
            if decode
            else state["prediction_strategy"]
        )
