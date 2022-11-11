# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from renate.memory.buffer import DataDict, DataTuple
from renate.models.renate_module import RenateModule


class Component(nn.Module, abc.ABC):
    """The abstract class implementing a Component, usable in the BaseExperienceReplayLearner.

    This is an abstract class from which each other component e.g. additional
    regularising loss or a module updater should inherit from.
    The components should be a modular and independent to an extend where they can be composed
    together in an ordered list to be deployed in the BaseExperienceReplayLearner.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._register_parameters(**kwargs)
        self._verify_attributes()

    def loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[DataTuple, DataDict],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Computes some user-defined loss which is added to the main training loss in the training step.

        Args:
            outputs_memory: The outputs of the model with respect to memory data (batch_memory).
            batch_memory: The batch of data sampled from the memory buffer, including the meta data.
            intermediate_representation_memory: Intermediate feature representations of the network upon passing the input through the network.
        """
        return torch.tensor(0.0)

    def on_train_start(self, model: RenateModule) -> None:
        """Updates the model parameters.

        Args:
            model: The model used for training.
        """
        pass

    def on_train_batch_end(self, model: RenateModule) -> None:
        """Internally records a training and optimizer step in the component.

        Args:
            model: The model used for training.
        """
        pass

    @property
    def weight(self) -> torch.Tensor:
        """The weight of the loss component."""
        return self._weight

    @property
    def sample_new_memory_batch(self) -> torch.Tensor:
        """Whether to sample a new memory batch or not."""
        return self._sample_new_memory_batch

    def _verify_attributes(self) -> None:
        """Verify if attributes have valid values."""
        pass

    def _register_parameters(self) -> None:
        """Function to register parameters of the component."""
        pass
