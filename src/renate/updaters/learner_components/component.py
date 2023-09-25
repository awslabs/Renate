# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
from typing import Any, Dict, List, Optional, Tuple

import torch

from renate.models import RenateModule
from renate.types import NestedTensors


class Component(abc.ABC):
    """The abstract class implementing a Component, usable in the BaseExperienceReplayLearner.

    This is an abstract class from which each other component e.g. additional
    regularising loss or a module updater should inherit from.
    The components should be a modular and independent to an extent where they can be composed
    together in an ordered list to be deployed in the BaseExperienceReplayLearner.

    Args:
        weight: A scaling coefficient which should scale the loss which gets returned.
        sample_new_memory_batch: Whether a new batch of data should be sampled from the memory
            buffer when the loss is calculated.
    """

    def __init__(self, weight: float = 0, sample_new_memory_batch: bool = False) -> None:
        self.weight = weight
        self.sample_new_memory_batch = sample_new_memory_batch
        self._verify_attributes()

    def loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[Tuple[NestedTensors, torch.Tensor], Dict[str, torch.Tensor]],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Computes some user-defined loss which is added to the main training loss in the training
        step.

        Args:
            outputs_memory: The outputs of the model with respect to memory data (batch_memory).
            batch_memory: The batch of data sampled from the memory buffer, including the meta data.
            intermediate_representation_memory: Intermediate feature representations of the network
                upon passing the input through the network.
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

    def _verify_attributes(self) -> None:
        """Verify if attributes have valid values."""
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load relevant information from checkpoint."""
        pass

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Add relevant information to checkpoint."""
        pass
