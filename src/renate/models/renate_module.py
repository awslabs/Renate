# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Set

import torch
import torch.nn as nn

from renate.models.layers import ContinualNorm


class RenateModule(torch.nn.Module, ABC):
    """A simple model wrapping the torch.nn.Module and providing additional functionalities
    to save the model, load it, add task-specific parameters to the model and retrieve internal
    representations of the inputs.

    Args:
        constructor_arguments: hyperparameters needed to instantiate the model (e.g., number of layers)
        loss_fn: The loss function being optimized during the training.
    """

    def __init__(self, constructor_arguments: dict, loss_fn: torch.nn.Module):
        super(RenateModule, self).__init__()
        self._constructor_arguments = copy.deepcopy(constructor_arguments)
        self.loss_fn = loss_fn
        self._tasks_params_ids: Set[str] = set()
        self._intermediate_representation_cache: List[torch.Tensor] = []
        self._hooks: List[Callable] = []

    @classmethod
    def from_state_dict(cls, state_dict):
        """Load the model from the state dict.

        Args:
            state_dict: the state dict of the model. This method works under the assumption that the content
            of the state_dict has been created by a RenateModule setting the appropriate values in the extra state.
            To ensure that users need to pass all the constructor arguments to the parent init method in a dictionary
            and call the `add_task_params` of the parent class when implementing the same method in their own module.
        """

        hyperparameters = state_dict["_extra_state"]["hyperparameters"]
        model = cls(**hyperparameters)

        for task in state_dict["_extra_state"]["tasks_params_ids"]:
            model.add_task_params(task)

        model.load_state_dict(state_dict)

        return model

    def get_extra_state(self) -> Any:
        """Get the hyperparameters, loss and task ids necessary to reconstruct the model."""
        return {
            "hyperparameters": self._constructor_arguments,
            "tasks_params_ids": self._tasks_params_ids,
            "loss_fn": self.loss_fn,
        }

    def set_extra_state(self, state: Any):
        """Extract the content of the `_extra_state` and set the related values in the module."""
        self._constructor_arguments = state["hyperparameters"]
        self._tasks_params_ids = state["tasks_params_ids"]
        self.loss_fn = state["loss_fn"]

    @abstractmethod
    def forward(self, x: torch.Tensor, task_id: Optional[str] = None) -> torch.Tensor:
        """Performs a forward pass on the inputs and returns the predictions.

        Task ID can be used to specify, for example, the output head to perform the evaluation with
        a specific data Chunk ID.

        Args:
            x: The input tensor.
            task_id: The identifier of the task for which predictions are made.
        """
        pass

    def _add_task_params(self, task_id: str) -> None:
        """User-facing function which adds new parameters associated to a specific task to the model, if any.

        This function should only be defined, but not called. For calling use `add_task_params()`, which also
        performs checks e.g. on whether the task id currently exists.

        Args:
            task_id: The task id for which the new parameters are added.
        """
        pass

    def get_params(self, task_id: Optional[str] = None) -> List[nn.Parameter]:
        """User-facing function which returns the list of parameters for the core model and a specific `task_id`.

        This function is then later used in the `Learner` to update only portion of the parameters. Corresponding
        to the current task.

        Args:
            task_id: The task id for which the new parameters are added.
        """
        return list(self.parameters())

    def add_task_params(self, task_id: Optional[str] = None) -> None:
        """Registers new parameters associated to a specific task to the model.

        This function should not be modified by the user. When overriding this method,
        make sure to keep the call to the same method in the parent class using
        `super(RenateModule, self).add_task_params(task_id)`.

        The method should not modify modules created in previous calls, beyond the ones defined
        in `self._add_task_params()`. The order of the calls is not guaranteed when the model
        is loaded after being saved.

        Args:
            task_id: The task id for which the new parameters are added.
        """
        if task_id in self._tasks_params_ids:
            return
        self._add_task_params(task_id)
        self._tasks_params_ids.add(task_id)

    def get_logits(self, x: torch.Tensor, task_id: Optional[str] = None) -> torch.Tensor:
        """Returns the logits for a given pair of input and task.
        By default, this method returns the output of the forward pass.

        Args:
            x: The input tensor.
            task_id: The task id.
        """
        return self.forward(x, task_id)

    def get_intermediate_representation(self) -> List[torch.Tensor]:
        """Returns the cached intermediate representation."""
        return self._intermediate_representation_cache

    def replace_batch_norm_with_continual_norm(self, num_groups: int = 32):
        """Replaces every occurence of Batch Normalization with Continual Normalization in the current module.

        Args:
            num_groups: Number of groups when considering the group normalization in the Continual Normalization.

        """

        def _replace(module):
            for name, child in module.named_children():
                if not list(module.children()):
                    _replace(child)
                if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    setattr(
                        module,
                        name,
                        ContinualNorm(
                            num_features=child.num_features,
                            eps=child.eps,
                            momentum=child.momentum,
                            affine=child.affine,
                            track_running_stats=child.track_running_stats,
                            num_groups=num_groups,
                        ),
                    )

        _replace(self)

    def _intermediate_representation_caching_hook(self) -> Callable:
        """Hook to cache intermediate representations during training."""

        def hook(m: torch.nn.Module, _, output: torch.Tensor) -> None:
            if m.training:
                self._intermediate_representation_cache.append(output)

        return hook

    def register_intermediate_representation_caching_hook(self, module: torch.nn.Module) -> None:
        """User-facing interface to select which module should cache intermediate representations during training.
        Store the reference to the hook to enable its removal.

        Args:
            module: The module to be hooked.
        """
        hook = module.register_forward_hook(self._intermediate_representation_caching_hook())
        self._hooks.append(hook)

    def deregister_hooks(self) -> None:
        """User-facing interface to remove all the hooks that were registered."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.reset_intermediate_representation_cache()

    def reset_intermediate_representation_cache(self) -> None:
        """Resets the intermediate representation cache."""
        self._intermediate_representation_cache = []
