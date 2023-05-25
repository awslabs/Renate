# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Set

import torch

from renate.models.layers import ContinualNorm
from renate.types import NestedTensors
from renate.utils.deepspeed import convert_to_tensor, recover_object_from_tensor


class RenateModule(torch.nn.Module, ABC):
    """A class for torch models with some additional functionality for continual learning.

    ``RenateModule`` derives from ``torch.nn.Module`` and provides some additional functionality
    relevant to continual learning. In particular, this concerns saving and reloading the model
    when model hyperparameters (which might affect the architecture) change during hyperparameter
    optimization. There is also functionality to retrieve internal-layer representations for use
    in replay-based CL methods.

    When implementing a subclass of ``RenateModule``, make sure to call the base class' constructor
    and provide your model's constructor arguments. Besides that, you can define a
    ``RenateModule`` just like ``torch.nn.Module``.

    Example::

        class MyMNISTMLP(RenateModule):

        def __init__(self, num_hidden: int):
            super().__init__(
                constructor_arguments={"num_hidden": num_hidden}
                loss_fn=torch.nn.CrossEntropyLoss()
            )
            self._fc1 = torch.nn.Linear(28*28, num_hidden)
            self._fc2 = torch.nn.Linear(num_hidden, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self._fc1(x)
            x = torch.nn.functional.relu(x)
            return self._fc2(x)

    The state of a ``RenateModule`` can be retrieved via the ``RenateModule.state_dict()`` method,
    just as in ``torch.nn.Module``. When reloading a ``RenateModule`` from a stored state dict, use
    ``RenateModule.from_state_dict``. It wil automatically recover the hyperparameters and
    reinstantiate your model accordingly.

    Note: Some methods of ``RenateModule`` accept an optional ``task_id`` argument. This is in
    anticipation of future methods for continual learning scenarios where task identifiers are
    provided. It is currently not used.

    Args:
        constructor_arguments: Arguments needed to instantiate the model.
    """

    def __init__(self, constructor_arguments: dict):
        super(RenateModule, self).__init__()
        self._constructor_arguments = copy.deepcopy(constructor_arguments)
        self._tasks_params_ids: Set[str] = set()
        self._intermediate_representation_cache: List[torch.Tensor] = []
        self._hooks: List[Callable] = []

    @classmethod
    def from_state_dict(cls, state_dict):
        """Load the model from a state dict.

        Args:
            state_dict: The state dict of the model. This method works under the assumption that
                this has been created by `RenateModule.state_dict()`.
        """
        extra_state = recover_object_from_tensor(state_dict["_extra_state"])
        constructor_arguments = extra_state["constructor_arguments"]
        model = cls(**constructor_arguments)
        for task in extra_state["tasks_params_ids"]:
            model.add_task_params(task)
        # TODO: See https://github.com/awslabs/Renate/issues/236.
        # There are changes to the `class_means` or `componenets` of a model
        # that are not loaded, and should probably not be stored.
        model.load_state_dict(state_dict, strict=False)
        return model

    def get_extra_state(self, encode: bool = True) -> Any:
        """Get the constructor_arguments, and task ids necessary to reconstruct the model."""
        extra_state = {
            "constructor_arguments": self._constructor_arguments,
            "tasks_params_ids": self._tasks_params_ids,
        }
        return convert_to_tensor(extra_state) if encode else extra_state

    def set_extra_state(self, state: Any, decode: bool = True):
        """Extract the content of the ``_extra_state`` and set the related values in the module."""
        extra_state = recover_object_from_tensor(state) if decode else state
        self._constructor_arguments = extra_state["constructor_arguments"]
        self._tasks_params_ids = extra_state["tasks_params_ids"]

    @abstractmethod
    def forward(self, x: NestedTensors, task_id: Optional[str] = None) -> torch.Tensor:
        """Performs a forward pass on the inputs and returns the predictions.

        This method accepts a task ID, which may be provided by some continual learning scenarios.
        As an example, the task id may be used to switch between multiple output heads.

        Args:
            x: Input(s) to the model. Can be a single tensor, a tuple of tensor, or a dictionary
                mapping strings to tensors.
            task_id: The identifier of the task for which predictions are made.
        Returns:
            The model's predictions.
        """
        pass

    def get_params(self, task_id: Optional[str] = None) -> List[torch.nn.Parameter]:
        """User-facing function which returns the list of parameters.

        If a ``task_id`` is given, this should return only parameters used for the specific task.

        Args:
            task_id: The task id for which we want to retrieve parameters.
        """
        return list(self.parameters())

    def _add_task_params(self, task_id: str) -> None:
        """Adds new parameters, associated to a specific task, to the model.

        The method should not modify modules created in previous calls, beyond the ones defined
        in ``self._add_task_params()``. The order of the calls is not guaranteed when the model
        is loaded after being saved.

        Args:
            task_id: The task id for which the new parameters are added.
        """
        pass

    def add_task_params(self, task_id: Optional[str] = None) -> None:
        """Adds new parameters, associated to a specific task, to the model.

        This function should not be overwritten; use ``_add_task_params`` instead.

        Args:
            task_id: The task id for which the new parameters are added.
        """
        if task_id in self._tasks_params_ids:
            return
        self._add_task_params(task_id)
        self._tasks_params_ids.add(task_id)

    def get_logits(self, x: NestedTensors, task_id: Optional[str] = None) -> torch.Tensor:
        """Returns the logits for a given pair of input and task id.

        By default, this method returns the output of the forward pass. This may be overwritten
        with custom behavior, if necessary.

        Args:
            x: Input(s) to the model. Can be a single tensor, a tuple of tensor, or a dictionary
                mapping strings to tensors.
            task_id: The task id.
        """
        return self.forward(x, task_id)

    def get_intermediate_representation(self) -> List[torch.Tensor]:
        """Returns the cached intermediate representation."""
        return self._intermediate_representation_cache

    def replace_batch_norm_with_continual_norm(self, num_groups: int = 32) -> None:
        """Replaces every occurence of batch normalization with continual normalization.

        Pham, Q., Liu, C., & Hoi, S. (2022). Continual normalization: Rethinking batch
        normalization for online continual learning. arXiv preprint arXiv:2203.16102.

        Args:
            num_groups: Number of groups when considering the group normalization in continual
                normalization.
        """

        def _replace(module):
            for name, child in module.named_children():
                if not list(module.children()):
                    _replace(child)
                if isinstance(
                    child, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
                ):
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
        """Add a hook to cache intermediate representations during training.

        Store the reference to the hook to enable its removal.

        Args:
            module: The module to be hooked.
        """
        hook = module.register_forward_hook(self._intermediate_representation_caching_hook())
        self._hooks.append(hook)

    def deregister_hooks(self) -> None:
        """Remove all the hooks that were registered."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.reset_intermediate_representation_cache()

    def reset_intermediate_representation_cache(self) -> None:
        """Resets the intermediate representation cache."""
        self._intermediate_representation_cache = []


class RenateWrapper(RenateModule):
    """A simple wrapper around a torch model.

    If you are using a torch model with fixed hyperparameters, you can use this wrapper to expose
    it as a ``RenateModule``. In this case, do _not_ use the ``from_state_dict`` method but
    reinstantiate the model, wrap it, and call ``load_state_dict``. If a tuple or a dictionary of
    tensors is passed to the `RenateWrapper`'s forward function, it is unpacked before passing it
    to the torch model's forward function.

    Example::

        my_torch_model = torch.nn.Linear(28*28, 10)  # Instantiate your torch model.
        model = RenateWrapper(my_torch_model)
        state_dict = torch.load("my_state_dict.pt")
        model.load_state_dict(state_dict)

    Args:
        model: The torch model to be wrapped.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(constructor_arguments={})
        self._model = model

    def forward(self, x: NestedTensors, task_id: Optional[str] = None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            outputs = self._model(x)
        elif isinstance(x, tuple):
            outputs = self._model(*x)
        elif isinstance(x, dict):
            outputs = self._model(**x)
        else:
            raise TypeError(f"Expected tensor or tuple/dict of tensors; found {type(x)}.")

        if isinstance(outputs, tuple) and len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    @classmethod
    def from_state_dict(cls, state_dict):
        raise NotImplementedError(
            "RenateWrapper does not support `from_state_dict`. Instantiate the object using the "
            "standard constructor, then call `load_state_dict`."
        )
