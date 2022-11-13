# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from renate.memory.buffer import DataDict, DataTuple
from renate.models.renate_module import RenateModule
from renate.updaters.learner_components.component import Component


class WeightedLossComponent(Component, ABC):
    """The abstract class implementing a weighted loss function.

    This is an abstract class from which each other loss should inherit from.

    Args:
        weight: A scaling coefficient which should scale the loss which gets returned.
        sample_new_memory_batch: Whether a new batch of data should be sampled from the memory buffer when the loss is calculated.
    """

    def __init__(self, weight: float, sample_new_memory_batch: bool, **kwargs: Any) -> None:
        super().__init__(weight=weight, sample_new_memory_batch=sample_new_memory_batch, **kwargs)

    def _verify_attributes(self) -> None:
        """Verify if attributes have valid values."""
        super()._verify_attributes()
        assert self._weight >= 0, "Weight must be larger than 0."

    def _register_parameters(self, weight: float, sample_new_memory_batch: bool) -> None:
        """Register parameters of the loss."""
        super()._register_parameters()
        self.register_buffer("_weight", torch.tensor(weight, dtype=torch.float))
        self.register_buffer(
            "_sample_new_memory_batch", torch.tensor(sample_new_memory_batch, dtype=torch.bool)
        )

    def set_weight(self, weight: float) -> None:
        self._weight.data = torch.tensor(
            weight, dtype=self._weight.dtype, device=self._weight.device
        )
        self._verify_attributes()

    def loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[DataTuple, DataDict],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        if self.weight == 0:
            return torch.tensor(0.0)
        return self._loss(
            outputs_memory=outputs_memory,
            batch_memory=batch_memory,
            intermediate_representation_memory=intermediate_representation_memory,
        )

    @abstractmethod
    def _loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[DataTuple, DataDict],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        pass


class WeightedCustomLossComponent(WeightedLossComponent):
    """Adds a (weighted) user-provided custom loss contribution.

    Args:
        loss_fn: The loss function to apply.
        weight: A scaling coefficient which should scale the loss which gets returned.
        sample_new_memory_batch: Whether a new batch of data should be sampled from the memory buffer when the loss is calculated.
    """

    def __init__(
        self,
        loss_fn: Callable,
        weight: float,
        sample_new_memory_batch: bool,
        **kwargs: Any
    ) -> None:
        super().__init__(weight=weight, sample_new_memory_batch=sample_new_memory_batch, **kwargs)
        self._loss_fn = loss_fn

    def _loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[DataTuple, DataDict],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Mean-squared error between current and previous logits on memory."""
        (_, y_memory), _ = batch_memory
        return self.weight * self._loss_fn(outputs_memory, y_memory)


class WeightedMeanSquaredErrorLossComponent(WeightedLossComponent):
    """Mean squared error between the current and previous logits computed with respect to the memory sample.

    Args:
        weight: A scaling coefficient which should scale the loss which gets returned.
        sample_new_memory_batch: Whether a new batch of data should be sampled from the memory buffer when the loss is calculated.
    """

    def _loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[DataTuple, DataDict],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Mean-squared error between current and previous logits on memory."""
        logits = outputs_memory
        (_, _), meta_data = batch_memory
        previous_logits = meta_data["outputs"]
        return self.weight * F.mse_loss(logits, previous_logits, reduction="mean")


class WeightedCrossEntropyLossComponent(WeightedLossComponent):
    """Cross entropy between the current logits computed with respect to the memory labels.

    Args:
        weight: A scaling coefficient which should scale the loss which gets returned.
        sample_new_memory_batch: Whether a new batch of data should be sampled from the memory buffer when the loss is calculated.
    """

    def _loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[DataTuple, DataDict],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Cross entropy computation with respect to logits and labels."""
        logits = outputs_memory
        (_, y_memory), _ = batch_memory
        return self.weight * F.cross_entropy(logits, y_memory, reduction="mean")


class WeightedPooledOutputDistillationLossComponent(WeightedLossComponent):
    """Pooled output feature distillation with respect to intermediate network features.

    As described in: Douillard, Arthur, et al. "Podnet: Pooled outputs distillation for small-tasks incremental learning."
    European Conference on Computer Vision. Springer, Cham, 2020.

    Given the intermediate representations collected at different parts of the network, minimise their Euclidean distance
    with respect to the cached representation. There are different `distillation_type`s trading-off plasticity and stability
    of the resultant representations. `normalize` enables the user to normalize the resultant feature representations
    to ensure that they are less affected by their magnitude.

    Args:
        weight: Scaling coefficient which scales the loss with respect to all intermediate representations.
        sample_new_memory_batch: Whether a new batch of data should be sampled from the memory buffer when the loss is calculated.
        distillation_type: Which distillation type to apply with respect to all intermediate representations.
        normalize: Whether to normalize both the current and cached features before computing the Frobenius norm.
    """

    def __init__(
        self,
        weight: float,
        sample_new_memory_batch: bool,
        distillation_type: str = "spatial",
        normalize: bool = True,
    ) -> None:
        self._distillation_type = distillation_type
        super().__init__(
            weight=weight,
            sample_new_memory_batch=sample_new_memory_batch,
            normalize=normalize,
        )

    def _register_parameters(
        self, weight: float, sample_new_memory_batch: bool, normalize: bool
    ) -> None:
        """Register parameters of the loss."""
        super()._register_parameters(weight=weight, sample_new_memory_batch=sample_new_memory_batch)
        self.register_buffer("_normalize", torch.tensor(normalize, dtype=torch.bool))

    def _save_to_state_dict(
        self, destination: Dict[str, Any], prefix: str, keep_vars: bool
    ) -> None:
        """Save attributes to state dict."""
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "distillation_type"] = self._distillation_type

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Load attributes from state dict."""
        self._distillation_type = state_dict.pop(prefix + "distillation_type")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _verify_attributes(self) -> None:
        """Verify if attributes have valid values."""
        super()._verify_attributes()
        if self._distillation_type not in ["pixel", "channel", "width", "height", "gap", "spatial"]:
            raise ValueError(f"Invalid distillation type: {self._distillation_type}")

    def _sum_reshape(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Sum the tensor according to specific dimension and reshape."""
        batch_size = x.shape[0]
        return x.sum(dim=dim).reshape(batch_size, -1)

    def _pod(self, features: torch.Tensor, features_memory: torch.Tensor) -> torch.Tensor:
        """Pooled output distillation with respect to intermediate and cached intermediate features.

        Args:
            features: Current intermediate features.
            features_memory: Cached intermediate features.
        """
        if features.shape != features_memory.shape:
            raise ValueError(
                f"The shape of the features and the cached features should be the same: {features.shape}, and: {features_memory.shape}"
            )

        features = features.pow(2)
        features_memory = features_memory.pow(2)

        if self._distillation_type == "channels":
            features, features_memory = self._sum_reshape(features, 1), self._sum_reshape(
                features_memory, 1
            )
        elif self._distillation_type == "width":
            features, features_memory = self._sum_reshape(features, 2), self._sum_reshape(
                features_memory, 2
            )
        elif self._distillation_type == "height":
            features, features_memory = self._sum_reshape(features, 3), self._sum_reshape(
                features_memory, 3
            )
        elif self._distillation_type == "gap":
            features = F.adaptive_avg_pool2d(features, (1, 1))[..., 0, 0]
            features_memory = F.adaptive_avg_pool2d(features_memory, (1, 1))[..., 0, 0]
        elif self._distillation_type == "spatial":
            features_h, features_memory_h = self._sum_reshape(features, 3), self._sum_reshape(
                features_memory, 3
            )
            features_w, features_memory_w = self._sum_reshape(features, 2), self._sum_reshape(
                features_memory, 2
            )
            features = torch.cat([features_h, features_w], dim=-1)
            features_memory = torch.cat([features_memory_h, features_memory_w], dim=-1)

        if self._normalize:
            features = F.normalize(features, dim=1, p=2)
            features_memory = F.normalize(features_memory, dim=1, p=2)

        return torch.frobenius_norm(features - features_memory, dim=-1).mean(dim=0)

    def set_distillation_type(self, distillation_type: str) -> None:
        self._distillation_type = distillation_type
        self._verify_attributes()

    def set_normalize(self, normalize: bool) -> None:
        self._normalize = torch.tensor(normalize, dtype=torch.bool, device=self._normalize.device)
        self._verify_attributes()

    def _loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[DataTuple, DataDict],
        intermediate_representation_memory: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the pooled output with respect to current and cached intermediate outputs from memory."""
        loss = 0.0
        (_, _), meta_data = batch_memory
        for n in range(len(intermediate_representation_memory)):
            features = intermediate_representation_memory[n]
            features_memory = meta_data[f"intermediate_representation_{n}"]
            loss += self._pod(features, features_memory)
        return (self.weight * loss) / len(intermediate_representation_memory)


class WeightedCLSLossComponent(WeightedLossComponent):
    """Complementary Learning Systems Based Experience Replay.

    Arani, Elahe, Fahad Sarfraz, and Bahram Zonooz.
    "Learning fast, learning slow: A general continual learning method based on complementary learning system."
    arXiv preprint arXiv:2201.12604 (2022).

    The implementation follows the Algorithm 1 in the respective paper. The complete `Learner`
    implementing this loss, is the `CLSExperienceReplayLearner`.

    Args:
        weight: A scaling coefficient which should scale the loss which gets returned.
        sample_new_memory_batch: Whether a new batch of data should be sampled from the memory buffer when the loss is calculated.
        model: The model that is being trained.
        stable_model_update_weight: The weight used in the update of the stable model.
        plastic_model_update_weight:  The weight used in the update of the plastic model.
        stable_model_update_probability:  The probability of updating the stable model at each training step.
        plastic_model_update_probability:  The probability of updating the plastic model at each training step.
    """

    def __init__(
        self,
        weight: float,
        sample_new_memory_batch: bool,
        model: RenateModule,
        stable_model_update_weight: float,
        plastic_model_update_weight: float,
        stable_model_update_probability: float,
        plastic_model_update_probability: float,
    ) -> None:
        super().__init__(
            weight=weight,
            sample_new_memory_batch=sample_new_memory_batch,
            stable_model_update_weight=stable_model_update_weight,
            plastic_model_update_weight=plastic_model_update_weight,
            stable_model_update_probability=stable_model_update_probability,
            plastic_model_update_probability=plastic_model_update_probability,
            iteration=0,
        )
        self._plastic_model: RenateModule = copy.deepcopy(model)
        self._stable_model: RenateModule = copy.deepcopy(model)
        self._plastic_model.deregister_hooks()
        self._stable_model.deregister_hooks()

    def _register_parameters(
        self,
        weight: float,
        sample_new_memory_batch: bool,
        stable_model_update_weight: float,
        plastic_model_update_weight: float,
        stable_model_update_probability: float,
        plastic_model_update_probability: float,
        iteration: int,
    ) -> None:
        """Register the parameters of the loss component."""
        super()._register_parameters(
            weight=weight,
            sample_new_memory_batch=sample_new_memory_batch,
        )
        self.register_buffer(
            "_stable_model_update_weight",
            torch.tensor(stable_model_update_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "_plastic_model_update_weight",
            torch.tensor(plastic_model_update_weight, dtype=torch.float32),
        )
        self.register_buffer(
            "_stable_model_update_probability",
            torch.tensor(stable_model_update_probability, dtype=torch.float32),
        )
        self.register_buffer(
            "_plastic_model_update_probability",
            torch.tensor(plastic_model_update_probability, dtype=torch.float32),
        )
        self.register_buffer("_iteration", torch.tensor(iteration, dtype=torch.int64))

    def _verify_attributes(self) -> None:
        """Verify if attributes have valid values."""
        super()._verify_attributes()
        assert 0.0 <= self._stable_model_update_weight
        assert 0.0 <= self._plastic_model_update_weight
        assert 0.0 <= self._stable_model_update_probability <= 1.0
        assert 0.0 <= self._plastic_model_update_probability <= 1.0
        assert self._plastic_model_update_probability > self._stable_model_update_probability
        assert self._plastic_model_update_weight <= self._stable_model_update_weight

    def _loss(
        self,
        outputs_memory: torch.Tensor,
        batch_memory: Tuple[DataTuple, DataDict],
        intermediate_representation_memory: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Computes the consistency loss with respect to averaged plastic and stable models."""
        (x_memory, y_memory), _ = batch_memory
        with torch.no_grad():

            outputs_plastic = self._plastic_model(x_memory)
            outputs_stable = self._stable_model(x_memory)

            probs_plastic = F.softmax(outputs_plastic, dim=-1)
            probs_stable = F.softmax(outputs_stable, dim=-1)

            label_mask = F.one_hot(y_memory, num_classes=outputs_stable.shape[-1]) > 0
            idx = (probs_stable[label_mask] > probs_plastic[label_mask]).unsqueeze(1)

            outputs = torch.where(
                idx,
                outputs_stable,
                outputs_plastic,
            )

        consistency_loss = F.mse_loss(outputs_memory, outputs.detach(), reduction="mean")
        return self.weight * consistency_loss

    @torch.no_grad()
    def _update_model_variables(
        self, model: RenateModule, original_model: RenateModule, weight: torch.Tensor
    ) -> None:
        """Performs exponential moving average on the stored model copies.

        Args:
            model: Whether the plastic or the stable model is updated.
            weight: The minimum weight used in the exponential moving average to update the model.
        """
        alpha = min(
            1.0 - torch.tensor(1.0, device=self._iteration.device) / (self._iteration + 1), weight
        )
        for ema_p, p in zip(model.parameters(), original_model.parameters()):
            ema_p.data.mul_(alpha).add_(p.data, alpha=1 - alpha)

    def on_train_batch_end(self, model: RenateModule) -> None:
        """Updates the model copies with the current weights,
        given the specified probabilities of update, and increments iteration counter."""
        self._iteration += 1
        if (
            torch.rand(1, device=self._plastic_model_update_probability.device)
            < self._plastic_model_update_probability
        ):
            self._update_model_variables(
                self._plastic_model, model, self._plastic_model_update_weight
            )

        if (
            torch.rand(1, device=self._stable_model_update_probability.device)
            < self._stable_model_update_probability
        ):
            self._update_model_variables(
                self._stable_model, model, self._stable_model_update_weight
            )

    def set_stable_model_update_weight(self, stable_model_update_weight: float) -> None:
        self._stable_model_update_weight.data = torch.tensor(
            stable_model_update_weight,
            dtype=self._stable_model_update_weight.dtype,
            device=self._stable_model_update_weight.device,
        )
        self._verify_attributes()

    def set_plastic_model_update_weight(self, plastic_model_update_weight: float) -> None:
        self._plastic_model_update_weight.data = torch.tensor(
            plastic_model_update_weight,
            dtype=self._plastic_model_update_weight.dtype,
            device=self._plastic_model_update_weight.device,
        )
        self._verify_attributes()

    def set_stable_model_update_probability(self, stable_model_update_probability: float) -> None:
        self._stable_model_update_probability.data = torch.tensor(
            stable_model_update_probability,
            dtype=self._stable_model_update_probability.dtype,
            device=self._stable_model_update_probability.device,
        )
        self._verify_attributes()

    def set_plastic_model_update_probability(self, plastic_model_update_probability: float) -> None:
        self._plastic_model_update_probability.data = torch.tensor(
            plastic_model_update_probability,
            dtype=self._plastic_model_update_probability.dtype,
            device=self._plastic_model_update_probability.device,
        )
        self._verify_attributes()
