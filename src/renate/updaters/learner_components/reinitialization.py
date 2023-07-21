# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch

from renate.models.renate_module import RenateModule
from renate.updaters.learner_components.component import Component
from renate.utils.pytorch import reinitialize_model_parameters


class ReinitializationComponent(Component):
    """Resets the model using each layer's built-in reinitialization logic.

    See also `renate.utils.torch_utils.reinitialize_model_parameters`.
    """

    def on_train_start(self, model: RenateModule) -> None:
        reinitialize_model_parameters(model)


class ShrinkAndPerturbReinitializationComponent(Component):
    """Shrinking and Perturbation reinitialization through scaling the
    weights and adding random noise.

    Ash, J., & Adams, R. P. (2020). On warm-starting neural network training.
    Advances in Neural Information Processing Systems, 33, 3884-3894.

    Args:
        shrink_factor: A scaling coefficient applied to shrink the weights.
        sigma: variance of the random Gaussian noise added to the weights.
    """

    def __init__(self, shrink_factor: float, sigma: float) -> None:
        self._shrink_factor = shrink_factor
        self._sigma = sigma
        super().__init__()

    def _verify_attributes(self) -> None:
        """Verify if attributes have valid values."""
        super()._verify_attributes()
        assert self._shrink_factor > 0.0, "Shrink factor must be positive."
        assert self._sigma >= 0, "Sigma must be non-negative."

    @torch.no_grad()
    def on_train_start(self, model: RenateModule) -> None:
        """Shrink and perturb the model's weights."""
        for p in model.parameters():
            if self._shrink_factor != 1.0:
                p.mul_(self._shrink_factor)
            if self._sigma != 0.0:
                p.add_(self._sigma * torch.randn(p.size(), device=p.device))
