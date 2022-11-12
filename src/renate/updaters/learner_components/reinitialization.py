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
        super().__init__(
            shrink_factor=shrink_factor,
            sigma=sigma,
        )

    def _register_parameters(self, shrink_factor: float, sigma: float) -> None:
        """Register parameters of the loss."""
        super()._register_parameters()
        self.register_buffer("_shrink_factor", torch.tensor(shrink_factor, dtype=torch.float))
        self.register_buffer("_sigma", torch.tensor(sigma, dtype=torch.float))
        self.register_buffer("_weight", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("_sample_new_memory_batch", torch.tensor(False, dtype=torch.bool))

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

    def set_shrink_factor(self, shrink_factor: float) -> None:
        self._shrink_factor.data = torch.tensor(
            shrink_factor, dtype=self._shrink_factor.dtype, device=self._shrink_factor.device
        )
        self._verify_attributes()

    def set_sigma(self, sigma: float) -> None:
        self._sigma.data = torch.tensor(sigma, dtype=self._sigma.dtype, device=self._sigma.device)
        self._verify_attributes()
