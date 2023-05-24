# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch


class Kernel:
    """Base class for kernel functions."""

    def __init__(self):
        pass

    def _check_inputs(self, X0: torch.Tensor, X1: torch.Tensor):
        assert X0.dim() == X1.dim() == 2
        assert X0.size(1) == X1.size(1)
        assert X0.dtype is X1.dtype


class RBFKernel(Kernel):
    """A radial basis function kernel.

    This kernel has one hyperparameter, a scalar lengthscale. If this is set to `None` (default),
    the lengthscale will be set adaptively, at _each_ call to the kernel, via the median heuristic.

    Args:
        lengthscale: The kernel lengthscale. If `None` (default), this is set automatically via the
            median heuristic. Note: In this case, the lengthscale will be reset at each call to the
            kernel.
    """

    def __init__(self, lengthscale: Optional[float] = None):
        super().__init__()
        self._lengthscale = lengthscale

    @torch.no_grad()
    def __call__(self, X0: torch.Tensor, X1: torch.Tensor):
        self._check_inputs(X0, X1)
        dists = torch.cdist(X0, X1)
        lengthscale = self._lengthscale or torch.median(dists)
        return torch.exp(-0.5 * dists**2 / lengthscale**2)
