# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

from renate.shift.detector import ShiftDetectorWithFeatureExtractor
from renate.shift.kernels import RBFKernel
from renate.shift.mmd_helpers import mmd


class MMDCovariateShiftDetector(ShiftDetectorWithFeatureExtractor):
    """A kernel maximum mean discrepancy (MMD) test.

    This test was proposed by

    [1] Gretton, A., et al. A kernel two-sample test. JMLR (2012).

    We currently do not expose the choice of kernel. It defaults to an RBF kernel with a lengthscale
    set via the median heuristic.

    The detector computes an approximate p-value via a permutation test. The `score` method returns
    `1 - p_value` to conform to the convention that high scores indicate a shift.

    Args:
        feature_extractor: A pytorch model used as feature extractor.
        num_permutations: Number of permutations for permutation test.
        batch_size: Batch size used to iterate over datasets.
        num_preprocessing_workers: Number of workers used in data loaders.
        device: Device to use for computations inside the detector.
    """

    def __init__(
        self,
        feature_extractor: Optional[torch.nn.Module] = None,
        num_permutations: int = 1000,
        batch_size: int = 32,
        num_preprocessing_workers: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(feature_extractor, batch_size, num_preprocessing_workers, device)
        self._num_permutations = num_permutations

    def _fit_with_features(self, X: torch.Tensor):
        self._X_ref = X

    def _score_with_features(self, X: torch.Tensor) -> float:
        _, p_val = mmd(self._X_ref, X, kernel=RBFKernel(), num_permutations=self._num_permutations)
        return 1.0 - p_val.item()
