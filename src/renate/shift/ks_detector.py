# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from scipy.stats import kstest

from renate.shift.detector import ShiftDetectorWithFeatureExtractor


class KolmogorovSmirnovCovariateShiftDetector(ShiftDetectorWithFeatureExtractor):
    """A Kolmogorov-Smirnov (KS) test on each feature.

    A KS test is a univariate two-sample test, which we perform separately for each feature. To
    aggregate these tests without running into multiple-testing problems, we use a Bonferroni
    correction, as proposed in

        Rabanser et al. Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.
        NeurIPS 2019.
    """

    def _fit_with_features(self, X: torch.Tensor) -> None:
        self._X_ref = X

    def _score_with_features(self, X: torch.Tensor) -> float:
        n_features = X.size(1)
        p_vals = [
            kstest(X[:, i].numpy(), self._X_ref[:, i].numpy()).pvalue for i in range(n_features)
        ]
        # Bonferroni correction: Reject only if the minimal p-value among the multiple tests is
        # lower than `alpha / num_tests`, where `alpha` is the significance level. Equivalently, we
        # multiply the p-value by `num_tests`.
        p_val = min(1.0, min(p_vals) * n_features)
        return 1.0 - p_val
