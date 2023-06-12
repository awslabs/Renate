# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.shift.ks_detector import KolmogorovSmirnovCovariateShiftDetector
from renate.shift.mmd_detectors import MMDCovariateShiftDetector


@pytest.mark.parametrize(
    "detector",
    [
        MMDCovariateShiftDetector(feature_extractor=None, num_permutations=100),
        KolmogorovSmirnovCovariateShiftDetector(feature_extractor=None),
    ],
)
def test_shift_detector_identical_data(detector):
    """We expect low scores for identical data."""
    dataset = torch.utils.data.TensorDataset(torch.randn(100, 2))
    detector.fit(dataset)
    score = detector.score(dataset)
    assert score == 0.0


@pytest.mark.parametrize(
    "detector",
    [
        MMDCovariateShiftDetector(feature_extractor=None, num_permutations=100),
        KolmogorovSmirnovCovariateShiftDetector(feature_extractor=None),
    ],
)
def test_shift_detector_disjoint_data(detector):
    """We expect high scores for very different data (two disjoint Gaussian blobs)."""
    dataset_ref = torch.utils.data.TensorDataset(torch.randn(100, 2))
    dataset_query = torch.utils.data.TensorDataset(torch.randn(100, 2) + 2.0)
    detector.fit(dataset_ref)
    score = detector.score(dataset_query)
    assert score == 1.0
