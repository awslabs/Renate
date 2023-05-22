# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.shift.kernels import RBFKernel
from renate.shift.mmd_helpers import mmd


def test_mmd_identical_data():
    """We expect high p-values for identical data."""
    X = torch.randn(100, 2)
    _, p_val = mmd(X, X, kernel=RBFKernel(), num_permutations=100)
    assert p_val == 1.0


def test_shift_detector_disjoint_data():
    """We expect low p-values for very different data (two disjoint Gaussian blobs)."""
    X0 = torch.randn(100, 2)
    X1 = torch.randn(100, 2) + 2.0
    _, p_val = mmd(X0, X1, kernel=RBFKernel(), num_permutations=100)
    assert p_val == 0.0
