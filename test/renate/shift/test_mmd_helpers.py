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


def test_mmd_vs_manual():
    X0 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    X1 = torch.tensor([[1.0, 0.0], [2.0, 2.0]])
    mmd_val, _ = mmd(X0, X1, kernel=RBFKernel(lengthscale=1.0), num_permutations=0)
    # Compare against manual computation of the terms in the MMD formula.
    assert mmd_val.item() == pytest.approx(1.6065 + 1.0821 - 2 * 0.2687, abs=1e-4)
