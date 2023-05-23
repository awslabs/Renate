# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.shift.kernels import RBFKernel


@pytest.mark.parametrize("kernel", [RBFKernel()])
@pytest.mark.parametrize(
    "X1, X2",
    [
        (torch.randn(20, 2), torch.randn(20, 2)),
        (torch.randn(10, 2), torch.randn(20, 2)),
        (torch.randn(20, 2), torch.randn(10, 2)),
    ],
)
def test_kernel_shapes(kernel, X1, X2):
    K = kernel(X1, X2)
    assert K.size() == (X1.size(0), X2.size(0))


@pytest.mark.parametrize("kernel", [RBFKernel()])
def test_kernel_shape_mismatch(kernel):
    with pytest.raises(Exception):
        kernel(torch.randn(10, 2), torch.randn(20, 3))


def test_rbf_vs_manual_computation():
    X0 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    X1 = torch.tensor([[1.0, 0.0], [2.0, 2.0]])
    kernel = RBFKernel(lengthscale=1.0)
    K = kernel(X0, X1)
    K_exp = torch.exp(-0.5 * torch.tensor([[1.0, 8.0], [2.0, 5.0]]))
    assert torch.allclose(K, K_exp)


def test_rbf_kernel_limits():
    """Tests limit behavior of RBF kernel"""
    X = torch.randn(10, 2)
    # Small lengthscales should result in vanishing off-diagonal terms.
    kernel = RBFKernel(lengthscale=1e-8)
    K = kernel(X, X)
    assert torch.allclose(K, torch.eye(10))
    # High lengthscales should result in a matrix of all ones.
    kernel = RBFKernel(lengthscale=1e8)
    K = kernel(X, X)
    assert torch.allclose(K, torch.ones(10))
