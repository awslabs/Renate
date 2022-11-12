# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torchvision
from torch.utils.data import TensorDataset

from renate.utils import pytorch
from renate.utils.pytorch import randomly_split_data


@pytest.mark.parametrize("model", [torchvision.models.resnet18(pretrained=True)])
def test_reinitialize_model_parameters(model: torch.nn.Module):
    model.train()
    model(torch.randn(10, 3, 128, 128))  # Peform a forward pass, to change batchnorm buffers.
    params_before = [p.clone() for p in model.parameters()]
    buffers_before = [b.clone() for b in model.buffers()]
    pytorch.reinitialize_model_parameters(model)
    for p_before, p_after in zip(params_before, model.parameters()):
        assert not torch.allclose(p_before, p_after)
    for b_before, b_after in zip(buffers_before, model.buffers()):
        assert not torch.allclose(b_before, b_after)


@pytest.mark.parametrize(
    "proportions,expected_sizes",
    [
        [[0.5, 0.5], [50, 50]],
        [[0.3, 0.3, 0.4], [30, 30, 40]],
        [[0.333, 0.333, 0.334], [33, 33, 34]],
    ],
)
@pytest.mark.parametrize("splitter_func", [randomly_split_data])
def test_split_size(proportions, expected_sizes, splitter_func):
    dataset = TensorDataset(torch.randn(100, 1))
    splits = splitter_func(dataset, proportions, len(dataset))
    for i, split in enumerate(splits):
        assert len(split) == expected_sizes[i]


@pytest.mark.parametrize("splitter_func", [randomly_split_data])
def test_split_not_overlapping_splits(splitter_func):
    X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dataset = TensorDataset(X)
    splits = splitter_func(dataset, [0.5, 0.25, 0.25], len(dataset))
    for i in range(len(splits)):
        for j in range(len(splits)):
            if i == j:
                continue
            for k in range(len(splits[i])):
                assert splits[i][k] not in splits[j]


def test_random_splitting_sample_split_with_same_random_seed():
    X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dataset = TensorDataset(X)

    d_1_split_1, d_1_split_2 = randomly_split_data(dataset, [0.5, 0.5], 42)
    d_2_split_1, d_2_split_2 = randomly_split_data(dataset, [0.5, 0.5], 42)

    for i in range(5):
        assert torch.equal(d_1_split_1[i][0], d_2_split_1[i][0])
        assert torch.equal(d_1_split_2[i][0], d_2_split_2[i][0])
