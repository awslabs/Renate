# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import torchvision
from torch.utils.data import Sampler, TensorDataset

from renate.benchmark.datasets.vision_datasets import TorchVisionDataModule
from renate.benchmark.scenarios import ClassIncrementalScenario
from renate.memory.buffer import ReservoirBuffer
from renate.utils import pytorch
from renate.utils.pytorch import (
    ConcatRandomSampler,
    cat_nested_tensors,
    complementary_indices,
    get_length_nested_tensors,
    randomly_split_data,
    unique_classes,
)


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


def test_get_length_nested_tensors():
    expected_batch_size = 10
    t = torch.zeros(expected_batch_size)
    assert get_length_nested_tensors(t) == expected_batch_size
    tuple_tensor = (t, t)
    assert get_length_nested_tensors(tuple_tensor) == expected_batch_size
    dict_tensor = {"k1": t, "k2": t}
    assert get_length_nested_tensors(dict_tensor) == expected_batch_size


def test_cat_nested_tensors():
    tensor_dim = 2
    first_dim_ones = 8
    zeros = torch.zeros((2, tensor_dim))
    ones = torch.ones((first_dim_ones, tensor_dim))
    result = cat_nested_tensors((zeros, ones))
    assert get_length_nested_tensors(result) == 10
    assert result.mean() == 0.8
    tuple_tensor = (zeros, ones)
    result = cat_nested_tensors((tuple_tensor, tuple_tensor))
    assert get_length_nested_tensors(result) == 4
    assert result[0].sum() == 0
    assert result[1].sum() == 2 * first_dim_ones * tensor_dim
    dict_tensor = {"zeros": zeros, "ones": ones}
    result = cat_nested_tensors((dict_tensor, dict_tensor))
    assert get_length_nested_tensors(result) == 4
    assert result["zeros"].sum() == 0
    assert result["ones"].sum() == 2 * first_dim_ones * tensor_dim


def test_cat_nested_tensors_wrong_shape():
    tensor1 = torch.zeros((2, 2))
    tensor2 = torch.zeros((2, 3))
    with pytest.raises(RuntimeError, match=r"Sizes of tensors must match except in dimension 0.*"):
        cat_nested_tensors((tensor1, tensor2))
    with pytest.raises(RuntimeError, match=r"Sizes of tensors must match except in dimension 0.*"):
        cat_nested_tensors(((tensor1, tensor1), (tensor1, tensor2)))
    with pytest.raises(RuntimeError, match=r"Sizes of tensors must match except in dimension 0.*"):
        cat_nested_tensors(({"k1": tensor1, "k2": tensor1}, {"k1": tensor1, "k2": tensor2}))


@pytest.mark.parametrize(
    "num_outputs, indices, expected_output",
    [
        [5, {2, 4}, [0, 1, 3]],
        [torch.rand(5, 5).size(1), {1, 2, 3}, [0, 4]],
        [torch.rand(5, 5).shape[1], {1, 2, 3}, [0, 4]],
    ],
)
def test_complementary_indices(num_outputs, indices, expected_output):
    assert expected_output == complementary_indices(num_outputs, indices)


@pytest.mark.parametrize("test_dataset", [True, False])
def test_unique_classes(tmpdir, test_dataset):
    if test_dataset:
        class_groupings = np.arange(0, 100).reshape(10, 10).tolist()
        data_module = TorchVisionDataModule(tmpdir, dataset_name="CIFAR100", val_size=0.2)
        data_module.prepare_data()
        for chunk_id in range(len(class_groupings)):
            scenario = ClassIncrementalScenario(
                data_module=data_module, groupings=class_groupings, chunk_id=chunk_id
            )
            scenario.setup()
            ds = scenario.val_data()
            predicted_unique = unique_classes(ds)

            assert predicted_unique == set(class_groupings[chunk_id])
    else:
        X = torch.randn(10, 3)
        y = torch.arange(0, 10)
        ds = torch.utils.data.TensorDataset(X, y)
        metadata = {"foo": torch.ones(10)}
        buffer = ReservoirBuffer(X.shape[0])
        buffer.update(ds, metadata)
        predicted_unique = unique_classes(buffer)
        assert predicted_unique == set(list(range(10)))


@pytest.mark.parametrize(
    "complete_dataset_iteration,expected_batches", [[None, 2], [0, 7], [1, 5], [2, 2]]
)
def test_concat_random_sampler(complete_dataset_iteration, expected_batches):
    sampler = ConcatRandomSampler(
        dataset_lengths=[15, 5, 20],
        batch_sizes=[2, 1, 8],
        complete_dataset_iteration=complete_dataset_iteration,
    )
    assert len(sampler) == expected_batches
    num_batches = 0
    for sample in sampler:
        assert all([s < 15 for s in sample[:2]])
        assert all([15 <= s < 20 for s in sample[2:3]])
        assert all([20 <= s < 40 for s in sample[3:]])
        num_batches += 1
    assert num_batches == expected_batches


def test_concat_random_sampler_distributed():
    """Tests behavior in case of distributed computing."""
    mock_sampler = Sampler(None)
    mock_sampler.rank = 1
    mock_sampler.num_replicas = 2
    expected_batches = 2
    sampler = ConcatRandomSampler(
        dataset_lengths=[16, 10], batch_sizes=[2, 2], sampler=mock_sampler
    )
    assert len(sampler) == expected_batches
    num_batches = 0
    for sample in sampler:
        assert all([7 < s < 16 for s in sample[:2]])
        assert all([21 <= s < 26 for s in sample[2:]])
        num_batches += 1
    assert num_batches == expected_batches
