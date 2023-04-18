# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import os
from collections import defaultdict

import pytest
import torch

from renate.data.datasets import NestedTensorDataset
from renate.memory.buffer import (
    GreedyClassBalancingBuffer,
    InfiniteBuffer,
    ReservoirBuffer,
    SlidingWindowBuffer,
)


def nested_tensors_equal(t1, t2):
    if type(t1) is not type(t2):
        print("Type mismatch")
        return False
    if isinstance(t1, torch.Tensor):
        return torch.allclose(t1, t2, rtol=1e-3)
    if isinstance(t1, tuple):
        if len(t1) != len(t2):
            print("Tuple length mismatch")
            return False
        return all(nested_tensors_equal(t1_, t2_) for t1_, t2_ in zip(t1, t2))
    if isinstance(t1, dict):
        if set(t1.keys()) != set(t2.keys()):
            print("Dict key mismatch")
            return False
        return all(nested_tensors_equal(t1[key], t2[key]) for key in t1.keys())


@pytest.mark.parametrize("buffer_cls", [ReservoirBuffer, SlidingWindowBuffer])
@pytest.mark.parametrize("max_size", [1, 10, 100])
@pytest.mark.parametrize("num_updates", [1, 10])
@pytest.mark.parametrize(
    "dataset",
    [
        torch.utils.data.TensorDataset(torch.randn(10, 2)),
        torch.utils.data.TensorDataset(torch.randn(100, 2)),
        torch.utils.data.TensorDataset(torch.randn(10, 2), torch.arange(10)),
        torch.utils.data.TensorDataset(torch.randn(100, 2), torch.arange(100)),
        NestedTensorDataset({"X": torch.randn(10, 2), "y": torch.arange(10)}),
        NestedTensorDataset({"X": torch.randn(100, 2), "y": torch.arange(100)}),
        NestedTensorDataset(({"X": torch.randn(10, 2), "z": torch.randn(10)}, torch.arange(10))),
        NestedTensorDataset(({"X": torch.randn(100, 2), "z": torch.randn(100)}, torch.arange(100))),
    ],
)
def test_buffer_respects_max_size(buffer_cls, max_size, num_updates, dataset):
    buffer = buffer_cls(max_size)
    assert len(buffer) == 0
    for i in range(num_updates):
        buffer.update(dataset)
        assert len(buffer) == min((i + 1) * len(dataset), max_size)


@pytest.mark.parametrize("buffer", [ReservoirBuffer(30), SlidingWindowBuffer(30)])
def test_batching_with_metadata(buffer):
    X = torch.randn(100, 3)
    y = torch.randint(0, 10, size=(100,))
    ds = torch.utils.data.TensorDataset(X, y)
    metadata = {"foo": torch.ones(100)}
    buffer.update(ds, metadata)

    dataloader = torch.utils.data.DataLoader(buffer, batch_size=3)
    for batch, metadata_batch in dataloader:
        assert len(batch) == 2
        assert isinstance(metadata_batch, dict)
        assert "foo" in metadata_batch
        assert isinstance(metadata_batch["foo"], torch.Tensor)
        assert torch.all(metadata_batch["foo"] == 1.0)


@pytest.mark.parametrize("buffer", [ReservoirBuffer(30), SlidingWindowBuffer(30)])
def test_get_and_set_metadata(buffer):
    X = torch.randn(100, 3)
    y = torch.randint(0, 10, size=(100,))
    ds = torch.utils.data.TensorDataset(X, y)
    metadata = {"foo": torch.ones(100)}
    buffer.update(ds, metadata)

    foo_values = buffer.get_metadata("foo")
    assert isinstance(foo_values, torch.Tensor)
    assert foo_values.size() == (len(buffer),)
    assert torch.all(foo_values == 1.0)

    buffer.set_metadata("foo", 2 * torch.ones(len(buffer)))
    foo_values = buffer.get_metadata("foo")
    assert isinstance(foo_values, torch.Tensor)
    assert foo_values.size() == (len(buffer),)
    assert torch.all(foo_values == 2.0)


@pytest.mark.parametrize("buffer", [ReservoirBuffer(30), SlidingWindowBuffer(30)])
def test_getitem_returns_points_and_metadata(buffer):
    X = torch.randn(100, 3)
    y = torch.randint(0, 10, size=(100,))
    ds = torch.utils.data.TensorDataset(X, y)
    metadata = {"foo": torch.ones(100)}
    buffer.update(ds, metadata)

    for idx in [0, 1, 10, 29]:
        datapoint, metadata = buffer[idx]
        assert isinstance(metadata, dict) and ("foo" in metadata)
        assert metadata["foo"] == 1.0
        assert isinstance(datapoint, tuple) and len(datapoint) == 2
        assert datapoint[0].size() == (3,)
        assert datapoint[1].size() == ()

    for idx in [30, 31, 50]:
        with pytest.raises(IndexError):
            datapoint[idx]


def test_reservoir_is_uniform():
    """Present buffer with numbers from 1-10k. It should subsample uniformly, resulting in a mean of
    5k with a standard deviation of ~91.
    """
    buffer = ReservoirBuffer(max_size=1000)
    for i in range(10):
        x = torch.arange(i * 1000, (i + 1) * 1000, dtype=torch.float) + 1
        ds = torch.utils.data.TensorDataset(x)
        buffer.update(ds)
    xs = torch.stack([buffer[i][0][0] for i in range(len(buffer))], dim=0)
    assert 5000 - 5 * 91 < xs.mean() < 5000 + 5 * 91


def test_greedy_is_balanced(tmpdir):
    """Present buffer with 10 classes with respect to 10000 samples, where each class has 1000
    samples. When inspecting the ._class_counts attribute, it should be uniform.
    """
    buffer = GreedyClassBalancingBuffer(max_size=100)
    X = torch.ones(10000, 1)
    Y = torch.ones(10000, 1, dtype=torch.long)
    for i in range(10):
        Y[i * 1000 : (i + 1) * 1000] = i
    # Randomise the Y such that the classes are not presented in a sequence
    Y = Y[torch.randperm(10000)]
    # Split the data into 10 chunks and perform 10 updates
    for i in range(10):
        ds = torch.utils.data.TensorDataset(
            X[i * 1000 : (i + 1) * 1000], Y[i * 1000 : (i + 1) * 1000]
        )
        buffer.update(ds)
        buffer.save(tmpdir)
        state_dict = buffer.state_dict()
        del buffer
        buffer = GreedyClassBalancingBuffer(max_size=100)
        buffer.load_state_dict(state_dict)
        buffer.load(tmpdir)
    counts = torch.tensor([len(v) for v in buffer._indices_by_class.values()], dtype=torch.float32)
    label_counts = defaultdict(int)
    assert torch.all(counts >= 10 - 1)
    assert torch.all(counts <= 10 + 1)
    for i in range(len(buffer)):
        (_, y), _ = buffer[i]
        label_counts[y.item()] += 1
    buffer_class_counts = {k: len(v) for k, v in buffer._indices_by_class.items()}
    assert buffer_class_counts == label_counts


@pytest.mark.parametrize("max_size", [1, 10, 100])
@pytest.mark.parametrize("num_batches", [1, 10])
@pytest.mark.parametrize("batch_size", [1, 10, 20])
def test_sliding_window_keeps_most_recent(max_size, num_batches, batch_size):
    buffer = SlidingWindowBuffer(max_size)
    for i in range(num_batches):
        ds = torch.utils.data.TensorDataset(torch.arange(i * batch_size, (i + 1) * batch_size))
        buffer.update(ds)
    xs = torch.stack([buffer[i][0][0] for i in range(len(buffer))], dim=0)
    assert xs.max() == num_batches * batch_size - 1
    assert xs.min() == max(num_batches * batch_size - max_size, 0)


@pytest.mark.parametrize(
    "state_dict",
    [
        {
            "max_size": None,
            "storage_mode": "in_memory",
            "seed": 23,
            "count": 23,
            "data_points": [],
            "metadata": [],
        },
        {
            "max_size": 5,
            "storage_mode": None,
            "seed": 23,
            "count": 23,
            "data_points": [(torch.ones((5,)),) for _ in range(6)],
            "metadata": [],
        },
        {
            "max_size": 5,
            "storage_mode": "in_memory",
            "seed": "1",
            "count": 23,
            "data_points": [(torch.ones((5,)),) for _ in range(6)],
            "metadata": [],
        },
        {
            "max_size": 5,
            "storage_mode": "unknown_storage_mode",
            "seed": 23,
            "count": None,
            "data_points": [(torch.ones((5,)),) for _ in range(6)],
            "metadata": [(torch.ones((5,)),) for _ in range(6)],
        },
        {
            "max_size": 5,
            "storage_mode": "unknown_storage_mode",
            "seed": 23,
            "count": 10,
            "data_points": torch.ones(
                5,
            ),
            "metadata": [(torch.ones((5,)),) for _ in range(6)],
        },
        {
            "max_size": 5,
            "storage_mode": "unknown_storage_mode",
            "seed": 23,
            "count": 23,
            "data_points": [(torch.ones((5,)),) for _ in range(6)],
            "metadata": torch.ones((5,)),
        },
    ],
)
def test_buffer_loading_invalid_state_dict(state_dict):
    buffer = SlidingWindowBuffer(max_size=10)
    with pytest.raises(Exception):
        buffer.load_state_dict(state_dict)


def test_infinite_buffer():
    buffer = InfiniteBuffer()
    for i in range(5):
        sample_dataset = torch.utils.data.TensorDataset(torch.ones((10, 3)) * i)
        buffer.update(sample_dataset)

    for i in range(5):
        for j in range(10):
            assert torch.all(torch.eq(buffer[i * 10 + j][0][0], torch.ones((3)) * i))


@pytest.mark.parametrize("max_size", [1, 10, 100])
@pytest.mark.parametrize("buffer_cls", [ReservoirBuffer, SlidingWindowBuffer])
def test_buffer_same_size_on_disk_after_updates(tmpdir, max_size, buffer_cls):
    """Test that the buffer size on disk is the same after consequtive updates
    given that the max_size is reached."""
    buffer = buffer_cls(max_size=max_size)
    for i in range(20):
        ds = torch.utils.data.TensorDataset(torch.arange(i * 10, (i + 1) * 10))
        buffer.update(ds)

    torch.save(buffer.state_dict(), os.path.join(tmpdir, "buffer.pt"))
    disk_size = os.path.getsize(os.path.join(tmpdir, "buffer.pt"))
    state_dict = torch.load(os.path.join(tmpdir, "buffer.pt"))
    buffer.load_state_dict(state_dict)
    for i in range(20):
        ds = torch.utils.data.TensorDataset(torch.arange(i * 10, (i + 1) * 10))
        buffer.update(ds)

    torch.save(buffer.state_dict(), os.path.join(tmpdir, "buffer.pt"))
    if not isinstance(buffer, InfiniteBuffer):
        assert disk_size == os.path.getsize(os.path.join(tmpdir, "buffer.pt"))
    else:
        assert disk_size < os.path.getsize(os.path.join(tmpdir, "buffer.pt"))


@pytest.mark.parametrize("buffer_cls", [ReservoirBuffer, SlidingWindowBuffer])
@pytest.mark.parametrize("max_size", [10, 100])
@pytest.mark.parametrize("num_updates", [0, 1, 2])
@pytest.mark.parametrize(
    "dataset",
    [
        torch.utils.data.TensorDataset(torch.randn(10, 2)),
        torch.utils.data.TensorDataset(torch.randn(10, 2), torch.arange(10)),
        NestedTensorDataset({"X": torch.randn(10, 2), "y": torch.arange(10)}),
        NestedTensorDataset(({"X": torch.randn(10, 2), "z": torch.randn(10)}, torch.arange(10))),
    ],
)
@pytest.mark.parametrize("metadata", [None, {"a": torch.arange(10), "b": torch.zeros(10, 2)}])
def test_load_and_save_buffer(tmpdir, buffer_cls, max_size, num_updates, dataset, metadata):
    """Tests loading an saving of the buffer state."""
    buffer = buffer_cls(max_size)
    for _ in range(2):
        for _ in range(num_updates):
            buffer.update(dataset, metadata)
        elements_before = [copy.deepcopy(buffer[i]) for i in range(len(buffer))]
        buffer.save(tmpdir)
        state_dict = buffer.state_dict()
        del buffer
        buffer = buffer_cls(max_size)
        buffer.load_state_dict(state_dict)
        buffer.load(tmpdir)
        for j in range(len(buffer)):
            assert nested_tensors_equal(buffer[j], elements_before[j])


@pytest.mark.parametrize("buffer_cls", [ReservoirBuffer, SlidingWindowBuffer])
@pytest.mark.parametrize("max_size", [10, 100])
@pytest.mark.parametrize("num_updates", [0, 1, 2])
@pytest.mark.parametrize(
    "dataset",
    [
        torch.utils.data.TensorDataset(torch.randn(10, 2), torch.arange(10)),
    ],
)
@pytest.mark.parametrize("metadata", [None, {"a": torch.arange(10), "b": torch.zeros(10, 2)}])
@pytest.mark.parametrize("buffer_transform", [None, lambda x: x * 2])
@pytest.mark.parametrize("buffer_target_transform", [None, lambda y: y // 5])
def test_load_and_save_buffer_with_transforms(
    tmpdir,
    buffer_cls,
    max_size,
    num_updates,
    dataset,
    metadata,
    buffer_transform,
    buffer_target_transform,
):
    """Tests loading an saving of the buffer state."""
    buffer = buffer_cls(
        max_size, transform=buffer_transform, target_transform=buffer_target_transform
    )
    for _ in range(2):
        for _ in range(num_updates):
            buffer.update(dataset, metadata)
        elements_before = [copy.deepcopy(buffer[i]) for i in range(len(buffer))]
        buffer.save(tmpdir)
        state_dict = buffer.state_dict()
        del buffer
        buffer = buffer_cls(
            max_size, transform=buffer_transform, target_transform=buffer_target_transform
        )
        buffer.load_state_dict(state_dict)
        buffer.load(tmpdir)
        for j in range(len(buffer)):
            assert nested_tensors_equal(buffer[j], elements_before[j])
