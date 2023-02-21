# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from collections import defaultdict

import pytest
import torch

from renate.memory import (
    GreedyClassBalancingBuffer,
    InfiniteBuffer,
    ReservoirBuffer,
    SlidingWindowBuffer,
)
from renate.memory.buffer import _make_storage, _insert_data_point, _get_data_point


class NestedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, nested_tensors):
        self._nested_tensors = nested_tensors
        self._length = self._get_len(nested_tensors)

    def _get_len(self, nested_tensors, expected_length=None) -> int:
        if isinstance(nested_tensors, torch.Tensor):
            length = nested_tensors.size(0)
            assert length == expected_length or expected_length is None
            return length
        elif isinstance(nested_tensors, tuple):
            for t in nested_tensors:
                expected_length = self._get_len(t, expected_length)
            return expected_length
        elif isinstance(nested_tensors, dict):
            for t in nested_tensors.values():
                expected_length = self._get_len(t, expected_length)
            return expected_length
        else:
            raise TypeError(f"Expected nested dict/tuple of tensors, found {type(nested_tensors)}.")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return _get_data_point(self._nested_tensors, idx)


@pytest.mark.parametrize("length", [1, 10])
@pytest.mark.parametrize(
    "data_point",
    [
        torch.tensor(1),
        torch.zeros(10),
        (torch.zeros(10, 10), torch.tensor(1)),
        ({"x": torch.zeros(10, 10), "z": torch.tensor(1)}, torch.tensor(1)),
    ],
)
def test_nested_tensor_storage(data_point, length):
    """Ensures that creation/insertion/extraction works for different nested tensor structures."""
    storage = _make_storage(data_point, length)
    _insert_data_point(storage, 0, data_point)
    _get_data_point(storage, 0)


@pytest.mark.parametrize("buffer_cls", [ReservoirBuffer, SlidingWindowBuffer])
@pytest.mark.parametrize("max_size", [1, 10, 100])
@pytest.mark.parametrize("num_updates", [1, 10])
@pytest.mark.parametrize(
    "dataset",
    [
        torch.utils.data.TensorDataset(torch.empty(10, 2)),
        torch.utils.data.TensorDataset(torch.empty(100, 2)),
        torch.utils.data.TensorDataset(torch.empty(10, 2), torch.arange(10)),
        torch.utils.data.TensorDataset(torch.empty(100, 2), torch.arange(100)),
        NestedTensorDataset({"X": torch.empty(10, 2), "y": torch.arange(10)}),
        NestedTensorDataset({"X": torch.empty(100, 2), "y": torch.arange(100)}),
        NestedTensorDataset(({"X": torch.empty(10, 2), "z": torch.empty(10)}, torch.arange(10))),
        NestedTensorDataset(({"X": torch.empty(100, 2), "z": torch.empty(100)}, torch.arange(100))),
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
def test_get_metadata(buffer):
    X = torch.randn(100, 3)
    y = torch.randint(0, 10, size=(100,))
    ds = torch.utils.data.TensorDataset(X, y)
    metadata = {"foo": torch.ones(100)}
    buffer.update(ds, metadata)

    metadata_ = buffer.metadata
    assert isinstance(metadata_, dict)
    assert "foo" in metadata_
    assert isinstance(metadata_["foo"], torch.Tensor)
    assert len(metadata_["foo"]) == len(buffer)
    assert torch.all(metadata_["foo"] == 1.0)


@pytest.mark.parametrize("buffer", [ReservoirBuffer(30), SlidingWindowBuffer(30)])
def test_getitem_returns_points_and_metadata(buffer):
    X = torch.randn(100, 3)
    y = torch.randint(0, 10, size=(100,))
    ds = torch.utils.data.TensorDataset(X, y)
    metadata = {"foo": torch.ones(100)}
    buffer.update(ds, metadata)

    for idx in [0, 1, 10, 29, -1, -2, -30]:
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


def test_greedy_is_balanced():
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
    counts = torch.tensor(
        [len(v) for v in buffer._class_to_index_map.values()], dtype=torch.float32
    )
    label_counts = defaultdict(int)
    assert 10 - 1 < counts.mean() < 10 + 1
    for i in range(len(buffer)):
        (_, y), _ = buffer[i]
        label_counts[y.item()] += 1
    buffer_class_counts = {k: len(v) for k, v in buffer._class_to_index_map.items()}
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


@pytest.mark.parametrize("max_size", [1, 10, 100])
def test_buffer_get_state_dict(max_size):
    buffer = ReservoirBuffer(max_size=max_size)
    for i in range(20):
        ds = torch.utils.data.TensorDataset(torch.arange(i * 10, (i + 1) * 10))
        metadata = {"x": torch.ones(10, 10)}
        buffer.update(ds, metadata)
    state_dict = buffer.state_dict()
    torch.all(torch.eq(buffer._data_points[0], state_dict["data_points"][0]))

    assert torch.all(torch.eq(buffer.metadata["x"], state_dict["metadata"]["x"]))

    for key in ["max_size", "storage_mode", "seed", "count", "data_points"]:
        assert getattr(buffer, "_" + key) == state_dict[key]
    assert buffer.metadata == state_dict["metadata"]
    assert state_dict["size"] == max_size
    assert len(state_dict["data_points"][0]) == max_size


@pytest.mark.parametrize("buffer_type", [ReservoirBuffer, SlidingWindowBuffer])
def test_buffer_load_state_dict(buffer_type):
    buffer = buffer_type(max_size=100)

    state_dict = {
        "buffer_class_name": buffer_type.__name__,
        "max_size": 100,
        "seed": 1,
        "count": 100,
        "size": 100,
        "data_points": {},
        "metadata": {},
        "storage_mode": "in_memory",
    }
    for i in range(100):
        x = torch.ones((5,))
        state_dict["data_points"]["0"] = (
            x.unsqueeze(0)
            if i == 0
            else torch.cat((state_dict["data_points"]["0"], x.unsqueeze(0)), dim=0)
        )
    state_dict["metadata"] = {"x": torch.ones(100, 5)}

    buffer.load_state_dict(state_dict)
    for key, value in state_dict["data_points"].items():
        assert torch.all(torch.eq(buffer._data_points[key], value))

    assert torch.all(torch.eq(buffer.metadata["x"], state_dict["metadata"]["x"]))

    for key in ["max_size", "storage_mode", "seed", "count", "data_points", "size"]:
        if not isinstance(getattr(buffer, "_" + key), list):
            assert getattr(buffer, "_" + key) == state_dict[key]


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
            assert torch.all(torch.eq(buffer[j + i * 10][0][0], torch.ones((3)) * i))


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
