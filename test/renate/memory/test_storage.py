# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.memory.storage import MemoryMappedTensorStorage


def nested_tensors_equal(t1, t2):
    if type(t1) is not type(t2):
        return False
    if isinstance(t1, torch.Tensor):
        return torch.equal(t1, t2)
    if isinstance(t1, tuple):
        if len(t1) != len(t2):
            return False
        return all(nested_tensors_equal(t1_, t2_) for t1_, t2_ in zip(t1, t2))
    if isinstance(t1, dict):
        if set(t1.keys()) != set(t2.keys()):
            return False
        return all(nested_tensors_equal(t1[key], t2[key]) for key in t1.keys())


@pytest.mark.parametrize("length", [0, 1, 10])
@pytest.mark.parametrize(
    "data_point",
    [
        torch.tensor(1),
        torch.ones(3),
        (torch.ones(3), torch.tensor(1)),
        ({"a": torch.ones(2, 3, 3), "b": torch.zeros(2)}, torch.tensor(4)),
        {"a": (torch.ones(2, 3, 3), torch.zeros(2)), "b": torch.tensor(2)},
    ],
)
def test_storage(tmpdir, length, data_point):
    """Tests the memory-mapped tensor storage for different nested tensor structures."""
    storage = MemoryMappedTensorStorage(tmpdir, data_point, length)
    for i in range(length):
        storage[i] = data_point

    del storage
    storage = MemoryMappedTensorStorage(tmpdir, data_point, length)

    for i in range(length):
        assert nested_tensors_equal(storage[i], data_point)
