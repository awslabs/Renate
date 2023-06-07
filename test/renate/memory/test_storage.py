# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.utils.data import Dataset

from renate.memory.storage import FileTensorStorage


def nested_tensors_equal(t1, t2):
    if type(t1) is not type(t2):
        return False
    if isinstance(t1, torch.Tensor):
        return torch.equal(t1, t2)
    if isinstance(t1, (tuple, list)):
        if len(t1) != len(t2):
            return False
        return all(nested_tensors_equal(t1_, t2_) for t1_, t2_ in zip(t1, t2))
    if isinstance(t1, dict):
        if set(t1.keys()) != set(t2.keys()):
            return False
        return all(nested_tensors_equal(t1[key], t2[key]) for key in t1.keys())


def make_dataset_same_sizes(dataset_length, return_type):
    class CustomDataset(Dataset):
        def __init__(self, x, y, return_type="tensor") -> None:
            self.x, self.y = x, y
            self.return_type = return_type

        def __getitem__(self, index):
            if self.return_type == "tensor":
                return self.x[index]
            elif self.return_type == "tuple":
                return (self.x[index], self.y[index])
            elif self.return_type == "dict":
                return {"x": self.x[index], "y": self.y[index]}
            elif self.return_type == "list":
                return [self.x[index], self.y[index]]

        def __len__(self):
            return self.x.shape[0]

    data_x = torch.rand(dataset_length, 3, 32, 32)
    data_y = torch.randint_like(data_x, 5)

    return CustomDataset(data_x, data_y, return_type=return_type)


def make_dataset_different_sizes(dataset_length, return_type):
    class CustomDataset(Dataset):
        def __init__(self, return_type="tensor") -> None:
            self.return_type = return_type

        def __getitem__(self, index):
            x = torch.ones(index + 2, index + 3).float() * (index + 1)
            y = torch.ones(index + 2, index + 3).int() * (index + 2)
            if self.return_type == "tensor":
                return x
            elif self.return_type == "tuple":
                return (x, y)
            elif self.return_type == "dict":
                return {"x": x, "y": y}
            elif self.return_type == "list":
                return [x, y]

        def __len__(self):
            return dataset_length

    return CustomDataset(return_type=return_type)


@pytest.mark.parametrize("length", [1, 10])
@pytest.mark.parametrize("return_type", ["tuple", "dict", "tensor", "list"])
@pytest.mark.parametrize(
    "dataset_maker_fn", [make_dataset_same_sizes, make_dataset_different_sizes]
)
def test_memory_storage_different_sizes(tmpdir, length, return_type, dataset_maker_fn):
    ds = dataset_maker_fn(length, return_type)
    storage = FileTensorStorage(tmpdir)
    storage.dump_dataset(ds)

    del storage
    storage = FileTensorStorage(tmpdir)
    for i in range(length):
        assert nested_tensors_equal(storage[i], ds[i])
