# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import TensorDataset

from renate.data import ImageDataset
from renate.data.datasets import _EnumeratedDataset, _TransformedDataset


class MulTransform:
    def __init__(self, factor=3):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor


@pytest.mark.parametrize("transform", [lambda x: x + 1, torch.sqrt, None])
@pytest.mark.parametrize("target_transform", [lambda x: x * 2, lambda x: x**2, None])
def test_transformed_dataset(transform, target_transform):
    X = torch.arange(10)
    y = torch.arange(10)
    ds = TensorDataset(X, y)
    ds_transformed = _TransformedDataset(ds, transform, target_transform)
    X_transformed = torch.stack([ds_transformed[i][0] for i in range(len(ds_transformed))], dim=0)
    y_transformed = torch.stack([ds_transformed[i][1] for i in range(len(ds_transformed))], dim=0)
    X_transformed_exp = X if transform is None else transform(X)
    y_transformed_exp = y if target_transform is None else target_transform(y)
    assert torch.equal(X_transformed, X_transformed_exp)
    assert torch.equal(y_transformed, y_transformed_exp)


def test_enumerated_dataset():
    ds = TensorDataset(torch.arange(10) ** 2)
    ds_enumerated = _EnumeratedDataset(ds)
    for i in range(10):
        idx, batch = ds_enumerated[i]
        assert i == idx
        assert batch[0] == idx**2


def test_image_dataset(tmpdir):
    image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    image = Image.fromarray(image)
    image.save(os.path.join(tmpdir, "image.png"))
    dataset = ImageDataset(
        [os.path.join(tmpdir, "image.png")],
        [3],
        transform=transforms.Compose([transforms.ToTensor(), MulTransform(2)]),
        target_transform=MulTransform(3),
    )
    data, target = dataset[0]
    assert torch.equal(data, torch.ones(3, 10, 10) * 2)
    assert torch.equal(target, torch.tensor(3 * 3))
