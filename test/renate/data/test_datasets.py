# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import TensorDataset

from renate.data.datasets import ImageDataset, _TransformedDataset


class MulTransform:
    def __init__(self, factor=3):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor


def test_transformed_dataset_returns_original_and_transformed():
    dataset = TensorDataset(torch.ones(10, 2), torch.ones(10, 2))
    transformed_dataset = _TransformedDataset(
        dataset,
        transform=MulTransform(factor=2),
        target_transform=MulTransform(factor=3),
        return_original_tensor=True,
    )
    for i in range(10):
        data = transformed_dataset[i]
        original = data["original"]
        transformed = data["transformed"]
        assert torch.equal(original[0], torch.ones(2))
        assert torch.equal(transformed[0], torch.ones(2) * 2)

        assert torch.equal(original[1], torch.ones(2))
        assert torch.equal(transformed[1], torch.ones(2) * 3)


def test_transformed_dataset_returns_only_transformed():
    dataset = TensorDataset(torch.ones(10, 2), torch.ones(10, 2))
    transformed_dataset = _TransformedDataset(
        dataset,
        transform=MulTransform(factor=4),
        target_transform=MulTransform(factor=5),
        return_original_tensor=False,
    )
    for i in range(10):
        data = transformed_dataset[i]
        assert torch.equal(data[0], torch.ones(2) * 4)
        assert torch.equal(data[1], torch.ones(2) * 5)


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
