# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Optional, Tuple
from typing import Dict, List, Union

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Dataset class for image datasets where the images are loaded as raw images.

    Args:
        data: List of data paths to the images.
        labels: Labels of images.
        transform: Transformation or augmentation to perform on the sample.
        target_transform: Transformation or augmentation to perform on the target.
    """

    def __init__(
        self,
        data: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        with open(self.data[idx], "rb") as f:
            image = Image.open(f)
            image.load()

        # In case of black and white images, we should convert them to rgb
        if len(image.size) <= 2:
            image = image.convert("RGB")

        target = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)
        return image, target.long()


class _TransformedDataset(Dataset):
    """A dataset wrapper that applies transformations.

    This wrapper assumes that the the passed `dataset` returns a pair `(x, y)`.

    Args:
        dataset: The dataset to wrap.
        transform: Transformation to perform on the covariates `x`.
        target_transform: Transformation to perform on the target `y`.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(Dataset, self).__init__()
        self._dataset = dataset
        self._transform = transform
        self._target_transform = target_transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Returns the transformed data point."""
        data, target = self._dataset[idx]
        if self._transform is not None:
            data = self._transform(data)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return data, target

    def __len__(self) -> int:
        """Returns the number of data points in the dataset."""
        return len(self._dataset)


class _EnumeratedDataset(Dataset):
    """A dataset wrapper that enumerates data points.

    Args:
        dataset: The dataset to wrap.
    """

    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self._dataset = dataset

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """Returns index and data point."""
        return torch.tensor(idx, dtype=torch.long), self._dataset[idx]

    def __len__(self) -> int:
        """Returns the number of data points in the dataset."""
        return len(self._dataset)
