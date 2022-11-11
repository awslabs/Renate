# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional, Tuple
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
    """A dataset that applies a given transformation to a given dataset.

    The class is used to apply additional transformations on a given dataset.
    :class:`renate.updaters.learner.ReplayLearner` uses `return_original_tensor` to get access to the original data
    points.
    Use of any transformations to data points or targets is optional.

    Args:
        dataset: The dataset to wrap.
        transform: Transformation to perform on the sample.
        target_transform: Transformation to perform on the target.
        return_original_tensor: If True, `__getitem__` a dict containing the original as well as the transformed tensor.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_original_tensor: bool = False,
    ) -> None:
        super(Dataset, self).__init__()
        self._dataset = dataset
        self._transform = transform
        self._target_transform = target_transform
        self._return_original_tensor = return_original_tensor

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[Tensor, Tensor], Dict[str, Tuple[Tensor, Tensor]]]:
        """Returns a dictionary containing the non-augmented and augmented samples."""
        data, target = self._dataset[idx]
        transformed_data, transformed_target = data, target

        if self._transform is not None:
            transformed_data = self._transform(data)

        if self._target_transform is not None:
            transformed_target = self._target_transform(target)
        if self._return_original_tensor:
            return {
                "original": (data, target),
                "transformed": (transformed_data, transformed_target),
            }
        return transformed_data, transformed_target

    def __len__(self) -> int:
        """Returns the number of data points in the dataset."""
        return len(self._dataset)
