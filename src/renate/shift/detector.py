# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from renate.utils.pytorch import move_tensors_to_device


class ShiftDetector:
    """Base class for distribution shift detectors.

    The main interface consists of two methods `fit` and `score`, which expect pytorch Dataset
    objects. One passes a reference dataset to the `fit` method. Then we can check query datasets
    for distribution shifts (relative to the reference dataset) using the `score` method. The
    `score` method returns a scalar shift score with the convention that high values indicate a
    distribution shift. For most methods, this score will be in [0, 1].

    Args:
        batch_size: Batch size used to iterate over datasets, e.g., for extracting features. This
            choice does not affect the result of the shift detector, but might affect run time.
        num_preprocessing_workers: Number of workers used in data loaders.
        device: Device to use for computations inside the detector.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_preprocessing_workers: int = 0,
        device: str = "cpu",
    ) -> None:
        self._batch_size = batch_size
        self._num_preprocessing_workers = num_preprocessing_workers
        self._device = device

    def fit(self, dataset: Dataset) -> None:
        """Fit the detector to a reference dataset."""
        raise NotImplementedError()

    def score(self, dataset: Dataset) -> float:
        """Compute distribution shift score for a query dataset."""
        raise NotImplementedError()

    def _make_data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Return a data loader to iterate over a dataset.

        Args:
            dataset: The dataset.
            shuffle: Whether to shuffle or not.
        """
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_preprocessing_workers,
        )


class ShiftDetectorWithFeatureExtractor(ShiftDetector):
    """Base class for detectors working on extracted features.

    These shift detectors extract some (lower-dimensional) features from the datasets, which are
    used as inputs to the shift detection methods. Subclasses have to overwrite `fit_with_features`
    and `score_with_features`.

    Args:
        feature_extractor: A pytorch model used as feature extractor.
        batch_size: Batch size used to iterate over datasets.
        num_preprocessing_workers: Number of workers used in data loaders.
        device: Device to use for computations inside the detector.
    """

    def __init__(
        self,
        feature_extractor: Optional[torch.nn.Module] = None,
        batch_size: int = 32,
        num_preprocessing_workers: int = 0,
        device: str = "cpu",
    ) -> None:
        super(ShiftDetectorWithFeatureExtractor, self).__init__(
            batch_size, num_preprocessing_workers, device
        )
        self._feature_extractor = feature_extractor or torch.nn.Identity()
        self._feature_extractor = self._feature_extractor.to(self._device)

    def fit(self, dataset: Dataset) -> None:
        """Fit the detector to a reference dataset."""
        X = self.extract_features(dataset)
        self._fit_with_features(X)

    def score(self, dataset: Dataset) -> float:
        """Compute distribution shift score for a query dataset."""
        X = self.extract_features(dataset)
        return self._score_with_features(X)

    @torch.no_grad()
    def extract_features(self, dataset: Dataset) -> torch.Tensor:
        """Extract features from a dataset."""
        dataloader = self._make_data_loader(dataset)
        Xs = []
        for batch in dataloader:
            X = move_tensors_to_device(batch[0], device=self._device)
            Xs.append(self._feature_extractor(X))
        X = torch.cat(Xs, dim=0).cpu()
        return X

    def _fit_with_features(self, X: torch.Tensor) -> None:
        raise NotImplementedError()

    def _score_with_features(self, X: torch.Tensor) -> float:
        raise NotImplementedError()
