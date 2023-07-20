# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets.utils import check_integrity

from renate import defaults
from renate.utils.file import download_folder_from_s3
from renate.utils.pytorch import randomly_split_data


class RenateDataModule(abc.ABC):
    """Data modules bundle code for data loading and preparation.

    A data module implements two methods for data preparation:
    - `prepare_data()` downloads the data to the local machine and unpacks it.
    - `setup()` creates pytorch dataset objects that return training, test and (possibly) validation
    data.
    These two steps are separated to streamline the process when launching multiple training jobs
    simultaneously, e.g., for hyperparameter optimization. In this case, `prepare_data()` is only
    called once per machine.

    After these two methods have been called, the data can be accessed using
    - `train_data()`
    - `test_data()`
    - `val_data()`,
    which return torch datasets (`torch.utils.data.Dataset`).

    Args:
        data_path: the path to the data to be loaded.
        src_bucket: the name of the s3 bucket.
        src_object_name: the folder path in the s3 bucket.
        val_size: Fraction of the training data to be used for validation.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        src_bucket: Union[Path, str, None] = None,
        src_object_name: Union[Path, str, None] = None,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super().__init__()
        self._data_path = data_path
        self._src_bucket = src_bucket
        self._src_object_name = src_object_name
        self._train_data: Optional[Dataset] = None
        self._val_data: Optional[Dataset] = None
        self._test_data: Optional[Dataset] = None
        self._train_collate_fn: Optional[Callable] = None
        self._val_collate_fn: Optional[Callable] = None
        self._test_collate_fn: Optional[Callable] = None
        assert 0.0 <= val_size <= 1.0
        self._val_size = val_size
        self._seed = seed

        self._dataset_name: str = ""

    @abc.abstractmethod
    def prepare_data(self) -> None:
        """Downloads datasets."""
        pass

    @abc.abstractmethod
    def setup(self) -> None:
        """Set up train, test and val datasets."""
        pass

    def train_data(self) -> Dataset:
        """Returns training dataset."""
        return self._train_data

    def val_data(self) -> Dataset:
        """Returns validation dataset."""
        return self._val_data

    def test_data(self) -> Dataset:
        """Returns test dataset."""
        return self._test_data

    def train_collate_fn(self) -> Optional[Callable]:
        """Returns collate_fn for train DataLoader."""
        return self._train_collate_fn

    def val_collate_fn(self) -> Optional[Callable]:
        """Returns collate_fn for validation DataLoader."""
        return self._val_collate_fn

    def test_collate_fn(self) -> Optional[Callable]:
        """Returns collate_fn for test DataLoader."""
        return self._test_collate_fn

    def _verify_file(self, file_name: str) -> bool:
        """A helper function that verifies that the required dataset files are downloaded and
        correct.
        """
        return check_integrity(
            os.path.join(self._data_path, self._dataset_name, file_name),
            self.md5s[file_name],
        )

    def _split_train_val_data(self, train_data: Dataset) -> Tuple[Dataset, Dataset]:
        """A helper function that splits the train data into train and validation sets."""
        if self._val_size == 0.0:
            return train_data, None
        else:
            return randomly_split_data(
                train_data, [1.0 - self._val_size, self._val_size], self._seed
            )


class CSVDataModule(RenateDataModule):
    """A data module loading data from CSV files.

    Args:
        data_path: Path to the folder containing the files.
        train_filename: Name of the CSV file containing the training data.
        test_filename: Name of the CSV file containing the test data.
        src_bucket: Name of an s3 bucket. If specified, the folder given by `src_object_name` will
            be downloaded from S3 to `data_path`.
        src_object_name: Folder path in the s3 bucket.
        target_name: the header of the column containing the target values.
        val_size: Fraction of the training data to be used for validation.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        train_filename: Union[Path, str] = "train.csv",
        test_filename: Union[Path, str] = "test.csv",
        target_name: str = "y",
        src_bucket: Union[Path, str, None] = None,
        src_object_name: Union[Path, str, None] = None,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super(CSVDataModule, self).__init__(
            data_path,
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            val_size=val_size,
            seed=seed,
        )
        self._train_filename = train_filename
        self._test_filename = test_filename
        self._target_name = target_name

    def prepare_data(self) -> None:
        """Downloads data folder from S3 if applicable."""
        if self._src_bucket is not None:
            download_folder_from_s3(self._src_bucket, self._src_object_name, self._data_path)

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        X, y = self._process_csv_data(str(self._train_filename))
        train_data = TensorDataset(X, y)
        self._train_data, self._val_data = self._split_train_val_data(train_data)
        X, y = self._process_csv_data(str(self._test_filename))
        self._test_data = TensorDataset(X, y)

    def _process_csv_data(self, filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reads data from a CSV file and returns features and labels."""
        data_path = os.path.join(self._data_path, filename)
        data = pd.read_csv(data_path)
        if self._target_name not in data.columns:
            raise KeyError(f"{self._target_name} is not a valid target name.")
        y = torch.from_numpy(data[self._target_name].to_numpy())
        data = data.loc[:, data.columns != self._target_name]
        X = torch.from_numpy(data.to_numpy()).float()
        return X, y
