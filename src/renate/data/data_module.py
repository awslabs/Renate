# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import os
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets.utils import check_integrity

from renate import defaults
from renate.utils.file import download_folder_from_s3
from renate.utils.pytorch import randomly_split_data


class RenateDataModule(abc.ABC):
    """The abstract class implementing RenateDatamodule class for loading Datasets with ease.

    This class can be extended to use different data formats, e.g. CSV data, image data, text data, etc.

    Args:
        data_path: the path to the data to be loaded.
        src_bucket: the name of the s3 bucket.
        src_object_name: the folder path in the s3 bucket.
        transform: Transformation or augmentation to perform on the sample.
        target_transform: Transformation or augmentation to perform on the target.
        val_size: If `val_size` is provided split the train data into train and validation according to `val_size`.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        src_bucket: str,
        src_object_name: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
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
        self._transform = transform
        self._target_transform = target_transform
        assert 0.0 <= val_size <= 1.0
        self._val_size = val_size
        self._seed = seed

        self._dataset_name: str = ""

    @abc.abstractmethod
    def prepare_data(self) -> None:
        """Downloads datasets."""
        pass

    @abc.abstractmethod
    def setup(self, stage: Optional[Literal["train", "val", "test"]] = None) -> None:
        """Make assignments: val/train/test splits."""
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

    def _verify_file(self, file_name: str) -> bool:
        """A helper function that verifies that the required dataset files are downloaded and correct."""
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
    """Dataset with data from a .CSV file.

    train/validation/test splits are stored in different.csv files. Label (target) column name should be specified.

    Args:
        data_path: the path to the folder containing the CSV datasets.
        src_bucket: the name of the s3 bucket.
        src_object_name: the folder path in the s3 bucket.
        filenames: Filenames for train/validation/test splits in the folder,
            e.g. {"train": "train_data.csv", "test": "test_data.csv"}
        target_name: the header of the column containing the target values.
        val_size: If `val_size` is provided split the train data into train and validation according to `val_size`.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        src_bucket: str,
        src_object_name: str,
        filenames: Dict[str, str],
        target_name: str = "y",
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super(CSVDataModule, self).__init__(
            data_path,
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            transform=None,
            target_transform=None,
            val_size=val_size,
            seed=seed,
        )
        self._filenames = filenames
        self._target_name = target_name

    def prepare_data(self) -> None:
        """Download data folder. We expect a folder with separate csv files for train/val/test.
        If the data is not available in local data_path, it is loaded from s3.
        """
        if self._src_bucket is None:
            raise ValueError("Source S3 bucket should be provided.")
        download_folder_from_s3(self._src_bucket, self._src_object_name, self._data_path)

    def setup(self, stage: Optional[Literal["train", "val", "test"]] = None) -> None:
        """Make assignments: train/validation/test splits."""
        # Assign train dataset
        if stage in ["train", "val"] or stage is None:
            X, y = self._process_csv_data(str(self._filenames["train"]))
            train_data = TensorDataset(X, y)
            self._train_data, self._val_data = self._split_train_val_data(train_data)

        # Assign test dataset
        if stage == "test" or stage is None:
            X, y = self._process_csv_data(str(self._filenames["test"]))
            self._test_data = TensorDataset(X, y)

    def _process_csv_data(self, filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """CSV data specific function to read the data from a file and return features and the labels."""
        data_path = os.path.join(self._data_path, filename)
        data = pd.read_csv(data_path)
        if self._target_name not in data.columns:
            raise KeyError(f"{self._target_name} is not a valid target name.")

        y = torch.from_numpy(data[self._target_name].to_numpy())
        data = data.loc[:, data.columns != self._target_name]
        X = torch.from_numpy(data.to_numpy()).float()
        return X, y
