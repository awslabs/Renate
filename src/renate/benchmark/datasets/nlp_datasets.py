# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from typing import Literal, Optional, Union

import torchtext
from torchtext.data import to_map_style_dataset

from renate import defaults
from renate.data.data_module import RenateDataModule
from renate.utils.file import download_folder_from_s3


class TorchTextDataModule(RenateDataModule):
    """Dataset with data from torchtext.

    Source: https://pytorch.org/text/stable/datasets.html

    Args:
        data_path: the path to the folder containing the dataset files.
        src_bucket: the name of the s3 bucket. If not provided, downloads the data from original source.
        src_object_name: the folder path in the s3 bucket.
        dataset_name: Name of the torchvision dataset.
        val_size: If `val_size` is provided split the train data into train and validation according to `val_size`.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        src_bucket: Optional[str] = None,
        src_object_name: Optional[str] = None,
        dataset_name: str = "AG_NEWS",
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super(TorchTextDataModule, self).__init__(
            data_path,
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            transform=None,
            target_transform=None,
            val_size=val_size,
            seed=seed,
        )
        self._dataset_name = dataset_name
        self._dataset_dict = {
            "AG_NEWS": (torchtext.datasets.AG_NEWS, "ag_news_csv"),
            "AmazonReviewFull": (torchtext.datasets.AmazonReviewFull, "amazon_review_full_csv"),
            "DBpedia": (torchtext.datasets.DBpedia, "dbpedia_csv"),
        }
        assert (
            self._dataset_name in self._dataset_dict
        ), f"Dataset {self._dataset_name} currently not supported."

    def prepare_data(self) -> None:
        """Download torchtext dataset with given dataset_name. If the data is not available in local data_path,
        it is downloaded. If s3 bucket is provided, the data is downloaded from s3, otherwise from the
        original source in setup().
        """
        _, dataset_pathname = self._dataset_dict[self._dataset_name]
        if self._src_bucket is not None:
            download_folder_from_s3(
                src_bucket=self._src_bucket,
                src_object_name=self._src_object_name,
                dst_dir=os.path.join(self._data_path, dataset_pathname),
            )

    def setup(self, stage: Optional[Literal["train", "val", "test"]] = None):
        """Make assignments: train/valid/test splits (Torchtext datasets only have train and test splits)."""
        cls, dataset_pathname = self._dataset_dict[self._dataset_name]
        if self._src_bucket is None:
            if stage in ["train", "val"] or stage is None:
                train_data = to_map_style_dataset(cls(root=self._data_path, split="train"))
                self._train_data, self._val_data = self._split_train_val_data(train_data)
            if stage == "test" or stage is None:
                self._test_data = to_map_style_dataset(cls(root=self._data_path, split="test"))
        else:
            if stage in ["train", "val"] or stage is None:
                with open(os.path.join(self._data_path, dataset_pathname, "train.csv"), "r") as f:
                    train_data = to_map_style_dataset(f)
                    self._train_data, self._val_data = self._split_train_val_data(train_data)

            if stage == "test" or stage is None:
                with open(os.path.join(self._data_path, dataset_pathname, "test.csv"), "r") as f:
                    self._test_data = to_map_style_dataset(f)
