# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from typing import Optional, Union

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
        val_size: Fraction of the training data to be used for validation.
        seed: Seed used to fix random number generation.
    """

    dataset_dict = {
        "AG_NEWS": (torchtext.datasets.AG_NEWS, "ag_news_csv"),
        "AmazonReviewFull": (torchtext.datasets.AmazonReviewFull, "amazon_review_full_csv"),
        "DBpedia": (torchtext.datasets.DBpedia, "dbpedia_csv"),
    }

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
            val_size=val_size,
            seed=seed,
        )
        if dataset_name not in TorchTextDataModule.dataset_dict:
            raise ValueError(f"Dataset {self._dataset_name} currently not supported.")
        self._dataset_name = dataset_name

    def prepare_data(self) -> None:
        """Download data.

        If s3 bucket is given, the data is downloaded from s3, otherwise from the original source.
        """
        cls, dataset_pathname = TorchTextDataModule.dataset_dict[self._dataset_name]
        if self._src_bucket is not None:
            download_folder_from_s3(
                src_bucket=self._src_bucket,
                src_object_name=self._src_object_name,
                dst_dir=os.path.join(self._data_path, dataset_pathname),
            )
        else:
            # Use torchtext to download.
            cls(root=self._data_path, split="train")
            cls(root=self._data_path, split="test")

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        cls, _ = TorchTextDataModule.dataset_dict[self._dataset_name]
        train_data = to_map_style_dataset(cls(root=self._data_path, split="train"))
        self._train_data, self._val_data = self._split_train_val_data(train_data)
        self._test_data = to_map_style_dataset(cls(root=self._data_path, split="test"))
