# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional, Union

from wild_time_data import load_dataset
from wild_time_data.core import available_time_steps, dataset_classes

from renate import defaults
from renate.data.data_module import RenateDataModule
from renate.utils.file import download_folder_from_s3


class WildTimeDataModule(RenateDataModule):
    """Data module wrapping torchvision datasets.

    Args:
        data_path: the path to the folder containing the dataset files.
        dataset_name: Name of the wild time dataset.
        src_bucket: the name of the s3 bucket. If not provided, downloads the data from original
            source.
        src_object_name: the folder path in the s3 bucket.
        time_step: Time slice to be loaded.
        val_size: Fraction of the training data to be used for validation.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        dataset_name: str,
        src_bucket: Optional[str] = None,
        src_object_name: Optional[str] = None,
        time_step: int = 0,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super().__init__(
            data_path=data_path,
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            val_size=val_size,
            seed=seed,
        )
        self._dataset_name = dataset_name
        self.time_step = time_step

    def prepare_data(self) -> None:
        """Download data.

        If s3 bucket is given, the data is downloaded from s3, otherwise from the original source.
        """
        if self._src_bucket is None:
            load_dataset(
                dataset_name=self._dataset_name,
                time_step=available_time_steps(self._dataset_name)[0],
                split="train",
                data_dir=self._data_path,
            )
        else:
            dst_dir = Path(self._data_path) / dataset_classes[self._dataset_name].file_name
            if not dst_dir.exists():
                download_folder_from_s3(
                    src_bucket=self._src_bucket,
                    src_object_name=self._src_object_name,
                    dst_dir=str(dst_dir),
                )

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        train_data = load_dataset(
            dataset_name=self._dataset_name,
            time_step=available_time_steps(self._dataset_name)[self.time_step],
            split="train",
            data_dir=self._data_path,
        )
        self._train_data, self._val_data = self._split_train_val_data(train_data)
        self._test_data = load_dataset(
            dataset_name=self._dataset_name,
            time_step=available_time_steps(self._dataset_name)[self.time_step],
            split="test",
            data_dir=self._data_path,
        )
