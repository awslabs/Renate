# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Any, Dict, Optional, Union

from transformers import PreTrainedTokenizer

from renate import defaults
from renate.benchmark.datasets.base import DataIncrementalDataModule
from renate.utils.file import download_folder_from_s3
from renate.utils.hf_utils import DataCollatorWithPaddingForWildTime


class WildTimeDataModule(DataIncrementalDataModule):
    """Data module wrapping around the Wild-Time data.

    Huaxiu Yao, Caroline Choi, Bochuan Cao, Yoonho Lee, Pang Wei Koh, Chelsea Finn:
    Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time. NeurIPS 2022

    Args:
        data_path: the path to the folder containing the dataset files.
        dataset_name: Name of the wild time dataset.
        src_bucket: the name of the s3 bucket. If not provided, downloads the data from original
            source.
        src_object_name: the folder path in the s3 bucket.
        time_step: Time slice to be loaded.
        tokenizer: Tokenizer to apply to the dataset. See https://huggingface.co/docs/tokenizers/
            for more information on tokenizers.
        tokenizer_kwargs: Keyword arguments passed when calling the tokenizer's ``__call__``
            function.
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
        tokenizer: Optional[PreTrainedTokenizer] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super().__init__(
            data_path=data_path,
            data_id=time_step,
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            val_size=val_size,
            seed=seed,
        )
        self._dataset_name = dataset_name
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs

    def prepare_data(self) -> None:
        """Download data.

        If s3 bucket is given, the data is downloaded from s3, otherwise from the original source.
        """
        if self._src_bucket is None:
            from wild_time_data import available_time_steps, load_dataset

            load_dataset(
                dataset_name=self._dataset_name,
                time_step=available_time_steps(self._dataset_name)[0],
                split="train",
                data_dir=self._data_path,
            )
        else:
            from wild_time_data.core import dataset_classes

            dst_dir = Path(self._data_path) / dataset_classes[self._dataset_name].file_name
            if not dst_dir.exists():
                download_folder_from_s3(
                    src_bucket=self._src_bucket,
                    src_object_name=self._src_object_name,
                    dst_dir=str(dst_dir),
                )

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        from wild_time_data import available_time_steps, load_dataset

        kwargs = {
            "dataset_name": self._dataset_name,
            "time_step": available_time_steps(self._dataset_name)[self.data_id],
            "data_dir": self._data_path,
            "in_memory": self._dataset_name != "fmow",
            "transform": None if self._dataset_name not in ["fmow", "yearbook"] else lambda x: x,
        }
        if self._tokenizer:
            kwargs["transform"] = lambda x: self._tokenizer(x, **(self._tokenizer_kwargs or {}))
        train_data = load_dataset(split="train", **kwargs)
        self._train_data, self._val_data = self._split_train_val_data(train_data)
        self._test_data = load_dataset(split="test", **kwargs)
        if self._dataset_name in ["huffpost", "arxiv"]:
            self._train_collate_fn = DataCollatorWithPaddingForWildTime(tokenizer=self._tokenizer)
            self._val_collate_fn = DataCollatorWithPaddingForWildTime(tokenizer=self._tokenizer)
            self._test_collate_fn = DataCollatorWithPaddingForWildTime(tokenizer=self._tokenizer)
