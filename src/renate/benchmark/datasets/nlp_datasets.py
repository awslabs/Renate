# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, Optional

import datasets
import torch
import transformers

from renate import defaults
from renate.data.data_module import RenateDataModule


class _InputTargetWrapper(torch.utils.data.Dataset):
    """Make a Hugging Face dataset comply with the `(input, target)` format."""

    def __init__(self, dataset, target_column: str = "label"):
        self._dataset = dataset
        self._target_column = target_column

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        target = item.pop(self._target_column)
        return item, target


class HuggingFaceTextDataModule(RenateDataModule):
    """Data module wrapping Hugging Face text datasets.

    This is a convenience wrapper to expose a Hugging Face dataset as a `RenateDataModule`. Datasets
    will be pre-tokenized and will return `input, target = dataset[i]`, where `input` is a
    dictionary with fields `["input_ids", "attention_mask"]`, and `target` is a tensor.

    We expect the dataset to have a "train" and a "test" split. An additional "validation" split
    will be used if present. Otherwise, a validation set may be split off of the training data
    using the `val_size` argument.

    Args:
        data_path: the path to the folder containing the dataset files.
        tokenizer: Tokenizer to apply to the dataset. See https://huggingface.co/docs/tokenizers/
            for more information on tokenizers.
        dataset_name: Name of the dataset, see https://huggingface.co/datasets. This is a wrapper
            for text datasets only.
        input_column: Name of the column containing the input text.
        target_column: Name of the column containing the target (e.g., class label).
        tokenizer_kwargs: Keyword arguments passed when calling the tokenizer's ``__call__``
           function. Typical options are `max_length`, `padding` and `truncation`.
           See https://huggingface.co/docs/tokenizers/
           for more information on tokenizers. If `None` is passed, this defaults to
           `{"padding": "max_length", max_length: 128, truncation: True}`.
        val_size: Fraction of the training data to be used for validation.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        dataset_name: str = "ag_news",
        input_column: str = "text",
        target_column: str = "label",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super(HuggingFaceTextDataModule, self).__init__(
            data_path=data_path,
            val_size=val_size,
            seed=seed,
        )
        self._dataset_name = dataset_name
        self._input_column = input_column
        self._target_column = target_column
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs or defaults.TOKENIZER_KWARGS

    def prepare_data(self) -> None:
        """Download data."""
        split_names = datasets.get_dataset_split_names(self._dataset_name)
        if "train" not in split_names:
            raise RuntimeError(f"Dataset {self._dataset_name} does not contain a 'train' split.")
        if "test" not in split_names:
            raise RuntimeError(f"Dataset {self._dataset_name} does not contain a 'test' split.")
        self._train_data = datasets.load_dataset(
            self._dataset_name, split="train", cache_dir=self._data_path
        )
        available_columns = list(self._train_data.features)
        if self._input_column not in available_columns:
            raise ValueError(
                f"Input column '{self._input_column}' does not exist in {self._dataset_name}. "
                f"Available columns: {available_columns}."
            )
        if self._target_column not in available_columns:
            raise ValueError(
                f"Target column '{self._target_column}' does not exist in {self._dataset_name}. "
                f"Available columns: {available_columns}."
            )
        self._test_data = datasets.load_dataset(
            self._dataset_name, split="test", cache_dir=self._data_path
        )
        if "validation" in split_names:
            logging.info(f"Using 'validation' split of dataset {self._dataset_name}.")
            self._val_data = datasets.load_dataset(
                self._dataset_name, split="validation", cache_dir=self._data_path
            )
        else:
            logging.info(
                f"No 'validation' split in dataset {self._dataset_name}. Splitting validation data "
                f"from the 'train' split using `val_size={self._val_size}`."
            )
            self._val_data = None

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        self.prepare_data()  # This will use cached datasets if they have already been downloaded.

        def tokenize_fn(batch):
            return self._tokenizer(batch[self._input_column], **self._tokenizer_kwargs)

        columns = ["input_ids", "attention_mask", self._target_column]

        self._train_data = self._train_data.map(tokenize_fn, batched=True)
        self._train_data.set_format(type="torch", columns=columns)
        self._train_data = _InputTargetWrapper(self._train_data, self._target_column)
        self._test_data = self._test_data.map(tokenize_fn, batched=True)
        self._test_data.set_format(type="torch", columns=columns)
        self._test_data = _InputTargetWrapper(self._test_data, self._target_column)
        if self._val_data is not None:
            self._val_data = self._val_data.map(tokenize_fn, batched=True)
            self._val_data.set_format(type="torch", columns=columns)
            self._val_data = _InputTargetWrapper(self._val_data, self._target_column)
        else:
            self._train_data, self._val_data = self._split_train_val_data(self._train_data)
