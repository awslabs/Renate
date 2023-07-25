# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import logging
from typing import Any, Dict, Optional

import datasets
import torch
import transformers
from datasets import load_dataset

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


class MultiTextDataModule(RenateDataModule):
    """
    Inspired by the dataset used in "Episodic Memory in Lifelong Language Learning" by d’Autume et al.
    it is a collection of five different datasets: AGNews, Yelp, Amazon reviews, DBPedia, Yahoo Answers.

    The output space if the union of the output space of all the datasets.
    The dataset has 33 classes: 4 from AGNews, 5 from Yelp, 14 from DBPedia, 5 from Amazon reviews,
    10 from Yahoo. Amazon and Yelp have similar semantics and the classes have been merged.

    The maximum allowed training set size is 115k (the size of the smallest dataset).
    The dataset is balanced across datasets by construction and each data batch contains data from
    a single dataset.

    Args:
        data_path: the path to the folder where the data files will be downloaded to.
        tokenizer: Tokenizer to apply to the dataset. See https://huggingface.co/docs/tokenizers/
            for more information on tokenizers.
        tokenizer_kwargs: Keyword arguments passed when calling the tokenizer's ``__call__``
           function. Typical options are `max_length`, `padding` and `truncation`.
           See https://huggingface.co/docs/tokenizers/
           for more information on tokenizers. If `None` is passed, this defaults to
           `{"padding": "max_length", max_length: 128, truncation: True}`.
        train_size: the size of the data batch, must be smaller than 115k.
        val_size: Fraction of the training data to be used for validation.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        train_size: int = 1000,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super().__init__(
            data_path=data_path,
            val_size=val_size,
            seed=seed,
        )
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs or defaults.TOKENIZER_KWARGS
        self._train_size = train_size
        self._multi_dataset_info = {
            "ag_news": ["text", "label"],
            "yelp_review_full": ["text", "label"],
            "amazon_reviews_multi": ["review_body", "stars"],
            "dbpedia_14": ["content", "label"],
            "yahoo_answers_topics": ["question_title", "topic"],
        }
        self._observed_labels = {}

    def prepare_data(self) -> None:
        """Download dataset."""

        for dataset in self._multi_dataset_info.keys():
            for split in ["train", "test"] + (["validation"] if self._val_size > 0 else []):
                if dataset == "amazon_reviews_multi":
                    load_dataset(dataset, name="en", split=split, cache_dir=self._data_path)
                else:
                    load_dataset(dataset, split=split, cache_dir=self._data_path)

    def setup(self) -> None:
        """Set up train, test and val datasets."""

        def preprocess(example, dataset_name, text_field_name, label_field_name):
            return {
                **self._tokenizer(example[text_field_name], **self._tokenizer_kwargs),
                "label": get_label(dataset_name + str(example[label_field_name])),
            }

        def get_split(split_name, dataset_name):
            dataset = load_dataset(dataset_name, split=split_name, cache_dir=self._data_path)

            new_features = dataset.features.copy()
            # the following is hack needed because the output space of the new dataset is
            # the union of the output spaces of the single datasets
            new_features[self._multi_dataset_info[dataset_name][1]] = datasets.ClassLabel(
                num_classes=33
            )

            dataset = dataset.cast(new_features)
            rnd_idx = torch.randint(low=0, high=len(dataset), size=(self._train_size,)).tolist()
            dataset = dataset.select(indices=rnd_idx)
            dataset = dataset.map(
                functools.partial(
                    preprocess,
                    dataset_name=dataset_name,
                    text_field_name=self._multi_dataset_info[dataset_name][0],
                    label_field_name=self._multi_dataset_info[dataset_name][1],
                ),
                remove_columns=list(dataset.features),
                num_proc=4,
            )

            dataset.set_format(type="torch")

            return _InputTargetWrapper(dataset)

        def get_label(label: str):
            if "yelp_review_full" in label:
                label = label.replace("yelp_review_full", "amazon_reviews_multi")

            if len(self._observed_labels.keys()) == 0:
                self._observed_labels[label] = 0
                return 0

            elif label not in self._observed_labels.keys():
                max_val = max(self._observed_labels.values())
                self._observed_labels[label] = max_val + 1
                return max_val + 1

            else:
                return self._observed_labels[label]

        self._train_data = []
        self._test_data = []
        if self._val_size > 0:
            self._val_data = []

        for dataset in self._multi_dataset_info:
            self._train_data.append(get_split("train", dataset))
            self._test_data.append(get_split("test", dataset))
            if self._val_size > 0:
                self._val_data.append(get_split("validation", dataset))
