# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, Optional

import torch
import transformers
from datasets import get_dataset_split_names, load_dataset

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
        split_names = get_dataset_split_names(self._dataset_name)
        if "train" not in split_names:
            raise RuntimeError(f"Dataset {self._dataset_name} does not contain a 'train' split.")
        if "test" not in split_names:
            raise RuntimeError(f"Dataset {self._dataset_name} does not contain a 'test' split.")
        self._train_data = load_dataset(
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
        self._test_data = load_dataset(self._dataset_name, split="test", cache_dir=self._data_path)
        if "validation" in split_names:
            logging.info(f"Using 'validation' split of dataset {self._dataset_name}.")
            self._val_data = load_dataset(
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


max_length = 384
stride = 128


class HuggingFaceExtractiveQADataModule(RenateDataModule):
    """Data module wrapping Hugging Face datasets for extractive question answering.

    TODO:"""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        dataset_name: str = "TODO",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super().__init__(
            data_path=data_path,
            val_size=val_size,
            seed=seed,
        )
        self._dataset_name = dataset_name
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs or defaults.TOKENIZER_KWARGS

    def prepare_data(self) -> None:
        """Download data."""
        pass

    def _preprocess_training_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self._tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def _preprocess_validation_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self._tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        raw_datasets = load_dataset("squad")
        self._train_data = raw_datasets["train"].map(
            self._preprocess_training_examples,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            num_proc=5,
        )
        self._train_data.set_format("torch")

        self._test_data = (
            raw_datasets["validation"]
            .map(
                self._preprocess_validation_examples,
                batched=True,
                remove_columns=raw_datasets["validation"].column_names,
                num_proc=5,
            )
            .remove_columns(["example_id", "offset_mapping"])
        )
        self._test_data.set_format("torch")

        self._val_data = self._test_data


class HuggingFaceLanguageModelingModule(RenateDataModule):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        dataset_name: str = "TODO",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super().__init__(
            data_path=data_path,
            val_size=val_size,
            seed=seed,
        )
        self._dataset_name = dataset_name
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs or defaults.TOKENIZER_KWARGS

    def prepare_data(self) -> None:
        """Download data."""
        pass

    def setup(self) -> None:
        eli5 = load_dataset("eli5", split="train_asks[:5000]")
        eli5 = eli5.train_test_split(test_size=0.2)
        eli5 = eli5.flatten()

        def preprocess_function(examples):
            return self._tokenizer([" ".join(x) for x in examples["answers.text"]])

        tokenized_eli5 = eli5.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=eli5["train"].column_names,
        )

        block_size = 128

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of block_size.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_dataset = tokenized_eli5.map(group_texts, batched=True)
        self._train_data = lm_dataset["train"]
        self._train_data.set_format("torch")
        self._val_data = lm_dataset["test"]
        self._val_data.set_format("torch")
