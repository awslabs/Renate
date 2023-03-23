# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Callable, Optional, Union

import datasets
import torch
import transformers

import renate.defaults as defaults
from renate.benchmark.datasets.nlp_datasets import HuggingfaceTextDataModule
from renate.benchmark.scenarios import ClassIncrementalScenario, Scenario
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule
from renate.models.renate_module import RenateWrapper


# class CompactTransformer(torch.nn.Module):
#     """Wrap a transformer to operate on single X-tensor.

#     This wraps a transformer model to take in a single tensor `X` of shape
#     `(batch_size, sequence_length, 2)`, where the last dimension stacks the token_ids and attention
#     mask.

#     Args:
#         model: The transformer model.
#     """

#     def __init__(self, model) -> None:
#         super(CompactTransformer, self).__init__()
#         self.model = model

#     def forward(self, X: torch.Tensor):
#         input_ids = X[:, :, 0]
#         attention_mask = X[:, :, 1]
#         return self.model(input_ids, attention_mask=attention_mask)[0]


# class CompactTextDataset(torch.utils.data.Dataset):
#     """Make a text dataset compatible with the (x, y) format expected by renate.

#     The dataset is assumed to return a dictionary containing the fields `input_ids`,
#     `attention_mask` and `label`, each containing torch tensors. The `input_ids` and
#     `attention_mask` will be stacked along an additional dimension to conform to the assumption of
#     a single x-tensor.
#     """

#     def __init__(self, dataset):
#         self._dataset = dataset

#     def __len__(self):
#         return len(self._dataset)

#     def __getitem__(self, idx):
#         elt = self._dataset[idx]
#         x = torch.stack([elt["input_ids"], elt["attention_mask"]], dim=1)
#         x = torch.squeeze(x, dim=0)  # First shape dim is always 1
#         y = elt["label"]
#         return x, y


# class HuggingfaceDataModule(RenateDataModule):
#     """Data module wrapping Huggingface datasets.

#     Args:
#         data_path: the path to the folder containing the dataset files.
#         src_bucket: the name of the s3 bucket. If not provided, downloads the data from original
#             source.
#         src_object_name: the folder path in the s3 bucket.
#         dataset_name: Name of the torchvision dataset.
#         val_size: Fraction of the training data to be used for validation.
#         seed: Seed used to fix random number generation.
#     """

#     def __init__(
#         self,
#         data_path: str,
#         dataset_name: str = "rotten_tomatoes",
#         tokenizer=None,
#         max_length: int = 128,
#         val_size: float = defaults.VALIDATION_SIZE,
#         seed: int = defaults.SEED,
#     ):
#         super(HuggingfaceDataModule, self).__init__(
#             data_path=data_path,
#             val_size=val_size,
#             seed=seed,
#         )
#         self._dataset_name = dataset_name
#         self._tokenizer = tokenizer
#         self._max_length = max_length

#     def prepare_data(self) -> None:
#         """Download data."""
#         self._train_data = datasets.load_dataset(
#             self._dataset_name, split="train", cache_dir=self._data_path
#         )
#         self._test_data = datasets.load_dataset(
#             self._dataset_name, split="test", cache_dir=self._data_path
#         )

#     def setup(self) -> None:
#         """Set up train, test and val datasets."""

#         def tokenize_fn(batch):
#             return self._tokenizer(
#                 batch["text"], padding="max_length", truncation=True, max_length=self._max_length
#             )

#         self._train_data = self._train_data.map(tokenize_fn, batched=True)
#         self._test_data = self._test_data.map(tokenize_fn, batched=True)
#         self._train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
#         self._test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
#         self._train_data = CompactTextDataset(self._train_data)
#         self._test_data = CompactTextDataset(self._test_data)
#         self._train_data, self._val_data = self._split_train_val_data(self._train_data)


def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    """Returns a model instance."""
    transformer_model = transformers.DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, return_dict=False
    )
    model = RenateWrapper(transformer_model, loss_fn=torch.nn.CrossEntropyLoss())
    if model_state_url is not None:
        state_dict = torch.load(str(model_state_url))
        model.load_state_dict(state_dict)
    return model


def data_module_fn(
    data_path: Union[Path, str], chunk_id: int, seed: int = defaults.SEED
) -> Scenario:
    """Returns a class-incremental scenario instance."""
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    data_module = HuggingfaceTextDataModule(
        str(data_path),
        dataset_name="rotten_tomatoes",
        tokenizer=tokenizer,
        val_size=0.2,
        seed=seed,
    )
    # class_incremental_scenario = ClassIncrementalScenario(
    #     data_module=data_module,
    #     class_groupings=[[0, 1], [2, 3]],
    #     chunk_id=chunk_id,
    # )
    # return class_incremental_scenario
    return data_module


def train_transform() -> Callable:
    """Returns a transform function to be used in the training."""
    return None


def test_transform() -> Callable:
    """Returns a transform function to be used for validation or testing."""
    return None
