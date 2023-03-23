# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import transformers

import renate.defaults as defaults
from renate.benchmark.datasets.nlp_datasets import HuggingfaceTextDataModule
from renate.benchmark.scenarios import Scenario
from renate.models import RenateModule
from renate.models.renate_module import RenateWrapper


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
