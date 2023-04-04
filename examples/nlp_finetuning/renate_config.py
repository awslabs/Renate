# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional, Union

import torch
import transformers

import renate.defaults as defaults
from renate.benchmark.datasets.nlp_datasets import HuggingfaceTextDataModule
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule
from renate.models.renate_module import RenateWrapper


def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    """Returns a DistilBert classification model."""
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
) -> RenateDataModule:
    """Returns one of two movie review datasets depending on `chunk_id`."""
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    dataset_name = "imdb" if chunk_id else "rotten_tomatoes"
    data_module = HuggingfaceTextDataModule(
        str(data_path),
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        val_size=0.2,
        seed=seed,
    )
    return data_module
