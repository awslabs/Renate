# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional, Union

import renate.defaults as defaults
import torch
from renate.benchmark.datasets.nlp_datasets import HuggingfaceTextDataModule
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule
from renate.models.model_utils import fine_tuning_mode, make_tokenizer, make_transformers_model
from renate.models.renate_module import RenateWrapper

MODEL_TYPE = "EleutherAI/gpt-j-6B"
FINE_TUNE_MODE = "peft"
INFERENCE_ONLY = False


def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    model = make_transformers_model(
        MODEL_TYPE, enable_gradient_checkpointing=True, load_in_8bit=INFERENCE_ONLY
    )
    ### wrap in peft or not
    if not INFERENCE_ONLY:
        transformer_model = fine_tuning_mode(model, FINE_TUNE_MODE)

    model = RenateWrapper(transformer_model, loss_fn=torch.nn.CrossEntropyLoss())

    if model_state_url is not None:
        state_dict = torch.load(str(model_state_url))
        model.load_state_dict(state_dict)
    return model


def data_module_fn(
    data_path: Union[Path, str], chunk_id: int, seed: int = defaults.SEED
) -> RenateDataModule:
    """Returns one of two movie review datasets depending on `chunk_id`."""
    tokenizer = make_tokenizer(MODEL_TYPE)

    dataset_name = "imdb" if chunk_id else "rotten_tomatoes"
    data_module = HuggingfaceTextDataModule(
        str(data_path),
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        val_size=0.2,
        seed=seed,
    )
    return data_module
