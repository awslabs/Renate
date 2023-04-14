# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional, Union

import renate.defaults as defaults
import torch
import transformers

from peft import (
    LoraConfig,
    get_peft_model,
)
from renate.benchmark.datasets.nlp_datasets import HuggingfaceTextDataModule
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule
from renate.models.renate_module import RenateWrapper

MODEL_TYPE = "gpt2-xl"
ALLOWED_MODEL_TYPES = {"distilbert", "bert-large-uncased", "gpt2", "gpt2-xl"}
MODEL_PARALLEL = False


def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    assert (
        MODEL_TYPE in ALLOWED_MODEL_TYPES
    ), f"MODEL_TYPE set in renate_config.py is wrongly set as {MODEL_TYPE}"

    peft_config = LoraConfig(
        task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1
    )

    model_config = transformers.AutoConfig.from_pretrained(MODEL_TYPE)

    if MODEL_TYPE == "distilbert":
        model_config.__dict__.update(dict(num_labels=2, return_dict=False))
    elif MODEL_TYPE == "bert-large-uncased":
        model_config.__dict__.update(
            dict(
                num_labels=2, return_dict=False, output_hidden_states=False, output_attentions=False
            )
        )
    elif "gpt2" in MODEL_TYPE:
        model_config.__dict__.update(
            dict(
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
                pad_token_id=0,
                num_labels=2,
                return_dict=False,
            )
        )

    transformer_model = transformers.AutoModelForSequenceClassification.from_config(
        model_config
    )
    transformer_model.gradient_checkpointing_enable()
    if MODEL_PARALLEL:
        transformer_model.transformer.parallelize()

    transformer_model = get_peft_model(transformer_model, peft_config)
    model = RenateWrapper(transformer_model, loss_fn=torch.nn.CrossEntropyLoss())

    if model_state_url is not None:
        state_dict = torch.load(str(model_state_url))
        model.load_state_dict(state_dict)
    return model


def data_module_fn(
    data_path: Union[Path, str], chunk_id: int, seed: int = defaults.SEED
) -> RenateDataModule:
    """Returns one of two movie review datasets depending on `chunk_id`."""
    if MODEL_TYPE == "distilbert":
        tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    else:
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    dataset_name = "imdb" if chunk_id else "rotten_tomatoes"
    data_module = HuggingfaceTextDataModule(
        str(data_path),
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        val_size=0.2,
        seed=seed,
    )
    return data_module
