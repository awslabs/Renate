# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast

from renate.benchmark.datasets.nlp_datasets import HuggingFaceTextDataModule
from renate.benchmark.models.transformer import HuggingFaceSequenceClassificationTransformer
from renate.models import RenateModule


def model_fn(
    pretrained_model_name: str,
    num_outputs: int,
    peft_type: Optional[str] = None,
    model_state_url: Optional[str] = None,
) -> RenateModule:
    return HuggingFaceSequenceClassificationTransformer(
        pretrained_model_name=pretrained_model_name,
        peft_type=peft_type,
        num_outputs=num_outputs,
    )


def data_module_fn(
    data_path: str,
    seed: int,
    dataset_name: str,
    pretrained_model_name: str,
    input_column: str,
    target_column: str,
    val_size: float = 0.0,
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    if isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
        tokenizer.pad_token = tokenizer.eos_token
    return HuggingFaceTextDataModule(
        data_path=data_path,
        dataset_name=dataset_name,
        input_column=input_column,
        target_column=target_column,
        tokenizer=tokenizer,
        val_size=val_size,
        seed=seed,
    )
