# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

from transformers import AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast

from renate.benchmark.datasets.nlp_datasets import HuggingFaceTextDataModule
from renate.benchmark.models.transformer import (
    HuggingFaceSequenceClassificationTransformer,
    HuggingFaceSequenceClassificationTransformerWithLora,
)
from renate.models import RenateModule


def model_fn(
    pretrained_model_name: str,
    num_outputs: int,
    peft_type: Optional[str] = None,
    model_state_url: Optional[str] = None,
    lora_r: Optional[int] = 8,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    lora_bias: str = "none",
    lora_modules_to_save: Optional[List[str]] = None,
    lora_init_lora_weights: bool = True,
) -> RenateModule:
    """Returns the model that will be trained.

    Args:
        pretrained_model_name: Hugging Face pretrained model id.
        num_outputs: Number of classes.
        peft_type: Indicator for this example that allows for switching between finetuning and PEFT
            with LoRa.
        model_state_url: Only for continual learning. Location of old model checkpoint.
        lora_r: Attention dimension of LoRa.
        lora_alpha: Alpha term used in LoRa.
        lora_dropout: Dropout rate used for LoRa.
        lora_bias: Type of bias for Lora. Options: ``"none"``, ``"all"``, and ``"lora_only"``.
        lora_modules_to_save: List of layers to be trained and saved in addition to Lora layers.
        lora_init_lora_weights: Indicate whether to initialize Lora weights.
    """
    if peft_type is None:
        return HuggingFaceSequenceClassificationTransformer(
            pretrained_model_name=pretrained_model_name, num_outputs=num_outputs
        )
    elif peft_type == "lora":
        return HuggingFaceSequenceClassificationTransformerWithLora(
            pretrained_model_name=pretrained_model_name,
            num_outputs=num_outputs,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            bias=lora_bias,
            modules_to_save=lora_modules_to_save,
            init_lora_weights=lora_init_lora_weights,
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
    """Returns an object that gives Renate access to the data.

    Args:
        data_path: Location where the data is downloaded to.
        seed: Seed.
        dataset_name: Hugging Face dataset name.
        pretrained_model_name: Hugging Face pretrained model id.
        input_column: Column in Hugging Face dataset that refers to the input sentence.
        target_column: Column that refers to the classification label.
        val_size: Proportion of training data used for validation.
    """
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
