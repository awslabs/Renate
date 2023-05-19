# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from peft import (
    AdaLoraConfig,
    LoraConfig,
    PeftModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)
from torch import Tensor
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    T5ForConditionalGeneration,
)

from renate.models import RenateModule


def from_pretrained(
    pretrained_model_name: str, num_labels: int, return_dict: bool
) -> PreTrainedModel:
    auto_class = AutoModelForSequenceClassification
    if pretrained_model_name in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
        auto_class = T5ForConditionalGeneration
    return auto_class.from_pretrained(
        pretrained_model_name, num_labels=num_labels, return_dict=return_dict
    )


class HuggingFaceSequenceClassificationTransformer(RenateModule):
    """RenateModule which wraps around Hugging Face transformers.

    Args:
        pretrained_model_name: Hugging Face model id.
        num_outputs: Number of outputs.
        loss_fn: The loss function to be optimized during the training.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        constructor_arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        if constructor_arguments is None:
            constructor_arguments = {}
        constructor_arguments.update(
            {
                "pretrained_model_name": pretrained_model_name,
                "num_outputs": num_outputs,
            }
        )
        super().__init__(constructor_arguments=constructor_arguments, loss_fn=loss_fn)
        self._model = from_pretrained(
            pretrained_model_name, num_labels=num_outputs, return_dict=False
        )

    def use_peft(self, pretrained_model_name: str, peft_type: str) -> None:
        if peft_type == "lora":
            lora_kwargs = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.1}
            if pretrained_model_name == "gpt2":
                lora_kwargs["fan_in_fan_out"] = True
            peft_config = LoraConfig(**lora_kwargs)
        elif peft_type == "prefix-tuning":
            prefix_tuning_kwargs = {"num_virtual_tokens": 20}
            peft_config = PrefixTuningConfig(
                task_type=TaskType.SEQ_CLS, inference_mode=False, **prefix_tuning_kwargs
            )
        elif peft_type == "prompt-tuning":
            peft_config = PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=8)
        elif peft_type == "p-tuning":
            peft_config = PromptEncoderConfig(
                task_type=TaskType.SEQ_CLS, num_virtual_tokens=20, encoder_hidden_size=128
            )
        else:
            raise ValueError(
                f"Unknown `peft_type` '{peft_type}'. "
                "Available options: lora, prefix-tuning, prompt-tuning, p-tuning."
            )
        self._model = get_peft_model(self._model, peft_config)

    def forward(self, x: Dict[str, Tensor], task_id: Optional[str] = None) -> torch.Tensor:
        return self._model(**x)[0]

    def _add_task_params(self, task_id: str) -> None:
        assert not len(self._tasks_params_ids), "Transformer does not work for multiple tasks."


class HuggingFaceSequenceClassificationTransformerWithLora(
    HuggingFaceSequenceClassificationTransformer
):
    """RenateModule which wraps around Hugging Face transformers that uses Lora.

    Args:
        pretrained_model_name: Hugging Face model id.
        num_outputs: Number of outputs.
        loss_fn: The loss function to be optimized during the training.
        r: Attention dimension of Lora.
        alpha: Alpha in Lora.
        dropout: Dropout in Lora.
        bias: Type of bias for Lora. Options: ``"none"``, ``"all"``, and ``"lora_only"``.
        modules_to_save: List of layers to be trained and saved in addition to Lora layers.
        init_lora_weights: Indicate whether to initialize Lora weights.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        r: int = 8,
        alpha: Optional[int] = None,
        dropout: Optional[float] = None,
        bias: str = "none",
        modules_to_save: Optional[List[str]] = None,
        init_lora_weights: bool = True,
    ) -> None:
        constructor_arguments = {
            "r": r,
            "alpha": alpha,
            "dropout": dropout,
            "bias": bias,
            "modules_to_save": modules_to_save,
            "init_lora_weights": init_lora_weights,
        }
        super().__init__(
            pretrained_model_name=pretrained_model_name,
            num_outputs=num_outputs,
            loss_fn=loss_fn,
            constructor_arguments=constructor_arguments.copy(),
        )
        self._model = add_lora(
            model=self._model,
            pretrained_model_name=pretrained_model_name,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            **constructor_arguments,
        )


class HuggingFaceSequenceClassificationTransformerWithPrefixTuning(
    HuggingFaceSequenceClassificationTransformer
):
    """RenateModule which wraps around Hugging Face transformers that uses Prefix Tuning.

    Args:
        pretrained_model_name: Hugging Face model id.
        num_outputs: Number of outputs.
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        loss_fn: The loss function to be optimized during the training.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        encoder_hidden_size: Hidden size of prompt encoder.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
        prefix_projection: Project prefix embeddings.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int,
        num_virtual_tokens: int,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        token_dim: Optional[int] = None,
        encoder_hidden_size: Optional[int] = None,
        num_transformer_submodules: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        prefix_projection: bool = False,
    ) -> None:
        constructor_arguments = {
            "num_virtual_tokens": num_virtual_tokens,
            "token_dim": token_dim,
            "encoder_hidden_size": encoder_hidden_size,
            "num_transformer_submodules": num_transformer_submodules,
            "num_attention_heads": num_attention_heads,
            "num_layers": num_layers,
            "prefix_projection": prefix_projection,
        }

        super().__init__(
            pretrained_model_name=pretrained_model_name,
            num_outputs=num_outputs,
            loss_fn=loss_fn,
            constructor_arguments=constructor_arguments.copy(),
        )
        self._model = add_prefix_tuning(
            model=self._model,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            **constructor_arguments,
        )


class HuggingFaceSequenceClassificationTransformerWithPromptTuning(
    HuggingFaceSequenceClassificationTransformer
):
    """RenateModule which wraps around Hugging Face transformers that uses PromptTuning.

    Args:
        pretrained_model_name: Hugging Face model id.
        num_outputs: Number of outputs.
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        loss_fn: The loss function to be optimized during the training.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
        prompt_tuning_init: Initialization method for prompt embeddings. Default: ``"RANDOM"``.
        prompt_tuning_init_text: Text used to initialize the prompt embeddings if ``prompt_tuning_init=="TEXT"``.
        tokenizer_name_or_path: Name of tokenizer used to tokenize ``prompt_tuning_init_text``.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int,
        num_virtual_tokens: int,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        token_dim: Optional[int] = None,
        num_transformer_submodules: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        prompt_tuning_init: str = PromptTuningInit.RANDOM,
        prompt_tuning_init_text: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
    ) -> None:
        constructor_arguments = {
            "num_virtual_tokens": num_virtual_tokens,
            "token_dim": token_dim,
            "num_transformer_submodules": num_transformer_submodules,
            "num_attention_heads": num_attention_heads,
            "num_layers": num_layers,
            "prompt_tuning_init": prompt_tuning_init,
            "prompt_tuning_init_text": prompt_tuning_init_text,
            "tokenizer_name_or_path": tokenizer_name_or_path,
        }

        super().__init__(
            pretrained_model_name=pretrained_model_name,
            num_outputs=num_outputs,
            loss_fn=loss_fn,
            constructor_arguments=constructor_arguments.copy(),
        )
        self._model = add_prompt_tuning(
            model=self._model,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            **constructor_arguments,
        )


class HuggingFaceSequenceClassificationTransformerWithPTuning(
    HuggingFaceSequenceClassificationTransformer
):
    """RenateModule which wraps around Hugging Face transformers that uses P-Tuning.

    Args:
        pretrained_model_name: Hugging Face model id.
        num_outputs: Number of outputs.
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        loss_fn: The loss function to be optimized during the training.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
        encoder_reparameterization_type: Reparameterization method for prompt encoder. Options: ``"MLP"`` or ``"LSTM"``.
        encoder_hidden_size: Prompt encoder hidden size.
        encoder_num_layers: Number of layers of the prompt encoder.
        encoder_dropout: Prompt encoder dropout.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int,
        num_virtual_tokens: int,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        token_dim: Optional[int] = None,
        num_transformer_submodules: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        encoder_reparameterization_type: str = PromptEncoderReparameterizationType.MLP,
        encoder_hidden_size: Optional[int] = None,
        encoder_num_layers: int = 2,
        encoder_dropout: float = 0.0,
    ) -> None:
        constructor_arguments = {
            "num_virtual_tokens": num_virtual_tokens,
            "token_dim": token_dim,
            "num_transformer_submodules": num_transformer_submodules,
            "num_attention_heads": num_attention_heads,
            "num_layers": num_layers,
            "encoder_reparameterization_type": encoder_reparameterization_type,
            "encoder_hidden_size": encoder_hidden_size,
            "encoder_num_layers": encoder_num_layers,
            "encoder_dropout": encoder_dropout,
        }

        super().__init__(
            pretrained_model_name=pretrained_model_name,
            num_outputs=num_outputs,
            loss_fn=loss_fn,
            constructor_arguments=constructor_arguments.copy(),
        )
        self._model = add_p_tuning(
            model=self._model,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            **constructor_arguments,
        )


def add_lora(
    model: PreTrainedModel,
    pretrained_model_name: str,
    alpha: int,
    dropout: float,
    task_type: Optional[str] = None,
    inference_mode: bool = False,
    r: int = 8,
    bias: str = "none",
    modules_to_save: Optional[List[str]] = None,
    init_lora_weights: bool = True,
) -> PeftModel:
    """Manipulates the given model to efficiently train it using LoRa.

    Args:
        model: Hugging Face pretrained model
        pretrained_model_name: Hugging Face model id.
        alpha: Alpha in Lora.
        dropout: Dropout in Lora.
        task_type: NLP task type.
        inference_mode: whether to load in inference mode.
        r: Attention dimension of Lora.
        bias: Type of bias for Lora. Options: ``"none"``, ``"all"``, and ``"lora_only"``.
        modules_to_save: List of layers to be trained and saved in addition to Lora layers.
        init_lora_weights: Indicate whether to initialize Lora weights.
    """
    lora_kwargs = {}
    if pretrained_model_name == "gpt2":
        lora_kwargs["fan_in_fan_out"] = True
    return get_peft_model(
        model,
        LoraConfig(
            task_type=task_type,
            inference_mode=inference_mode,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            modules_to_save=modules_to_save,
            init_lora_weights=init_lora_weights,
            **lora_kwargs,
        ),
    )


def add_prefix_tuning(
    model: PreTrainedModel,
    task_type: str,
    num_virtual_tokens: int,
    token_dim: Optional[int] = None,
    encoder_hidden_size: Optional[int] = None,
    inference_mode: bool = False,
    num_transformer_submodules: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    num_layers: Optional[int] = None,
    prefix_projection: bool = False,
) -> PeftModel:
    """Manipulates the given model to efficiently train it using Prefix Tuning.

    Args:
        model: Hugging Face pretrained model
        task_type: NLP task type
        inference_mode: whether to load in inference mode
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
        encoder_hidden_size: Hidden size of prompt encoder.
        prefix_projection: Project prefix embeddings.
    """
    return get_peft_model(
        model,
        PrefixTuningConfig(
            task_type=task_type,
            inference_mode=inference_mode,
            num_virtual_tokens=num_virtual_tokens,
            token_dim=token_dim,
            num_transformer_submodules=num_transformer_submodules,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            encoder_hidden_size=encoder_hidden_size,
            prefix_projection=prefix_projection,
        ),
    )


def add_prompt_tuning(
    model: PreTrainedModel,
    task_type: str,
    num_virtual_tokens: int,
    token_dim: Optional[int] = None,
    inference_mode: bool = False,
    num_transformer_submodules: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    num_layers: Optional[int] = None,
    prompt_tuning_init: str = PromptTuningInit.RANDOM,
    prompt_tuning_init_text: Optional[str] = None,
    tokenizer_name_or_path: Optional[str] = None,
) -> PeftModel:
    """Manipulates the given model to efficiently train it using Prompt Tuning.

    Args:
        model: Hugging Face pretrained model
        task_type: NLP task type
        inference_mode: whether to load in inference mode
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
        prompt_tuning_init: Initialization method for prompt embeddings. Default: ``"RANDOM"``.
        prompt_tuning_init_text: Text used to initialize the prompt embeddings if ``prompt_tuning_init=="TEXT"``.
        tokenizer_name_or_path: Name of tokenizer used to tokenize ``prompt_tuning_init_text``.
    """
    return get_peft_model(
        model,
        PromptTuningConfig(
            task_type=task_type,
            inference_mode=inference_mode,
            num_virtual_tokens=num_virtual_tokens,
            token_dim=token_dim,
            num_transformer_submodules=num_transformer_submodules,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            prompt_tuning_init=prompt_tuning_init,
            prompt_tuning_init_text=prompt_tuning_init_text,
            tokenizer_name_or_path=tokenizer_name_or_path,
        ),
    )


def add_p_tuning(
    model: PreTrainedModel,
    task_type: str,
    num_virtual_tokens: int,
    token_dim: Optional[int] = None,
    inference_mode: bool = False,
    num_transformer_submodules: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    num_layers: Optional[int] = None,
    encoder_reparameterization_type: str = PromptEncoderReparameterizationType.MLP,
    encoder_hidden_size: Optional[int] = None,
    encoder_num_layers: int = 2,
    encoder_dropout: float = 0.0,
) -> PeftModel:
    """Manipulates the given model to efficiently train it using Prompt Tuning.

    Args:
        model: Hugging Face pretrained model
        task_type: NLP task type
        inference_mode: whether to load in inference mode
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
        encoder_reparameterization_type: Reparameterization method for prompt encoder.
        encoder_hidden_size: Prompt encoder hidden size.
        encoder_num_layers: Number of layers of the prompt encoder.
        encoder_dropout: Prompt encoder dropout.
    """
    return get_peft_model(
        model,
        PromptEncoderConfig(
            task_type=task_type,
            inference_mode=inference_mode,
            num_virtual_tokens=num_virtual_tokens,
            token_dim=token_dim,
            num_transformer_submodules=num_transformer_submodules,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            encoder_reparameterization_type=encoder_reparameterization_type,
            encoder_hidden_size=encoder_hidden_size,
            encoder_num_layers=encoder_num_layers,
            encoder_dropout=encoder_dropout,
        ),
    )
