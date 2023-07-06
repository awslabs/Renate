# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Any

import torch
from torch import Tensor
from transformers import (
    AutoModelForQuestionAnswering,
    T5ForConditionalGeneration,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

from peft import (
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


class HuggingFaceTransformer(RenateModule):
    """
    Base RenateModule which wraps around Hugging Face transformers.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        constructor_arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        if constructor_arguments is None:
            constructor_arguments = {}
        constructor_arguments.update(
            {
                "pretrained_model_name": pretrained_model_name,
            }
        )

        super().__init__(
            constructor_arguments={
                "pretrained_model_name": pretrained_model_name,
            },
        )

    def forward(self, x: Dict[str, Tensor], task_id: Optional[str] = None) -> torch.Tensor:
        x.pop("token_type_ids", None)
        return self._model(**x, use_cache=False)[0]

    def _add_task_params(self, task_id: str) -> None:
        assert not len(self._tasks_params_ids), "Transformer does not work for multiple tasks."


class HuggingFaceSequenceClassificationTransformer(HuggingFaceTransformer):
    """SequenceClassification RenateModule which wraps around Hugging Face transformers.

    Args:
        pretrained_model_name: Hugging Face model id.
        num_outputs: Number of outputs.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int,
        constructor_arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        if constructor_arguments is None:
            constructor_arguments = {}
        constructor_arguments.update(
            {
                "num_outputs": num_outputs,
            }
        )
        super().__init__(
            pretrained_model_name=pretrained_model_name, constructor_arguments=constructor_arguments
        )
        self._model = from_pretrained(
            pretrained_model_name, num_labels=num_outputs, return_dict=False
        )


class HuggingFaceQuestionAnsweringTransformer(HuggingFaceTransformer):
    """RenateModule which wraps around Hugging Face transformers.

    Args:
        pretrained_model_name: Hugging Face model id.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        constructor_arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            pretrained_model_name=pretrained_model_name, constructor_arguments=constructor_arguments
        )
        self._model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model_name, return_dict=False
        )


class HuggingFaceLanguageModelingTransformer(HuggingFaceTransformer):
    def __init__(
        self,
        pretrained_model_name: str,
        constructor_arguments: Optional[Dict[str, Any]] = None,
        causal: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        enable_activation_checkpointing: bool = False,
    ) -> None:
        super().__init__(pretrained_model_name, constructor_arguments)
        modelcls = AutoModelForCausalLM if causal else AutoModelForMaskedLM
        self._model = modelcls.from_pretrained(
            pretrained_model_name,
            return_dict=False,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True,
            # device_map={"": 0}
        )
        if enable_activation_checkpointing and self._model.supports_gradient_checkpointing:
            self._model.gradient_checkpointing_enable()
            if hasattr(self._model, "enable_input_require_grads"):
                self._model.enable_input_require_grads()


class HuggingFaceSequenceClassificationTransformerWithLora(
    HuggingFaceSequenceClassificationTransformer
):
    """RenateModule which wraps around Hugging Face transformers that uses Lora.

    Args:
        pretrained_model_name: Hugging Face model id.
        num_outputs: Number of outputs.
        alpha: Alpha in Lora.
        dropout: Dropout in Lora.
        r: Attention dimension of Lora.
        bias: Type of bias for Lora. Options: ``"none"``, ``"all"``, and ``"lora_only"``.
        modules_to_save: List of layers to be trained and saved in addition to Lora layers.
        init_lora_weights: Indicate whether to initialize Lora weights.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int,
        alpha: int,
        dropout: float,
        r: int = 8,
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
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        encoder_hidden_size: Hidden size of prompt encoder.
        num_transformer_submodules (`int`): The number of transformer submodules in the base
            transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
        prefix_projection: Project prefix embeddings.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int,
        num_virtual_tokens: int,
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
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base
            transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
        prompt_tuning_init: Initialization method for prompt embeddings. Default: ``"RANDOM"``.
        prompt_tuning_init_text: Text used to initialize the prompt embeddings if
            ``prompt_tuning_init=="TEXT"``.
        tokenizer_name_or_path: Name of tokenizer used to tokenize ``prompt_tuning_init_text``.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int,
        num_virtual_tokens: int,
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
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base
            transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
        encoder_reparameterization_type: Reparameterization method for prompt encoder.
            Options: ``"MLP"`` or ``"LSTM"``.
        encoder_hidden_size: Prompt encoder hidden size.
        encoder_num_layers: Number of layers of the prompt encoder.
        encoder_dropout: Prompt encoder dropout.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int,
        num_virtual_tokens: int,
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
        num_transformer_submodules (`int`): The number of transformer submodules in the base
            transformer model.
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
        num_transformer_submodules (`int`): The number of transformer submodules in the base
            transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
        prompt_tuning_init: Initialization method for prompt embeddings. Default: ``"RANDOM"``.
        prompt_tuning_init_text: Text used to initialize the prompt embeddings if
            ``prompt_tuning_init=="TEXT"``.
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
        num_transformer_submodules (`int`): The number of transformer submodules in the base
            transformer model.
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
