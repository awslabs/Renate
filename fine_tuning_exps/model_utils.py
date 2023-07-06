from typing import Optional, List
from renate.benchmark.models.transformer import HuggingFaceLanguageModelingTransformer
from peft import prepare_model_for_int8_training
from quantization import quantization
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
import torch


TARGET_MODULES = {
    "tiiuae/falcon-7b": ["query_key_value"],
    "tiiuae/falcon-40b": ["query_key_value"],
}


def make_model(
    pretrained_model_name: str,
    causal: bool,
    quantize: bool,
    alpha: int = 0,
    dropout: float = 0,
    r: int = 8,
    bias: str = "none",
    modules_to_save: Optional[List] = None,
    init_lora_weights: bool = True,
):
    """This supports only int bit training (if that works) and LoRA or full FT."""
    # with quantization(mode="llm.int8" if quantize else None):
    model = HuggingFaceLanguageModelingTransformer(
        pretrained_model_name=pretrained_model_name,
        causal=causal,
        load_in_4bit=quantize,
        enable_activation_checkpointing=True,
    )
    # prepare_model_for_int8_training(model)
    if alpha > 0:
        add_lora(
            model._model,
            pretrained_model_name=pretrained_model_name,
            alpha=alpha,
            bias=bias,
            dropout=dropout,
            r=r,
            target_modules=modules_to_save,
            init_lora_weights=init_lora_weights,
            modules_to_update=TARGET_MODULES.get(pretrained_model_name, None),
            task_type=TaskType.CAUSAL_LM if causal else TaskType.SEQ_2_SEQ_LM,
        )
    return model


def add_lora(
    model: torch.nn.Module,
    pretrained_model_name: str,
    modules_to_update: List[str],
    alpha: int,
    dropout: float,
    task_type: Optional[str] = None,
    inference_mode: bool = False,
    r: int = 8,
    bias: str = "none",
    target_modules: Optional[List[str]] = None,
    init_lora_weights: bool = True,
) -> PeftModel:
    print()
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
            target_modules=modules_to_update,
            modules_to_save=target_modules,
            init_lora_weights=init_lora_weights,
            **lora_kwargs,
        ),
    )
