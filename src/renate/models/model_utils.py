import warnings

import transformers
from peft import LoraConfig, get_peft_model

ALLOWED_MODEL_TYPES = {"distilbert-base-uncased", "bert-large-uncased", "gpt2", "gpt2-xl", "EleutherAI/gpt-j-6B"}
ALLOWED_FINE_TUNE_MODES = {"full", "peft", "lora"}


def make_transformers_model(model_name, enable_gradient_checkpointing=True, load_in_8bit=False, **kwargs):
    """Instantiates a model from HF repository. Doesn't load pretrained weights because
    of hard disk space issues on SageMaker. Change the `from_config` call to `from_pretrained`.
    """

    assert (
        model_name in ALLOWED_MODEL_TYPES
    ), f"model_name asked for is wrongly set as {model_name}. Allowed: {ALLOWED_MODEL_TYPES}"

    model_config = transformers.AutoConfig.from_pretrained(model_name)
    model_config.num_labels = 2
    model_config.return_dict = False

    additional_flags = dict()

    if model_name == "bert-large-uncased":
        additional_flags.update(dict(output_hidden_states=False, output_attentions=False))
    elif model_name in ["gpt2", "gpt2-xl", "gpt-j"]:
        additional_flags.update(
            dict(
                output_hidden_states=False, output_attentions=False, use_cache=False, pad_token_id=0
            )
        )

    additional_flags.update(kwargs)
    ## update flags in model
    for k, v in additional_flags.items():
        setattr(model_config, k, v)

    model_config.load_in_8bit = load_in_8bit
    if load_in_8bit:
        model_config.device_map = "auto"

    transformer_model = transformers.AutoModelForSequenceClassification.from_config(model_config)

    if enable_gradient_checkpointing:
        if transformer_model.supports_gradient_checkpointing:
            transformer_model.gradient_checkpointing_enable()

    return transformer_model


def make_tokenizer(tokenizer_name):
    assert (
        tokenizer_name in ALLOWED_MODEL_TYPES
    ), f"tokenizer_name asked for is wrongly set as {tokenizer_name}. Allowed: {ALLOWED_MODEL_TYPES}"

    return transformers.AutoTokenizer.from_pretrained("bert-base-uncased")


def fine_tuning_mode(model, mode="full", **fine_tune_config):
    assert (
        mode in ALLOWED_FINE_TUNE_MODES
    ), f"fine tuning mode set to {mode}. Currently supported fine-tuning modes are {ALLOWED_FINE_TUNE_MODES}"

    if mode in ["peft" or "lora"]:
        warnings.warn("COdebase doesn't allow PEFT customization. Uses standard params for LoRA.")
        peft_config = LoraConfig(
            task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1
        )

        return get_peft_model(model, peft_config)
    return model