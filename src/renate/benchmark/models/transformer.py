# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch import Tensor
from transformers import AutoModelForSequenceClassification

from renate.models import RenateModule


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
        peft_type: Optional[str] = None,
    ) -> None:
        super().__init__(
            constructor_arguments={
                "pretrained_model_name": pretrained_model_name,
                "num_outputs": num_outputs,
                "peft_type": peft_type,
            },
            loss_fn=loss_fn,
        )
        # if pretrained_model_name in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
        #    auto_class = AutoModelForConditionalGeneration
        self._model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, num_labels=num_outputs, return_dict=False
        )
        if peft_type:
            self.use_peft(pretrained_model_name, peft_type)

    def use_peft(self, pretrained_model_name: str, peft_type: str) -> None:
        if peft_type == "lora":
            lora_kwargs = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.1}
            if pretrained_model_name == "gpt2":
                lora_kwargs["fan_in_fan_out"] = True
            peft_config = LoraConfig(**lora_kwargs)
        else:
            raise ValueError(f"Unknown `peft_type` '{peft_type}'.")  # TODO: list types
        self._model = get_peft_model(self._model, peft_config)

    def forward(self, x: Dict[str, Tensor], task_id: Optional[str] = None) -> torch.Tensor:
        return self._model(**x)[0]

    def _add_task_params(self, task_id: str) -> None:
        assert not len(self._tasks_params_ids), "Transformer does not work for multiple tasks."
