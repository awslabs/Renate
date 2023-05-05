# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional

import torch
import torch.nn as nn
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
    ) -> None:
        super().__init__(
            constructor_arguments={
                "pretrained_model_name": pretrained_model_name,
                "num_outputs": num_outputs,
            },
            loss_fn=loss_fn,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, num_labels=num_outputs, return_dict=False
        )

    def forward(self, x: Dict[str, Tensor], task_id: Optional[str] = None) -> torch.Tensor:
        return self._model(**x)[0]

    def _add_task_params(self, task_id: str) -> None:
        assert not len(self._tasks_params_ids), "Transformer does not work for multiple tasks."
