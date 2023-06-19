# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict

import torch
from torch import Tensor
from transformers import AutoModelForSequenceClassification

from renate.benchmark.models.base import RenateBenchmarkingModule


class AutoModelFeatureExtractorForSequenceClassification(torch.nn.Module):
    def __init__(self, pretrained_model_name: str, num_outputs: int) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, num_outputs=num_outputs
        )
        # There are several options what the classifier is called.

        if not hasattr(self.model, "classifier") or not isinstance(
            self.model.classifier, torch.nn.Linear
        ):
            raise ValueError(
                f"""The chosen transformer type {pretrained_model_name} is not supported 
                for continual learning in Renate. Choose a model type that whose base 
                model class outputs features that are fed directly into a classifier
                and not one that needs additional operations."""
            )
        self.model.classifier = torch.nn.Identity()

    def forward(self, x: Dict[str, Tensor]):
        return self.model(**x).logits


class HuggingFaceSequenceClassificationTransformer(RenateBenchmarkingModule):
    """Module which wraps around Hugging Face transformers for sequence classification.

    Args:
        pretrained_model_name: Hugging Face model id.
        num_outputs: Number of outputs.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int,
    ) -> None:
        super().__init__(
            constructor_arguments={
                "pretrained_model_name": pretrained_model_name,
                "num_outputs": num_outputs,
            },
            num_outputs=num_outputs,
        )
        self._backbone = AutoModelFeatureExtractorForSequenceClassification(
            pretrained_model_name, num_outputs
        )
