# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional

import torch
from torch import Tensor
from transformers import AutoModelForTextEncoding

from renate.benchmark.models.base import RenateBenchmarkingModule
from renate.models.prediction_strategies import ICaRLClassificationStrategy, PredictionStrategy
from renate import defaults


class HuggingFaceSequenceClassificationTransformer(RenateBenchmarkingModule):
    """RenateBenchmarkingModule which wraps around Hugging Face transformers.

    Args:
        pretrained_model_name: Hugging Face model id.
        num_outputs: Number of outputs.
        prediction_strategy: Continual learning strategies may alter the prediction at train or test
            time.
        add_icarl_class_means: If ``True``, additional parameters used only by the
            ``ICaRLModelUpdater`` are added. Only required when using that updater.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_outputs: int = 10,
        prediction_strategy: Optional[PredictionStrategy] = None,
        add_icarl_class_means: bool = True,
    ):
        model = AutoModelForTextEncoding.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name
        )
        constructor_args = dict(pretrained_model_name=pretrained_model_name)
        super().__init__(
            embedding_size=model.config.hidden_size,
            num_outputs=num_outputs,
            constructor_arguments=constructor_args,
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )

        self._backbone = model

    def forward(self, x: Dict[str, Tensor], task_id: str = defaults.TASK_ID) -> torch.Tensor:
        out = self.get_backbone(task_id=task_id)(**x, return_dict=True)
        if hasattr(out, "pooler_output"):
            x = out.pooler_output
        else:
            x = out.last_hidden_state[:, 0]  # 0th element is used for classification.
        if isinstance(self._prediction_strategy, ICaRLClassificationStrategy):
            return self._prediction_strategy(x, self.training, class_means=self.class_means)
        else:
            assert (
                self._prediction_strategy is None
            ), f"Unknown prediction strategy of type {type(self._prediction_strategy)}."
        return self.get_predictor(task_id)(x)
