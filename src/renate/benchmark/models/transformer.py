# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from transformers import AutoModelForTextEncoding, PreTrainedModel

from renate.benchmark.models.base import RenateBenchmarkingModule
from renate.models.prediction_strategies import PredictionStrategy


class FeatureExtractorTextTransformer(PreTrainedModel):
    """This is a facade class to extract the correct output from the transformer model."""

    def __init__(self, pretrained_model_name_or_path: str):
        model = AutoModelForTextEncoding.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        super().__init__(model.config)
        self._model = model

    def forward(self, x):
        out = self._model(**x, return_dict=True)
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        else:
            return out.last_hidden_state[:, 0]  # 0th element is used for classification.


class HuggingFaceSequenceClassificationTransformer(RenateBenchmarkingModule):
    """RenateBenchmarkingModule which wraps around Hugging Face transformers.

    Args:
        pretrained_model_name_or_path: Hugging Face model id.
        num_outputs: Number of outputs.
        prediction_strategy: Continual learning strategies may alter the prediction at train or test
            time.
        add_icarl_class_means: If ``True``, additional parameters used only by the
            ``ICaRLModelUpdater`` are added. Only required when using that updater.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_outputs: int = 10,
        prediction_strategy: Optional[PredictionStrategy] = None,
        add_icarl_class_means: bool = True,
    ):
        model = FeatureExtractorTextTransformer(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        constructor_args = dict(pretrained_model_name_or_path=pretrained_model_name_or_path)
        super().__init__(
            embedding_size=model.config.hidden_size,
            num_outputs=num_outputs,
            constructor_arguments=constructor_args,
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )

        self._backbone = model

    def get_features(self, *args, **kwargs):
        return self._backbone(*args, **kwargs)
