# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from renate import defaults
from renate.benchmark.models.base import RenateBenchmarkingModule
from renate.benchmark.models.transformer import HuggingFaceSequenceClassificationTransformer
from renate.benchmark.models.vision_transformer import VisionTransformer
from renate.models.prediction_strategies import ICaRLClassificationStrategy, PredictionStrategy


logger = logging.getLogger(__name__)


class SPromptTransformer(RenateBenchmarkingModule):
    def __init__(
        self,
        pretrained_model_name_or_path="google/vit-base-patch16-224",
        image_size: int = 32,
        patch_size: int = 4,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        num_outputs: int = 10,
        prediction_strategy: Optional[PredictionStrategy] = None,
        add_icarl_class_means: bool = True,
        prompt_size: int = 10,
    ):
        if "vit" in pretrained_model_name_or_path:
            transformer = VisionTransformer(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                image_size=image_size,
                patch_size=patch_size,
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                prediction_strategy=prediction_strategy,
                add_icarl_class_means=add_icarl_class_means,
                num_outputs=num_outputs,
            )
            self._is_text_transformer = False
        else:
            transformer = HuggingFaceSequenceClassificationTransformer(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                prediction_strategy=prediction_strategy,
                add_icarl_class_means=add_icarl_class_means,
                num_outputs=num_outputs,
            )

            self._is_text_transformer = True

        super().__init__(
            embedding_size=transformer._embedding_size,
            num_outputs=num_outputs,
            constructor_arguments=dict(
                **transformer._constructor_arguments,
                prompt_size=prompt_size,
            ),
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )
        self._M = prompt_size
        self._s_prompts = torch.nn.ParameterDict()
        self._backbone = nn.ModuleDict({"transformer": transformer})
        for n, p in self._backbone.named_parameters():
            p.requires_grad = False
        self._s_prompts.requires_grad_(True)
        if self._is_text_transformer:
            ## This is to find the Embedding layer.
            for named_param, value in self._backbone["transformer"].named_parameters():
                if value.shape[0] == self._backbone["transformer"]._backbone.config.vocab_size:
                    self.word_embeddings = self._backbone["transformer"].get_submodule(
                        named_param.replace(".weight", "")
                    )
                    break

    def add_s_prompts(self, task_id: str = defaults.TASK_ID) -> None:
        # This cannot be a part of add_task_params as the super.__init__ function calls
        # add_task_params and thus we would be trying parameters to the non-existent
        # self.s_prompts
        self._s_prompts[f"{len(self._s_prompts)}"] = nn.Parameter(
            torch.empty((self._M, self._embedding_size)).uniform_(-1, 1)
        )
        self._s_prompts.requires_grad_(True)

    def compute_features(
        self, x: Union[torch.Tensor, Dict[str, Any]], task_id: str
    ) -> torch.Tensor:
        # Not that task_id is not yet set. So we access the last inserted prompt.
        if self.training:
            prompt = self._s_prompts[list(self._s_prompts)[-1]]
        else:
            prompt = self._nearest_prompt(x)
        return (
            self._prompt_vit(x, prompt)
            if not self._is_text_transformer
            else self._prompt_text_transformer(x, prompt)
        )

    def _prompt_text_transformer(self, x, prompt):
        inputs_embeds = self.word_embeddings(x["input_ids"])
        if prompt is not None:
            if prompt.size(0) != inputs_embeds.size(0):
                prompt = prompt.unsqueeze(0).expand(
                    inputs_embeds.size(0), -1, -1
                )  # Expand one prompt to batch size
            inputs_embeds = torch.cat((prompt, inputs_embeds), dim=1)
        return self._backbone["transformer"].get_features({"inputs_embeds": inputs_embeds})[:, 0, :]

    def _prompt_vit(self, x, prompt=None):
        patch_embeddings = self._backbone["transformer"].get_submodule("_backbone.embeddings")(x)
        if prompt is not None:
            if prompt.size(0) != x.size(0):
                prompt = prompt.unsqueeze(0).expand(
                    x.size(0), -1, -1
                )  # Expand one prompt to batch size# Expand one prompt to batch size
            input_concat_prompt = torch.cat([patch_embeddings, prompt], dim=1)
        else:
            input_concat_prompt = patch_embeddings
        encoded_features = self._backbone["transformer"].get_submodule("_backbone.encoder")(
            input_concat_prompt, return_dict=False
        )[0]
        encoded_features = self._backbone["transformer"].get_submodule("_backbone.layernorm")(
            encoded_features
        )
        return encoded_features[:, 0, :]

    def forward(self, x: torch.Tensor, task_id: str = defaults.TASK_ID) -> torch.Tensor:
        x = self.compute_features(x, task_id=task_id)
        if isinstance(self._prediction_strategy, ICaRLClassificationStrategy):
            return self._prediction_strategy(x, self.training, class_means=self.class_means)
        else:
            assert (
                self._prediction_strategy is None
            ), f"Unknown prediction strategy of type {type(self._prediction_strategy)}."
        return self.get_predictor(task_id)(x)

    def _nearest_prompt(self, x):
        if hasattr(self, "_training_feat_centroids"):
            feats = self._backbone["transformer"].get_features(x)[:, 0, :]  # BxD
            nearest_p_inds = torch.cdist(feats, self._training_feat_centroids, p=1).argmax(1)
            train_feat_task_ids = self._training_feat_task_ids[nearest_p_inds]  # B x 1
            return torch.cat([self._s_prompts[f"{i}"] for i in train_feat_task_ids])
        else:
            return None

    def maybe_append_task_centroids(self, centroids):
        if not hasattr(self, "_training_feat_centroids"):
            print("Here")
            self.register_buffer("_training_feat_centroids", centroids)
            self.register_buffer(
                "_training_feat_task_ids", len(self._s_prompts) * torch.ones(centroids.size(0))
            )
        else:
            self._training_feat_centroids = torch.cat([self._training_feat_centroids, centroids])
            self._training_feat_task_ids = torch.cat(
                [self._training_feat_task_ids, len(self._s_prompts) * torch.ones(centroids.size(0))]
            )
