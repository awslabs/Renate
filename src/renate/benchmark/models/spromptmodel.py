# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from renate import defaults
from renate.benchmark.models.base import RenateBenchmarkingModule
from renate.benchmark.models.transformer import HuggingFaceSequenceClassificationTransformer
from renate.benchmark.models.vision_transformer import VisionTransformer
from renate.models.prediction_strategies import PredictionStrategy


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
        task_id: int = 0,
        clusters_per_task: int = 5,
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
                task_id=task_id + 1,  # we store a +1 for the next update step.
                clusters_per_task=clusters_per_task,
            ),
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )
        self._M = prompt_size
        self._task_id = task_id

        self._backbone = nn.ModuleDict({"transformer": transformer})
        self._s_prompts = torch.nn.ParameterDict()

        self.register_buffer(
            "_training_feat_centroids",
            torch.zeros(task_id * clusters_per_task, transformer._embedding_size),
        )
        self.register_buffer(
            "_training_feat_task_ids",
            torch.full((self._training_feat_centroids.size(0),), task_id, dtype=torch.int8),
        )
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

        self._backbone.forward = self.forward_for_monkey_patching

    def add_s_prompts(self) -> None:
        # This cannot be a part of add_task_params as the super.__init__ function calls
        # add_task_params and thus we would be trying parameters to the non-existent
        # self.s_prompts
        self._s_prompts[f"{self._task_id}"] = nn.Parameter(
            torch.empty((self._M, self._embedding_size)).uniform_(-1, 1)
        )
        self._s_prompts.requires_grad_(True)

    def forward_for_monkey_patching(
        self, x: Union[torch.Tensor, Dict[str, Any]], task_id: str = None
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

    def _prompt_vit(self, x, prompt):
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

    def _nearest_prompt(self, x):
        if self._training_feat_centroids.numel() != 0:
            feats = self._backbone["transformer"].get_features(x)  # BxD
            nearest_p_inds = torch.cdist(feats, self._training_feat_centroids, p=1).argmin(1)
            train_feat_task_ids = self._training_feat_task_ids[nearest_p_inds]  # B x 1
            return torch.cat([self._s_prompts[f"{i}"] for i in train_feat_task_ids])
        else:
            return None

    def append_task_centroids(self, centroids):
        self._training_feat_centroids = torch.cat([self._training_feat_centroids, centroids])
        self._training_feat_task_ids = torch.cat(
            [
                self._training_feat_task_ids,
                torch.full(
                    (centroids.size(0),),
                    self._task_id,
                    dtype=torch.int8,
                    device=self._training_feat_task_ids.device,
                ),
            ]
        )
