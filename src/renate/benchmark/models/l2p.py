# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from renate import defaults
from renate.benchmark.models.base import RenateBenchmarkingModule
from renate.benchmark.models.vision_transformer import (
    VisionTransformer,
    VisionTransformerB16,
    VisionTransformerB32,
)
from renate.models.prediction_strategies import ICaRLClassificationStrategy, PredictionStrategy


class PromptPool(nn.Module):
    """Implements the prompt pool for L2P"""

    def __init__(
        self,
        pool_size: int = 10,
        pool_selection_size: int = 5,
        prompt_size: int = 5,
        prompt_key_dim: int = 768,
        embedding_dim: int = 768,
        train_prompt_keys: bool = True,
        similarity_fn: Union[Callable, str] = "cosine",
    ):
        super().__init__()
        self.M = pool_size  ## total pool size
        self.N = pool_selection_size  ## number of prompts selected per input
        self.Lp = prompt_size  ## each prompt is equal to how many tokens
        self.d = embedding_dim  ##
        self.pd = prompt_key_dim

        self._parse_similarity_fn(similarity_fn)
        self.train_prompt_keys = train_prompt_keys  ## This is unused for now

        self.prompt_pool = nn.Parameter(torch.rand(self.M, self.Lp, self.d))
        self.prompt_keys = nn.Parameter(torch.rand(self.M, self.pd))

    def _parse_similarity_fn(self, similarity_fn: Union[Callable, str]) -> None:
        if callable(similarity_fn):
            self.similarity_fn = similarity_fn
        elif not isinstance(similarity_fn, str):
            raise ValueError(
                "similarity_fn has to be a callable or a string representing similarity metric. "
                "But got {similarity_fn}"
            )
        elif similarity_fn == "cosine":
            self.similarity_fn = lambda x, y: 1 - (
                (x / torch.einsum("ij,ij->i", x, x).unsqueeze(-1)).matmul(
                    (y / torch.einsum("ij,ij->i", y, y).unsqueeze(-1)).t()
                )
            )
        else:
            raise ValueError(
                f"Currently only cosine similarity is supported, but got {similarity_fn}"
            )

    def forward(self, x):
        # Here x refers to the features already extracted. These can be [CLS] token representation
        # or something else altogether. But it has to be of dimension B x self.pd.
        similarity_matrix = self.similarity_fn(x, self.prompt_keys)
        kbest_similarity, kbest_indices = similarity_matrix.topk(k=self.N)
        selected_prompts = self.prompt_pool[kbest_indices, ...]
        selected_prompts = selected_prompts.view(
            selected_prompts.size(0), -1, selected_prompts.size(3)
        )
        return selected_prompts, kbest_similarity


class PromptedVisionTransformer(RenateBenchmarkingModule):
    def __init__(
        self,
        vit: Optional[VisionTransformer] = None,
        prompter: Optional[PromptPool] = None,
        prompt_embedding_features: str = "cls",
        patch_pooler: str = "prompt_mean",
        num_outputs: int = 10,
        prediction_strategy: Optional[PredictionStrategy] = None,
        add_icarl_class_means: bool = True,
    ) -> None:
        if vit is None:
            vit = VisionTransformerB32()
        if prompter is None:
            prompter = PromptPool()
        super().__init__(
            embedding_size=vit._embedding_size,
            num_outputs=num_outputs,
            constructor_arguments={
                "prompt_embedding_features": prompt_embedding_features,
            },
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )

        self._backbone = nn.ModuleDict({"vit": vit, "prompter": prompter})
        self.prompt_embedding_features = prompt_embedding_features
        self.patch_pooler = patch_pooler
        self.similarity_score: Optional[torch.Tensor] = None

        assert self.prompt_embedding_features in [
            "cls",
            "mean",
        ], f"Invalid method to extract prompt embedding features. Got {prompt_embedding_features}"

        assert self.patch_pooler in [
            "cls",
            "mean",
            "prompt_mean",
        ], f"Invalid method to extract prompt embedding features. Got {patch_pooler}"

        for p in self._backbone["vit"].parameters():
            p.requires_grad = False
        for p in self._backbone["prompter"].parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, task_id: str = defaults.TASK_ID) -> torch.Tensor:
        prompt_pool_input = self._backbone["vit"].get_logits(x)
        if self.prompt_embedding_features == "cls":
            # retrieve cls token features. This is used in L2P paper.
            prompt_pool_input = prompt_pool_input[:, 0, :]
        elif self.prompt_embedding_features == "mean":
            # compute mean patch features.
            prompt_pool_input = prompt_pool_input[:, 1:, :].mean(1)

        # Compute the prompts to be stacked
        prompts, prompt_similarity = self._backbone["prompter"](prompt_pool_input)
        # compute patch embeddings
        patch_embeddings = self._backbone["vit"].get_submodule("_backbone.embeddings")(x)
        # concatenate both.
        input_concat_prompt = torch.cat([patch_embeddings, prompts], dim=1)
        ## rest of processing. this code is part of the ViTModel class in HF Transformers.
        encoded_features = self._backbone["vit"].get_submodule("_backbone.encoder")(
            input_concat_prompt, return_dict=False
        )[0]
        encoded_features = self._backbone["vit"].get_submodule("_backbone.layernorm")(
            encoded_features
        )

        ## Save similarity
        self.similarity_score = prompt_similarity.mean(0).sum()

        if self.patch_pooler == "cls":
            seq_cls_token = encoded_features[:, 0, :]
        elif self.patch_pooler == "mean":
            seq_cls_token = encoded_features[:, 1:, :].mean(1)
        elif self.patch_pooler == "prompt_mean":
            num_prompts = prompts.size(1)
            seq_cls_token = encoded_features[:, 1 : num_prompts + 1, :].mean(1)

        if isinstance(self._prediction_strategy, ICaRLClassificationStrategy):
            return self._prediction_strategy(
                seq_cls_token, self.training, class_means=self.class_means
            )
        else:
            assert (
                self._prediction_strategy is None
            ), f"Unknown prediction strategy of type {type(self._prediction_strategy)}."
        return self.get_predictor(task_id)(seq_cls_token)
