# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import logging
import random
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

from renate import defaults
from renate.benchmark.models.base import RenateBenchmarkingModule
from renate.benchmark.models.vision_transformer import VisionTransformer
from renate.models.prediction_strategies import ICaRLClassificationStrategy, PredictionStrategy
from renate.utils.deepspeed import convert_to_tensor, recover_object_from_tensor

logger = logging.getLogger(__name__)


# class PromptPool(nn.Module):
#     """Implements the prompt pool for L2P"""

#     def __init__(
#         self,
#         pool_size: int = 10,
#         pool_selection_size: int = 5,
#         prompt_size: int = 5,
#         prompt_key_dim: int = 768,
#         embedding_dim: int = 768,
#         train_prompt_keys: bool = True,
#         similarity_fn: Union[Callable, str] = "cosine",
#     ):
#         super().__init__()
#         self.M = pool_size  ## total pool size
#         self.N = pool_selection_size  ## number of prompts selected per input
#         self.Lp = prompt_size  ## each prompt is equal to how many tokens
#         self.d = embedding_dim  ##
#         self.pd = prompt_key_dim

#         self._parse_similarity_fn(similarity_fn)
#         self.train_prompt_keys = train_prompt_keys  ## This is unused for now

#         self.prompt_pool = nn.Parameter(torch.rand(self.M, self.Lp, self.d))
#         self.prompt_keys = nn.Parameter(torch.rand(self.M, self.pd))

#         self.key_hist = torch.zeros((self.M, ), dtype=torch.float32)

#     def _parse_similarity_fn(self, similarity_fn: Union[Callable, str]) -> None:
#         if callable(similarity_fn):
#             self.similarity_fn = similarity_fn
#         elif not isinstance(similarity_fn, str):
#             raise ValueError(
#                 "similarity_fn has to be a callable or a string representing similarity metric. "
#                 "But got {similarity_fn}"
#             )
#         elif similarity_fn == "cosine":
#             self.similarity_fn = lambda x, y: 1 - (
#                 (x / torch.einsum("ij,ij->i", x, x).unsqueeze(-1)).matmul(
#                     (y / torch.einsum("ij,ij->i", y, y).unsqueeze(-1)).t()
#                 )
#             )
#         else:
#             raise ValueError(
#                 f"Currently only cosine similarity is supported, but got {similarity_fn}"
#             )

#     def forward(self, x):
#         # Here x refers to the features already extracted. These can be [CLS] token representation
#         # or something else altogether. But it has to be of dimension B x self.pd.
#         similarity_matrix = self.similarity_fn(x, self.prompt_keys)

#         kbest_similarity, kbest_indices = similarity_matrix.topk(k=self.N)
#         selected_prompts = self.prompt_pool[kbest_indices, ...]
#         selected_prompts = selected_prompts.view(
#             selected_prompts.size(0), -1, selected_prompts.size(3)
#         )
#         # logging
#         self.key_hist[kbest_indices.to("cpu")] += 1
#         if random.random() > 0.9:
#             logger.info("Histogram of chosen prompts")
#             logger.info(self.key_hist.view(-1))
#         return selected_prompts, kbest_similarity


class PromptPool(nn.Module):
    def __init__(
        self,
        pool_size: int = 10,
        pool_selection_size: int = 5,
        prompt_size: int = 5,
        prompt_key_dim: int = 768,
        embedding_dim: int = 768,
        train_prompt_keys: bool = True,
        similarity_fn: Union[Callable, str] = "cosine",
        embedding_key="clss",
        batchwise_prompt=True,
    ):
        super().__init__()

        self.length = prompt_size
        self.embed_dim = embedding_dim
        self.prompt_pool = True
        self.embedding_key = embedding_key
        prompt_init = "uniform"
        self.prompt_key = True
        self.pool_size = pool_size
        self.top_k = pool_selection_size
        self.batchwise_prompt = batchwise_prompt
        prompt_key_init = "uniform"
        prompt_key = True

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, prompt_size, embedding_dim)
            if prompt_init == "zero":
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == "uniform":
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embedding_dim)
            if prompt_key_init == "zero":
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == "uniform":
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None):
        out = dict()
        if self.prompt_pool:
            # if self.embedding_key == "mean":
            #     x_embed_mean = torch.mean(x_embed, dim=1)
            # elif self.embedding_key == "max":
            #     x_embed_mean = torch.max(x_embed, dim=1)[0]
            # elif self.embedding_key == "mean_max":
            #     x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            # elif self.embedding_key == "cls":
            #     if cls_features is None:
            #         x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
            #     else:
            #         x_embed_mean = cls_features
            # else:
            #     raise NotImplementedError("Not supported way of calculating embedding keys!")
            x_embed_mean = x_embed
            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size

            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat(
                            [
                                prompt_id,
                                torch.full(
                                    (self.pool_size - prompt_id.shape[0],),
                                    torch.min(idx.flatten()),
                                    device=prompt_id.device,
                                ),
                            ]
                        )
                        id_counts = torch.cat(
                            [
                                id_counts,
                                torch.full(
                                    (self.pool_size - id_counts.shape[0],),
                                    0,
                                    device=id_counts.device,
                                ),
                            ]
                        )
                    _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                    major_prompt_id = prompt_id[major_idx]  # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k
            else:
                idx = prompt_mask  # B, top_k

            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(
                batch_size, top_k * length, c
            )  # B, top_k * length, C

            out["prompt_idx"] = idx

            # Debugging, return sim as well
            out["prompt_norm"] = prompt_norm
            out["x_embed_norm"] = x_embed_norm
            out["similarity"] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx]  # B, top_k, C
            out["selected_key"] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out["reduce_sim"] = reduce_sim
        else:
            if self.prompt_init == "zero":
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == "uniform":
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        # out["total_prompt_len"] = batched_prompt.shape[1]
        # out["prompted_embedding"] = torch.cat([batched_prompt, x_embed], dim=1)

        return batched_prompt, reduce_sim


class PromptedVisionTransformer(RenateBenchmarkingModule):
    def __init__(
        self,
        pretrained_model_name_or_path="google/vit-base-patch32-224-in21k",
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
        pool_size: int = 10,
        pool_selection_size: int = 5,
        prompt_size: int = 5,
        prompt_key_dim: int = 768,
        train_prompt_keys: bool = True,
        similarity_fn: Union[Callable, str] = "cosine",
        prompt_embedding_features: str = "cls",
        patch_pooler: str = "prompt_mean",
    ) -> None:
        vit = VisionTransformer(
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
        prompter = PromptPool(
            embedding_dim=vit._embedding_size,
            pool_size=pool_size,
            pool_selection_size=pool_selection_size,
            prompt_size=prompt_size,
            prompt_key_dim=prompt_key_dim,
            train_prompt_keys=train_prompt_keys,
            similarity_fn=similarity_fn,
        )
        super().__init__(
            embedding_size=vit._embedding_size,
            num_outputs=num_outputs,
            constructor_arguments=dict(
                num_outputs=num_outputs,
                prompt_embedding_features=prompt_embedding_features,
                patch_pooler=patch_pooler,
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
                pool_size=pool_size,
                pool_selection_size=pool_selection_size,
                prompt_size=prompt_size,
                prompt_key_dim=prompt_key_dim,
                train_prompt_keys=train_prompt_keys,
                similarity_fn=similarity_fn,
            ),
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
        self._backbone["vit"].eval()
        for p in self._backbone["prompter"].parameters():
            p.requires_grad = True
        self._feature_extractor = copy.deepcopy(self._backbone["vit"])

    def forward(self, x: torch.Tensor, task_id: str = defaults.TASK_ID) -> torch.Tensor:
        with torch.no_grad():
            prompt_pool_input = self._feature_extractor.get_logits(x)
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
        self.similarity_score = prompt_similarity  # .sum().div(x.size(0))

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
