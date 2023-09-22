# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import logging
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from renate import defaults
from renate.benchmark.models.base import RenateBenchmarkingModule
from renate.benchmark.models.transformer import HuggingFaceSequenceClassificationTransformer
from renate.benchmark.models.vision_transformer import VisionTransformer
from renate.models.prediction_strategies import PredictionStrategy


logger = logging.getLogger(__name__)


class PromptPool(nn.Module):
    """Implements the prompt pool for L2P

    Args:
        pool_size: Total size of the prompt pool.
        pool_selection_size: Number of prompts to select from the pool.
        prompt_size: Number of tokens each prompt is equivalent to.
        prompt_key_dim: Dimensions of the prompt key used to compute similarity. It has to be same
            to the dimensions of `x` in forward.
        embedding_dim: Output dimension of the token/patch embedding layer
        train_prompt_keys: Whether to train the prompt keys. Currently unused.
        similarity_fn: Similarity function between input features and prompt keys
        per_batch_prompt: Flag to use the same prompts for all elements in the batch
    """

    def __init__(
        self,
        pool_size: int = 10,
        pool_selection_size: int = 5,
        prompt_size: int = 5,
        prompt_key_dim: int = 768,
        embedding_dim: int = 768,
        train_prompt_keys: bool = True,
        similarity_fn: Union[Callable, str] = "cosine",
        per_batch_prompt: bool = True,
    ):
        super().__init__()
        self._M = pool_size  ## total pool size
        self._N = pool_selection_size  ## number of prompts selected per input
        self._Lp = prompt_size  ## each prompt is equal to how many tokens
        self._d = embedding_dim  ##
        self._pd = prompt_key_dim
        self._per_batch_prompt = per_batch_prompt

        self._parse_similarity_fn(similarity_fn)
        self.train_prompt_keys = train_prompt_keys  ## This is unused for now
        self.prompt_pool = nn.Parameter(torch.empty((self._M, self._Lp, self._d)).uniform_(-1, 1))
        self.prompt_keys = nn.Parameter(torch.empty((self._M, self._pd)).uniform_(-1, 1))

        self.key_hist = torch.zeros((self._M,), dtype=torch.float32)

    def _parse_similarity_fn(self, similarity_fn: Union[Callable, str]) -> None:
        if callable(similarity_fn):
            self.similarity_fn = similarity_fn
        elif not isinstance(similarity_fn, str):
            raise ValueError(
                "similarity_fn has to be a callable or a string representing similarity metric. "
                "But got {similarity_fn}"
            )
        elif similarity_fn == "cosine":
            normalization_fn = functools.partial(torch.nn.functional.normalize, p=2)
            self.similarity_fn = lambda x, y: normalization_fn(x).matmul(normalization_fn(y).t())
        else:
            raise ValueError(
                f"Currently only cosine similarity is supported, but got {similarity_fn}"
            )

    def forward(self, x: torch.Tensor, manual_prompt_indices: Optional[torch.LongTensor] = None):
        """
        Args:
            x: Image features extracted. It can be [CLS] token or something else of
                dimension B x self.pd..
            manual_prompt_indices: Indices to manually select prompts from pool, instead of
                selecting from
        """
        if manual_prompt_indices is None:
            similarity_matrix = self.similarity_fn(x, self.prompt_keys)
            _, idx = torch.topk(similarity_matrix, k=self._N, dim=1)
            if self._per_batch_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                if prompt_id.shape[0] < self._M:
                    ## The logic for this is taken from the public l2p implementation.
                    temp_pid = torch.full((self._M,), idx.min(), device=prompt_id.device)
                    temp_pid[: prompt_id.shape[0]] = prompt_id
                    prompt_id = temp_pid

                    temp_idc = torch.zeros((self._M,), device=id_counts.device)
                    temp_idc[: id_counts.shape[0]] = id_counts
                    id_counts = temp_idc

                    _, major_idx = torch.topk(id_counts, k=self._N)
                    idx = prompt_id[major_idx].expand(x.shape[0], -1)  # B, top_k
            loss_value = similarity_matrix[:, idx].sum() / (x.shape[0] * x.shape[0])
        else:
            idx = manual_prompt_indices  # should be of size B, top_k
            loss_value = torch.tensor(0.0, device=x.device)

        selected_prompts = self.prompt_pool[idx].flatten(1, 2)
        return selected_prompts, loss_value


class PromptedTransformer(nn.Module):
    """This generic module is the basic prompted transformer. It takes in a model string and creates
    the appropriate transformer (ViT or Text transformer). If no prompts are provided in the forward
    call, image/text features are returned. If a prompt is provided, it is concatenated to the
    embedding layer output and the resultant features are returned.

    Args:
        pretrained_model_name_or_path: A string that denotes which pretrained model from the HF hub
            to use.
        num_outputs: Size of the output.
        prediction_strategy: Continual learning strategies may alter the prediction at train or test
            time.
        add_icarl_class_means: If ``True``, additional parameters used only by the
            ``ICaRLModelUpdater`` are added. Only required when using that updater.
    """

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
    ) -> None:
        super().__init__()
        if "vit" in pretrained_model_name_or_path:
            self.transformer = VisionTransformer(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                image_size=image_size,
                patch_size=patch_size,
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                num_outputs=num_outputs,
                prediction_strategy=prediction_strategy,
                add_icarl_class_means=add_icarl_class_means,
            )
            self.is_text_transformer = False
        else:
            self.transformer = HuggingFaceSequenceClassificationTransformer(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                num_outputs=num_outputs,
                prediction_strategy=prediction_strategy,
                add_icarl_class_means=add_icarl_class_means,
            )
            for named_param, value in self.transformer.named_parameters():
                if value.shape[0] == self.transformer._backbone.config.vocab_size:
                    self.word_embeddings = self.transformer.get_submodule(
                        named_param.replace(".weight", "")
                    )
                    break

            self.is_text_transformer = True

        self.transformer._tasks_params.clear()
        self.transformer.eval()
        for p in self.transformer.parameters():
            p.requires_grad_(False)

    def forward(
        self, x: torch.Tensor, prompt: Optional[torch.Tensor] = None, cls_feat: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: Input torch tensor.
            prompt: Prompt tensor. Defaults to None.
            cls_feat: Whether to extract [CLS] token or to return full feature tensor.
                Ignored for text transformer. Defaults to True.
        """
        if prompt is None:
            return (
                self.transformer.get_features(x)
                if self.is_text_transformer
                else self.transformer.get_features(x, cls_feat=cls_feat)
            )
            # text transformers dont support cls_feat.
        elif self.is_text_transformer:
            # The implicit assumption here is that x for text transformers is the input_ids.
            # This simplified forward pass has 4 steps:
            # 1. Get prompts
            # 2. Get embeddings from inputs.
            # 3. Concat prompt and inputs
            # 4. Forward prop inputs_embeds to get the features.
            inputs_embeds = self.word_embeddings(x["input_ids"])
            if prompt.size(0) != inputs_embeds.size(0):
                prompt = prompt.unsqueeze(0).expand(
                    inputs_embeds.size(0), -1, -1
                )  # Expand one prompt to batch size
            inputs_embeds = torch.cat((prompt, inputs_embeds), dim=1)
            return self.transformer.get_features({"inputs_embeds": inputs_embeds})
        else:
            patch_embeddings = self.transformer.get_submodule("_backbone.embeddings")(x)
            if prompt.size(0) != x.size(0):
                prompt = prompt.unsqueeze(0).expand(
                    x.size(0), -1, -1
                )  # Expand one prompt to batch size# Expand one prompt to batch size
            input_concat_prompt = torch.cat([patch_embeddings, prompt], dim=1)

            encoded_features = self.transformer.get_submodule("_backbone.encoder")(
                input_concat_prompt, return_dict=False
            )[0]
            encoded_features = self.transformer.get_submodule("_backbone.layernorm")(
                encoded_features
            )
            return encoded_features[:, 0, :] if cls_feat else encoded_features


class LearningToPromptTransformer(RenateBenchmarkingModule):
    """
    Implements the vision transformer with prompt pool described in
    Wang, Zifeng, et al. "Learning to prompt for continual learning." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

    Args:
        pretrained_model_name_or_path: A string that denotes which pretrained model from the HF hub
            to use. If provided, it overrides other arguments about architecture.
        image_size: Size of the input image.
        patch_size: Size of the patches.
        num_layers: Number of Encoder layers.
        num_heads: Number of Attention heads.
        hidden_dim: Size of the Encoder's hidden state.
        mlp_dim: Size of the intermediate Multi-layer Perceptron in the Encoder.
        dropout: Dropout probability.
        attention_dropout: Dropout probability for the attention in the Multi-head Attention layer.
        num_outputs: Size of the output.
        prediction_strategy: Continual learning strategies may alter the prediction at train or test
            time.
        add_icarl_class_means: If ``True``, additional parameters used only by the
            ``ICaRLModelUpdater`` are added. Only required when using that updater.
        pool_size: Total size of the prompt pool.
        pool_selection_size: Number of prompts to select from the pool.
        prompt_size: Number of tokens each prompt is equivalent to.
        prompt_key_dim: Dimensions of the prompt key used to compute similarity. It has to be same
            to the dimensions of `x` in forward.
        train_prompt_keys: Whether to train the prompt keys. Currently unused.
        similarity_fn: Similarity function between input features and prompt keys.
        per_batch_prompt: Flag to use the same prompts for all elements in the batch.
        prompt_embedding_features: Image feature type used to compute the similarity to prompt keys.
        patch_pooler: Features to feed the classifier.
    """

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
        pool_size: int = 10,
        pool_selection_size: int = 5,
        prompt_size: int = 5,
        prompt_key_dim: int = 768,
        train_prompt_keys: bool = True,
        similarity_fn: Union[Callable, str] = "cosine",
        per_batch_prompt: bool = True,
        prompt_embedding_features: str = "cls",
        patch_pooler: str = "prompt_mean",
    ) -> None:
        transformer = PromptedTransformer(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_outputs=num_outputs,
            add_icarl_class_means=add_icarl_class_means,
            prediction_strategy=prediction_strategy,
        )
        prompter = PromptPool(
            embedding_dim=transformer.transformer._embedding_size,
            pool_size=pool_size,
            pool_selection_size=pool_selection_size,
            prompt_size=prompt_size,
            prompt_key_dim=prompt_key_dim,
            train_prompt_keys=train_prompt_keys,
            similarity_fn=similarity_fn,
            per_batch_prompt=per_batch_prompt,
        )

        super().__init__(
            embedding_size=transformer.transformer._embedding_size,
            num_outputs=num_outputs,
            constructor_arguments=dict(
                **transformer.transformer._constructor_arguments,
                pool_size=pool_size,
                pool_selection_size=pool_selection_size,
                prompt_size=prompt_size,
                prompt_key_dim=prompt_key_dim,
                train_prompt_keys=train_prompt_keys,
                similarity_fn=similarity_fn,
                per_batch_prompt=per_batch_prompt,
            ),
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )

        self._backbone = nn.ModuleDict({"transformer": transformer, "prompter": prompter})
        self._is_text_transformer = transformer.is_text_transformer
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

        for p in self._backbone["prompter"].parameters():
            p.requires_grad = True

        # The backbone's forward is monkey-patched to allow the parent class' forward to work
        # without any manual management.
        self._backbone.forward = self.forward_for_monkey_patching

    def forward_for_monkey_patching(
        self, x: torch.Tensor, task_id: str = defaults.TASK_ID
    ) -> torch.Tensor:
        with torch.no_grad():
            prompt_pool_input = self._backbone["transformer"](x, cls_feat=False)
        if not self._is_text_transformer:
            if self.prompt_embedding_features == "cls":
                # retrieve cls token features. This is used in L2P paper.
                prompt_pool_input = prompt_pool_input[:, 0, :]
            elif self.prompt_embedding_features == "mean":
                # compute mean patch features.
                prompt_pool_input = prompt_pool_input[:, 1:, :].mean(1)
            # Compute the prompts to be stacked
        prompts, prompt_similarity = self._backbone["prompter"](prompt_pool_input)
        self.similarity_score = prompt_similarity
        encoded_features = self._backbone["transformer"](x, prompts, cls_feat=False)
        if self._is_text_transformer:
            return encoded_features
        else:
            if self.patch_pooler == "cls":
                seq_cls_token = encoded_features[:, 0, :]
            elif self.patch_pooler == "mean":
                seq_cls_token = encoded_features[:, 1:, :].mean(1)
            elif self.patch_pooler == "prompt_mean":
                num_prompts = prompts.size(1)
                seq_cls_token = encoded_features[:, -num_prompts:, :].mean(1)
            return seq_cls_token
