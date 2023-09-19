# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter

from renate import defaults
from renate.models.prediction_strategies import PredictionStrategy
from renate.types import NestedTensors

from . import PromptedTransformer
from .base import RenateBenchmarkingModule

logger = logging.getLogger(__name__)


class SharedMultipleLinear(nn.ModuleDict):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        share_parameters: bool = True,
        num_updates: int = 0,
    ) -> None:
        self._share_parameters = share_parameters
        self.in_features = in_features
        self.out_features = out_features

        if share_parameters:
            # we only have a single linear.
            layer = nn.Linear(in_features=in_features, out_features=out_features)
            all_layers = {f"{id}": layer for id in range(num_updates)}
        else:
            all_layers = {
                f"{id}": nn.Linear(in_features=in_features, out_features=out_features)
                for id in range(num_updates)
            }
        super().__init__(all_layers)

    def increment_task(self) -> None:
        currlen = len(self)
        if self._share_parameters:
            self[f"{currlen}"] = self[list(self.keys())[0]]
        else:
            self[f"{currlen}"] = nn.Linear(
                in_features=self.in_features, out_features=self.out_features
            )


class PromptPool(nn.Module):
    def __init__(
        self, prompt_size: int = 10, embedding_size: int = 768, current_update_id: int = 0
    ) -> None:
        super().__init__()
        self._M = prompt_size
        self._embedding_size = embedding_size
        self._curr_task = current_update_id

        self._pool = nn.ParameterDict()
        for id in range(self._curr_task):
            self._pool[f"{id}"] = nn.Parameter(
                torch.empty((self._M, self._embedding_size)).uniform_(-1, 1)
            )

        self._pool.requires_grad_(True)

    def forward(self, id: int) -> torch.nn.Parameter:
        return self._pool[f"{id}"]

    def get_params(self, id: int) -> List[torch.nn.Parameter]:
        return [self._pool[f"{id}"]]

    def increment_task(self) -> None:
        self._pool[f"{len(self._pool)}"] = nn.Parameter(
            torch.empty((self._M, self._embedding_size)).uniform_(-1, 1)
        )


class TaskEstimator(nn.Module, ABC):
    @abstractmethod
    def update_task_prototypes(self):
        return

    @abstractmethod
    def infer_task(self):
        return


class TaskPrototypes(TaskEstimator):
    def __init__(self, task_id, clusters_per_task, embedding_size) -> None:
        super().__init__()
        self.register_buffer(
            "_training_feat_centroids",
            torch.empty(task_id * clusters_per_task, embedding_size),
        )
        self.register_buffer(
            "_training_feat_task_ids",
            torch.full(
                (self._training_feat_centroids.size(0),), fill_value=task_id, dtype=torch.long
            ),
        )
        self._clusters_per_task = clusters_per_task
        self._task_id = task_id
        self._embedding_size = embedding_size

    @torch.no_grad()
    def update_task_prototypes(
        self,
        features: Union[torch.Tensor, npt.ArrayLike],
        labels: Union[torch.Tensor, npt.ArrayLike],
    ):
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        # l2 normalize features:
        features = features / np.power(np.einsum("ij, ij -> i", features, features), 0.5)[:, None]

        centroids = torch.from_numpy(
            KMeans(n_clusters=self._clusters_per_task, random_state=0)
            .fit(features)
            .cluster_centers_
        ).to(self._training_feat_centroids.device)

        self._training_feat_centroids = torch.cat(
            [
                self._training_feat_centroids,
                centroids,
            ]
        )
        self._training_feat_task_ids = torch.cat(
            [
                self._training_feat_task_ids,
                torch.full(
                    (centroids.size(0),),
                    fill_value=self._task_id,
                    dtype=torch.int8,
                    device=self._training_feat_task_ids.device,
                ),
            ]
        )

    def infer_task(self, features: torch.Tensor) -> torch.Tensor:
        if self._training_feat_centroids.numel() > 0:
            features = torch.nn.functional.normalize(features)
            nearest_p_inds = torch.cdist(features, self._training_feat_centroids, p=2).argmin(1)
            return self._training_feat_task_ids[nearest_p_inds]
        else:
            return None


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
        per_task_classifier: bool = False,
    ):
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
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )
        super().__init__(
            embedding_size=transformer.transformer._embedding_size,
            num_outputs=num_outputs,
            constructor_arguments=dict(
                **transformer.transformer._constructor_arguments,
                prompt_size=prompt_size,
                task_id=task_id,
                clusters_per_task=clusters_per_task,
                per_task_classifier=per_task_classifier,
            ),
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )
        self._M = prompt_size
        self._task_id = task_id
        self._per_task_classifier = per_task_classifier
        logger.warning(f"Task id is {self._task_id}")

        prompt_pool = PromptPool(
            prompt_size=self._M,
            embedding_size=self._embedding_size,
            current_update_id=self._task_id,
        )

        self._backbone = nn.ModuleDict({"transformer": transformer, "prompt_pool": prompt_pool})
        self._tasks_params = SharedMultipleLinear(
            self._embedding_size, self._num_outputs, True, num_updates=self._task_id + 1
        )
        self._task_id_method = TaskPrototypes(
            task_id=task_id,
            clusters_per_task=clusters_per_task,
            embedding_size=self._embedding_size,
        )
        self._backbone["transformer"].requires_grad_(False)
        self._backbone["prompt_pool"].requires_grad_(True)
        # self._backbone.forward = self.forward_for_monkey_patching

    def increment_task(self) -> None:
        # This cannot be a part of add_task_params as the super.__init__ function calls
        # add_task_params and thus we would be trying parameters to the non-existent
        # self.s_prompts
        key = f"{self._task_id}"
        self._backbone["prompt_pool"].increment_task()
        self._add_task_params(key)
        self._tasks_params.increment_task()

    # def forward_for_monkey_patching(
    #     self, x: Union[torch.Tensor, Dict[str, Any]], task_id: str = None
    # ) -> torch.Tensor:
    #     prompt = None
    #     if self.training:
    #         prompt = self._backbone["prompt_pool"](self._task_id)
    #     else:
    #         task_ids = self._task_id_method.infer_task(self._backbone["transformer"](x))
    #         if task_ids is not None:
    #             prompt = torch.cat([self._backbone["prompt_pool"](i) for i in task_ids])

    #     features = self._backbone["transformer"](x, prompt)
    #     return features

    def update_task_identifier(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self._task_id_method.update_task_prototypes(features, labels)

    def set_extra_state(self, state: Any, decode=True):
        super().set_extra_state(state, decode)
        # once this is set (after loading. increase that by one.)
        self._constructor_arguments["task_id"] = self._task_id + 1

    def get_params(self, task_id: str = defaults.TASK_ID) -> List[Parameter]:
        import warnings

        warnings.warn(f"Length of opreds: {len(self._tasks_params)}, {self._tasks_params.keys()}")
        return self._backbone["prompt_pool"].get_params(self._task_id) + list(
            self.get_predictor(str(self._task_id)).parameters()
        )

    def get_logits(self, x: NestedTensors, task_id: Optional[str] = None) -> torch.Tensor:
        if task_id == "default_task":
            task_id = "0"
        return super().get_logits(x, task_id)

    def forward(self, x: torch.Tensor, task_id: str = defaults.TASK_ID) -> torch.Tensor:
        assert self._prediction_strategy is None, f"SPrompt supports no prediction strategy"

        prompt = None
        if self.training:
            prompt = self._backbone["prompt_pool"](self._task_id)
        else:
            task_ids = self._task_id_method.infer_task(self._backbone["transformer"](x))
            if task_ids is not None:
                prompt = torch.cat([self._backbone["prompt_pool"](i) for i in task_ids])

        features = self._backbone["transformer"](x, prompt)
        if self.training:
            return self.get_predictor(f"{self._task_id}")(features)
        else:
            if task_ids is not None:
                return torch.cat(
                    [
                        self.get_predictor(f"{t}")(feat.unsqueeze(0))
                        for t, feat in zip(task_ids, features)
                    ]
                )
            else:
                return self.get_predictor(f"{self._task_id}")(features)