# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class TaskEstimator(nn.Module, ABC):
    """An ABC that all task estimator methods inherit.

    They implement two methods `update_task_prototypes` and `infer_task`.
    """

    @abstractmethod
    def update_task_prototypes(self):
        return

    @abstractmethod
    def infer_task(self):
        return


class TaskPrototypes(TaskEstimator):
    """Task identification method proposed in S-Prompts.

    Args:
        task_id: The current update id of the method. Required to deserialize.
        clusters_per_task: Number of clusters to use in K-means.
        embedding_size: Embedding size of the transformer features.
    """

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
    ) -> None:
        # At training.
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
        # At inference.
        if self._training_feat_centroids.numel() > 0:
            features = torch.nn.functional.normalize(features)
            nearest_p_inds = torch.cdist(features, self._training_feat_centroids, p=2).argmin(1)
            return self._training_feat_task_ids[nearest_p_inds]
        else:
            return None
