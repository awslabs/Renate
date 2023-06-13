# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, Dataset

from .learner import Learner


class PeftLearner(Learner):
    ## Things to delete.
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        pass

    def save(self, output_state_dir: str) -> None:
        pass

    def load(self, input_state_dir: str) -> None:
        pass

    def on_model_update_start(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_dataset_collate_fn,
        val_dataset_collate_fn,
        task_id: Optional[str] = None,
    ) -> None:
        super().on_model_update_start(train_dataset, val_dataset, task_id)
        self._train_collate_fn = train_dataset_collate_fn
        self._val_collate_fn = val_dataset_collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            generator=self._rng,
            pin_memory=True,
            collate_fn=self._train_collate_fn,
            num_workers=2,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            generator=self._rng,
            pin_memory=True,
            collate_fn=self._val_collate_fn,
            num_workers=2,
        )

    def _create_metrics_collections(
        self, logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None
    ) -> None:
        """Creates all logged metrics."""
        metrics = torchmetrics.MetricCollection(logged_metrics)
        train_metrics = metrics.clone(prefix="train_")
        val_metrics = metrics.clone(prefix="val_")

        train_losses = nn.ModuleDict(
            {
                "loss": torchmetrics.MeanMetric(),
            }
        )
        val_losses = nn.ModuleDict({"loss": torchmetrics.MeanMetric()})

        self._metric_collections = nn.ModuleDict(
            {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
        )
        self._loss_collections = nn.ModuleDict(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
        )
