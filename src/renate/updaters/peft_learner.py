# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .learner import Learner


class PeftLearner(Learner):
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
        train_dataset_collate_fn=None,
        val_dataset_collate_fn=None,
        task_id: Optional[str] = None,
    ) -> None:
        super().on_model_update_start(train_dataset, val_dataset, task_id)
        self._train_collate_fn = train_dataset_collate_fn
        self._val_collate_fn = val_dataset_collate_fn

    def train_dataloader(self) -> DataLoader:
        shuffle = True
        sampler = (
            DistributedSampler(self._train_dataset) if torch.distributed.is_initialized() else None
        )
        if sampler:
            shuffle = None
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            generator=self._rng,
            pin_memory=True,
            collate_fn=self._train_collate_fn,
            num_workers=4,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        shuffle = False
        sampler = (
            DistributedSampler(self._val_dataset) if torch.distributed.is_initialized() else None
        )
        if sampler:
            shuffle = None
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            generator=self._rng,
            pin_memory=True,
            collate_fn=self._val_collate_fn,
            num_workers=4,
            sampler=sampler,
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
                "base_loss": torchmetrics.MeanMetric(),
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

    def validation_step(self, batch, batch_idx) -> None:
        """PyTorch Lightning function to estimate validation metrics."""
        inputs, targets = batch
        outputs = self(inputs)
        loss = self._loss_fn(outputs, targets)
        self._update_metrics(outputs, targets, "val")
        self._loss_collections["val_losses"]["loss"](loss)


class QAPeft(PeftLearner):
    def validation_step(self, batch, batch_idx) -> None:
        """PyTorch Lightning function to estimate validation metrics."""
        inputs = batch
        outputs = self(inputs)
        targets = None
        loss = self._loss_fn(outputs, targets)
        self._update_metrics(outputs, targets, "val")
        self._loss_collections["val_losses"]["loss"](loss)

    def training_step(self, batch, batch_idx):
        """PyTorch Lightning function to return the training loss."""
        inputs, targets = batch, None
        outputs = self(inputs)
        intermediate_representation = self._model.get_intermediate_representation()
        self._model.reset_intermediate_representation_cache()
        loss = self._loss_fn(outputs, targets).mean()
        self._update_metrics(outputs, targets, "train")
        self._loss_collections["train_losses"]["base_loss"](loss)
        return {
            "loss": loss,
            "outputs": outputs,
            "intermediate_representation": intermediate_representation,
        }
