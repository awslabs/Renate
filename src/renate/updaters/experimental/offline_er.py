# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchmetrics
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from renate import defaults
from renate.models import RenateModule
from renate.types import NestedTensors
from renate.updaters.learner import ReplayLearner
from renate.updaters.model_updater import SingleTrainingLoopUpdater
from renate.utils.pytorch import move_tensors_to_device


class OfflineExperienceReplayLearner(ReplayLearner):
    """Experience Replay in the offline version.

    The model will be trained on weighted mixture of losses computed on the new data and a replay
    buffer. In contrast to the online version, the buffer will only be updated after training has
    terminated.

    Args:
        memory_size: The maximum size of the memory.
        memory_batch_size: Size of batches sampled from the memory. The memory batch will be
            appended to the batch sampled from the current dataset, leading to an effective batch
            size of `memory_batch_size + batch_size`.
        loss_weight_new_data: The training loss will be a convex combination of the loss on the new
            data and the loss on the memory data. If a float (needs to be in [0, 1]) is given here,
            it will be used as the weight for the new data. If `None`, the weight will be set
            dynamically to `N_t / sum([N_1, ..., N_t])`, where `N_i` denotes the size of task/chunk
            `i` and the current task is `t`.
        buffer_transform: The transformation to be applied to the memory buffer data samples.
        buffer_target_transform: The target transformation to be applied to the memory buffer target
            samples.
    """

    def __init__(self, loss_weight_new_data: Optional[float] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if loss_weight_new_data is not None and not (0.0 <= loss_weight_new_data <= 1.0):
            raise ValueError(
                "Value of loss_weight_new_data needs to be between 0 and 1,"
                f"got {loss_weight_new_data}."
            )
        self._loss_weight_new_data = loss_weight_new_data
        self._num_points_previous_tasks: int = 0

    def _create_metrics_collections(
        self, logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None
    ) -> None:
        super()._create_metrics_collections(logged_metrics)
        self._loss_collections["train_losses"]["memory_loss"] = torchmetrics.MeanMetric()

    def on_model_update_start(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_dataset_collate_fn: Optional[Callable] = None,
        val_dataset_collate_fn: Optional[Callable] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """Called before a model update starts."""
        super().on_model_update_start(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataset_collate_fn=train_dataset_collate_fn,
            val_dataset_collate_fn=val_dataset_collate_fn,
            task_id=task_id,
        )
        self._num_points_current_task = len(train_dataset)

    def train_dataloader(self) -> DataLoader:
        train_loader = super().train_dataloader()
        loaders = {"current_task": train_loader}
        if len(self._memory_buffer) > self._memory_batch_size:
            loaders["memory"] = DataLoader(
                dataset=self._memory_buffer,
                batch_size=self._memory_batch_size,
                drop_last=True,
                shuffle=True,
                generator=self._rng,
                pin_memory=True,
                collate_fn=self._train_collate_fn,
            )
        return CombinedLoader(loaders, mode="max_size_cycle")

    def on_model_update_end(self) -> None:
        """Called right before a model update terminates."""
        self._memory_buffer.update(self._train_dataset)
        self._num_points_previous_tasks += self._num_points_current_task
        self._num_points_current_task = -1

    def training_step(
        self, batch: Dict[str, Tuple[NestedTensors, torch.Tensor]], batch_idx: int
    ) -> STEP_OUTPUT:
        """PyTorch Lightning function to return the training loss."""
        if self._loss_weight_new_data is None:
            alpha = self._num_points_current_task / (
                self._num_points_current_task + self._num_points_previous_tasks
            )
        else:
            alpha = self._loss_weight_new_data
        inputs, targets = batch["current_task"]
        device = inputs.device
        batch_size_current = inputs.shape[0]
        batch_size_mem = 0
        if "memory" in batch:
            (inputs_mem, targets_mem), _ = batch["memory"]
            batch_size_mem = inputs_mem.shape[0]
            inputs = torch.cat((inputs, inputs_mem), 0)
            targets = torch.cat((targets, targets_mem), 0)
        outputs = self(inputs)
        loss = self._loss_fn(outputs, targets)
        if "memory" in batch:
            weights = torch.Tensor(
                [
                    [alpha for _ in range(batch_size_current)]
                    + [(1 - alpha) for _ in range(batch_size_mem)]
                ]
            )
            self._loss_collections["train_losses"]["memory_loss"](loss[batch_size_current:].mean())
            self._loss_collections["train_losses"]["base_loss"](loss[:batch_size_current].mean())
            weights = move_tensors_to_device(weights, device=device)
            loss = weights / weights.mean() * loss
        else:
            self._loss_collections["train_losses"]["base_loss"](loss[:batch_size_current].mean())
        loss = loss.mean()
        self._update_metrics(outputs, targets, "train")
        return {"loss": loss}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["num_points_previous_tasks"] = self._num_points_previous_tasks

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        self._num_points_previous_tasks = checkpoint["num_points_previous_tasks"]


class OfflineExperienceReplayModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        memory_batch_size: int = defaults.BATCH_SIZE,
        loss_weight_new_data: Optional[float] = None,
        learning_rate_scheduler: Optional[partial] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
        batch_size: int = defaults.BATCH_SIZE,
        input_state_folder: Optional[str] = None,
        output_state_folder: Optional[str] = None,
        max_epochs: int = defaults.MAX_EPOCHS,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
        metric: Optional[str] = None,
        mode: defaults.SUPPORTED_TUNING_MODE_TYPE = "min",
        logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        early_stopping_enabled: bool = False,
        logger: Logger = defaults.LOGGER(**defaults.LOGGER_KWARGS),
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        strategy: str = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        seed: int = defaults.SEED,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
    ):
        learner_kwargs = {
            "memory_size": memory_size,
            "memory_batch_size": memory_batch_size,
            "loss_weight_new_data": loss_weight_new_data,
            "batch_size": batch_size,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=OfflineExperienceReplayLearner,
            learner_kwargs=learner_kwargs,
            input_state_folder=input_state_folder,
            output_state_folder=output_state_folder,
            max_epochs=max_epochs,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_interval=learning_rate_scheduler_interval,
            train_transform=train_transform,
            train_target_transform=train_target_transform,
            test_transform=test_transform,
            test_target_transform=test_target_transform,
            buffer_transform=buffer_transform,
            buffer_target_transform=buffer_target_transform,
            metric=metric,
            mode=mode,
            logged_metrics=logged_metrics,
            early_stopping_enabled=early_stopping_enabled,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            deterministic_trainer=deterministic_trainer,
        )
