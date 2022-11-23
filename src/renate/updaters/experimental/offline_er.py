# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchmetrics
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader, Dataset

from renate import defaults
from renate.models import RenateModule
from renate.updaters.learner import ReplayLearner
from renate.updaters.model_updater import SimpleModelUpdater


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

    def __init__(
        self,
        loss_weight_new_data: Optional[float] = None,
        **kwargs,
    ) -> None:
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
        self, train_dataset: Dataset, val_dataset: Dataset, task_id: Optional[str] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Called before a model update starts."""
        self._num_points_current_task = len(train_dataset)
        train_loader, val_loader = super().on_model_update_start(
            train_dataset, val_dataset, task_id
        )
        loaders = {"current_task": train_loader}
        if len(self._memory_buffer) > self._memory_batch_size:
            loaders["memory"] = DataLoader(
                dataset=self._memory_buffer,
                batch_size=self._memory_batch_size,
                drop_last=True,
                shuffle=True,
                generator=self._rng,
                pin_memory=True,
            )
        return CombinedLoader(loaders, mode="max_size_cycle"), val_loader

    def on_model_update_end(
        self, train_dataset: Dataset, val_dataset: Dataset, task_id: Optional[str] = None
    ) -> RenateModule:
        """Called right before a model update terminates."""
        self._memory_buffer.update(train_dataset)
        self._num_points_previous_tasks += self._num_points_current_task
        self._num_points_current_task = -1
        return super().on_model_update_end(train_dataset, val_dataset, task_id)

    def training_step(self, batch: Dict[str, List[torch.Tensor]], batch_idx: int) -> STEP_OUTPUT:
        """PyTorch Lightning function to return the training loss."""
        if self._loss_weight_new_data is None:
            alpha = self._num_points_current_task / (
                self._num_points_current_task + self._num_points_previous_tasks
            )
        else:
            alpha = self._loss_weight_new_data
        x, y = batch["current_task"]
        outputs = self(x)
        loss = self._model.loss_fn(outputs, y)
        self._loss_collections["train_losses"]["base_loss"](loss)
        self._update_metrics(outputs, y, "train")
        if "memory" in batch:
            x_mem, y_mem = batch["memory"]
            outputs_mem = self(x_mem)
            loss_mem = self._model.loss_fn(outputs_mem, y_mem)
            self._loss_collections["train_losses"]["memory_loss"](loss_mem)
            loss = alpha * loss + (1.0 - alpha) * loss_mem
        return {"loss": loss}

    def state_dict(self, **kwargs) -> Dict[str, Any]:
        """Returns the state of the learner."""
        state_dict = super().state_dict(**kwargs)
        state_dict["loss_weight_new_data"] = self._loss_weight_new_data
        state_dict["num_points_previous_tasks"] = self._num_points_previous_tasks
        return state_dict

    def load_state_dict(self, model: RenateModule, state_dict: Dict[str, Any], **kwargs) -> None:
        """Restores the state of the learner."""
        super().load_state_dict(model, state_dict, **kwargs)
        self._loss_weight_new_data = state_dict["loss_weight_new_data"]
        self._num_points_previous_tasks = state_dict["num_points_previous_tasks"]


class OfflineExperienceReplayModelUpdater(SimpleModelUpdater):
    def __init__(
        self,
        model: RenateModule,
        memory_size: int,
        memory_batch_size: int = defaults.BATCH_SIZE,
        loss_weight_new_data: Optional[float] = None,
        optimizer: defaults.SUPPORTED_OPTIMIZERS_TYPE = defaults.OPTIMIZER,
        learning_rate: float = defaults.LEARNING_RATE,
        learning_rate_scheduler: defaults.SUPPORTED_LEARNING_RATE_SCHEDULERS_TYPE = defaults.LEARNING_RATE_SCHEDULER,
        learning_rate_scheduler_gamma: float = defaults.LEARNING_RATE_SCHEDULER_GAMMA,
        learning_rate_scheduler_step_size: int = defaults.LEARNING_RATE_SCHEDULER_STEP_SIZE,
        momentum: float = defaults.MOMENTUM,
        weight_decay: float = defaults.WEIGHT_DECAY,
        batch_size: int = defaults.BATCH_SIZE,
        current_state_folder: Optional[str] = None,
        next_state_folder: Optional[str] = None,
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
        seed: int = defaults.SEED,
    ):
        learner_kwargs = {
            "memory_size": memory_size,
            "memory_batch_size": memory_batch_size,
            "loss_weight_new_data": loss_weight_new_data,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "learning_rate_scheduler": learning_rate_scheduler,
            "learning_rate_scheduler_gamma": learning_rate_scheduler_gamma,
            "learning_rate_scheduler_step_size": learning_rate_scheduler_step_size,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "seed": seed,
        }
        super().__init__(
            model,
            learner_class=OfflineExperienceReplayLearner,
            learner_kwargs=learner_kwargs,
            current_state_folder=current_state_folder,
            next_state_folder=next_state_folder,
            max_epochs=max_epochs,
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
        )
