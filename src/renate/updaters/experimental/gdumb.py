# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchmetrics
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from renate import defaults
from renate.memory import GreedyClassBalancingBuffer
from renate.models import RenateModule
from renate.types import NestedTensors
from renate.updaters.learner import ReplayLearner
from renate.updaters.model_updater import SingleTrainingLoopUpdater


class GDumbLearner(ReplayLearner):
    """A Learner that implements the GDumb strategy.

    Prabhu, Ameya, Philip HS Torr, and Puneet K. Dokania. "GDumb: A simple
    approach that questions our progress in continual learning." ECCV, 2020.

    It maintains a memory of  previously observed data points and does
    the training after updating the buffer. Note that, the model is
    reinitialized before training on the buffer.

    Args:
        memory_size: The maximum size of the memory.
        buffer_transform: The transform to be applied to the data points in the memory.
        buffer_target_transform: The transform to be applied to the targets in the memory.
        seed: A random seed.
    """

    def __init__(
        self,
        memory_size: int,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
        seed: int = defaults.SEED,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            memory_size=memory_size,
            seed=seed,
            **kwargs,
        )

        self._memory_buffer = GreedyClassBalancingBuffer(
            max_size=memory_size,
            seed=seed,
            transform=buffer_transform,
            target_transform=buffer_target_transform,
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        self._memory_buffer.load_state_dict(checkpoint["memory_buffer"])

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
        self._memory_buffer.update(train_dataset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._memory_buffer,
            batch_size=self._batch_size,
            shuffle=True,
            generator=self._rng,
            pin_memory=True,
            collate_fn=self._train_collate_fn,
        )

    def training_step(
        self,
        batch: Tuple[Tuple[NestedTensors, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """PyTorch Lightning function to return the training loss."""
        batch, _ = batch
        return super().training_step(batch=batch, batch_idx=batch_idx)


class GDumbModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        batch_memory_frac: int = defaults.BATCH_MEMORY_FRAC,
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
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {
            "memory_size": memory_size,
            "batch_memory_frac": batch_memory_frac,
            "batch_size": batch_size,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=GDumbLearner,
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
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            mask_unused_classes=mask_unused_classes,
        )
