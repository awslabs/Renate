# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchmetrics
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader, Dataset

from renate import defaults
from renate.memory import GreedyClassBalancingBuffer
from renate.models.renate_module import RenateModule
from renate.updaters.learner import Learner, ReplayLearner
from renate.updaters.model_updater import SimpleModelUpdater
from renate.utils.pytorch import reinitialize_model_parameters


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

    def load_state_dict(self, model: RenateModule, state_dict: Dict[str, Any], **kwargs) -> None:
        """Restores the state of the learner."""
        Learner.load_state_dict(self, model, state_dict, **kwargs)
        self._memory_batch_size = state_dict["memory_batch_size"]
        self._memory_buffer = GreedyClassBalancingBuffer()
        self._memory_buffer.load_state_dict(state_dict["memory_buffer"])

    def on_model_update_start(
        self, train_dataset: Dataset, val_dataset: Dataset, task_id: Optional[str] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Called before a model update starts."""
        self._memory_buffer.update(train_dataset)
        _, val_loader = super().on_model_update_start(train_dataset, val_dataset, task_id)
        train_loader = DataLoader(
            self._memory_buffer,
            batch_size=self._batch_size,
            shuffle=True,
            generator=self._rng,
            pin_memory=True,
        )
        reinitialize_model_parameters(self._model)
        return train_loader, val_loader

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """PyTorch Lightning function to return the training loss."""
        batch, _ = batch
        return super().training_step(batch=batch, batch_idx=batch_idx)


class GDumbModelUpdater(SimpleModelUpdater):
    def __init__(
        self,
        model: RenateModule,
        memory_size: int,
        memory_batch_size: int = defaults.BATCH_SIZE,
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
            learner_class=GDumbLearner,
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
