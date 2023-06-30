# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from pytorch_lightning.loggers.logger import Logger

import torch
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from renate import defaults
from renate.models import RenateModule
from renate.updaters.learner import Learner
from renate.updaters.model_updater import SingleTrainingLoopUpdater
from renate.types import NestedTensors
from renate.updaters.experimental.er import ReplayLearner


class LearningToPromptLearner(ReplayLearner):
    def __init__(
        self,
        memory_size: int,
        memory_batch_size: int = defaults.BATCH_SIZE,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
        seed: int = defaults.SEED,
        prompt_sim_loss_weight: float = defaults.PROMPT_SIM_LOSS_WEIGHT,
        **kwargs,
    ) -> None:
        super().__init__(
            memory_size=memory_size,
            memory_batch_size=memory_batch_size,
            buffer_transform=buffer_transform,
            buffer_target_transform=buffer_target_transform,
            seed=seed,
            **kwargs,
        )
        self.prompt_sim_loss_weight = prompt_sim_loss_weight

    def training_step(
        self, batch: Tuple[NestedTensors, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        loss_dict = super().training_step(batch, batch_idx=batch_idx)
        key_similarity = self.prompt_sim_loss_weight * getattr(self._model, "similarity_score", 0.0)
        loss_dict["loss"] = loss_dict["loss"] + key_similarity

        return loss_dict


class LearningToPromptModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[nn.Parameter]], Optimizer],
        memory_size: int,
        memory_batch_size: int = defaults.BATCH_SIZE,
        batch_size: int = defaults.BATCH_SIZE,
        seed: int = defaults.SEED,
        learner_kwargs: Optional[Dict[str, Any]] = None,
        input_state_folder: Optional[str] = None,
        output_state_folder: Optional[str] = None,
        max_epochs: int = defaults.MAX_EPOCHS,
        learning_rate_scheduler: Optional[Optional[Callable[[Optimizer], _LRScheduler]]] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
        prompt_sim_loss_weight: float = defaults.PROMPT_SIM_LOSS_WEIGHT,
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
        strategy: Optional[str] = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
    ):
        learner_kwargs = {
            "batch_size": batch_size,
            "memory_size": memory_size,
            "memory_batch_size": memory_batch_size,
            "seed": seed,
            "loss_fn": loss_fn,
            "prompt_sim_loss_weight": prompt_sim_loss_weight,
        }
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=LearningToPromptLearner,
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
