# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Callable, Dict, List, Optional

import torch
import torchmetrics
from pytorch_lightning.loggers.logger import Logger
from torch.nn import Parameter
from torch.optim import Optimizer

from renate import defaults
from renate.models import RenateModule
from renate.updaters.learner import Learner
from renate.updaters.model_updater import SingleTrainingLoopUpdater


class FineTuningModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
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
            "batch_size": batch_size,
            "seed": seed,
            "loss_fn": loss_fn,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=Learner,
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
            metric=metric,
            mode=mode,
            logged_metrics=logged_metrics,
            early_stopping_enabled=early_stopping_enabled,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            deterministic_trainer=deterministic_trainer,
            strategy=strategy,
            precision=precision,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            mask_unused_classes=mask_unused_classes,
        )
