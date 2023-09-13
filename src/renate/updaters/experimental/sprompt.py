# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.loggers.logger import Logger
from sklearn.cluster import KMeans
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from renate import defaults
from renate.benchmark.models.spromptmodel import SPromptTransformer
from renate.models import RenateModule
from renate.updaters.learner import Learner
from renate.updaters.model_updater import SingleTrainingLoopUpdater


class SPromptLearner(Learner):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        learning_rate_scheduler: Optional[Optional[Callable[[Optimizer], _LRScheduler]]] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
        batch_size: int = defaults.BATCH_SIZE,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
        logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        seed: int = defaults.SEED,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ) -> None:
        if not isinstance(model, SPromptTransformer):
            raise ValueError(
                "SPrompt Learner can only be used with a SPromptTransformer model."
                f"But got {type(model)}"
            )
        super().__init__(
            model,
            loss_fn,
            optimizer,
            learning_rate_scheduler,
            learning_rate_scheduler_interval,
            batch_size,
            train_transform,
            train_target_transform,
            test_transform,
            test_target_transform,
            logged_metrics,
            seed,
            mask_unused_classes,
        )

    def on_model_update_end(self) -> None:
        super().on_model_update_end()
        ## k-means
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self._model.to(device)
        features, labels = [], []
        with torch.inference_mode():
            for x, y in self.train_dataloader():
                features.append(self._model._backbone["transformer"](x.to(device)).cpu().numpy())
                labels.append(y.numpy())
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        self._model.update_task_identifier(features=features, labels=labels)

    def setup(self, stage: str) -> None:
        # We dont support distributed
        assert (
            self.trainer.world_size == 1
        ), "SPrompt learner does not support Multi-GPU training yet."
        if stage == "fit":
            # This needs to run before configure optimizers is called. The only hook is setup("fit")
            self._model.increment_task()


class SPromptModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[nn.Parameter]], Optimizer],
        batch_size: int = defaults.BATCH_SIZE,
        seed: int = defaults.SEED,
        learner_kwargs: Optional[Dict[str, Any]] = None,
        input_state_folder: Optional[str] = None,
        output_state_folder: Optional[str] = None,
        max_epochs: int = defaults.MAX_EPOCHS,
        learning_rate_scheduler: Optional[Optional[Callable[[Optimizer], _LRScheduler]]] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
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
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
        # clusters_per_task: int = defaults.CLUSTERS_PER_TASK,
    ):
        learner_kwargs = {
            "batch_size": batch_size,
            "seed": seed,
            "loss_fn": loss_fn,
            # "clusters_per_task": clusters_per_task,
        }
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=SPromptLearner,
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
