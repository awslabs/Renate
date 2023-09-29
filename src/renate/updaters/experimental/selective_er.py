# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchmetrics
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Parameter
from torch.optim import Optimizer

from renate import defaults
from renate.models import RenateModule
from renate.types import NestedTensors
from renate.updaters.experimental.offline_er import OfflineExperienceReplayLearner
from renate.updaters.model_updater import SingleTrainingLoopUpdater
from renate.utils.misc import maybe_populate_mask_and_ignore_logits
from renate.utils.pytorch import cat_nested_tensors, get_length_nested_tensors


class SelectiveExperienceReplayLearner(OfflineExperienceReplayLearner):
    """(Offline) experience replay with selective backprop

    Args:
        TODO
    """

    def __init__(
        self, subsampling_ratio: float = 0.5, subsampling_strategy="loss_topk", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._subsampling_strategy = subsampling_strategy
        self._effective_batch_size = round(subsampling_ratio * self._batch_size)
        if not self._effective_batch_size > 0:
            raise ValueError(
                f"Subsampling ratio {subsampling_ratio} results in an effective batch size of 0."
                "Choose a larger subsampling ratio."
            )

    def training_step(self, batch: Dict[str, Tuple[NestedTensors]], batch_idx: int) -> STEP_OUTPUT:
        """PyTorch Lightning function to return the training loss."""
        inputs, targets = batch["current_task"]
        batch_size_current = get_length_nested_tensors(inputs)
        if "memory" in batch:
            (inputs_mem, targets_mem), _ = batch["memory"]
            inputs = cat_nested_tensors((inputs, inputs_mem), 0)
            targets = torch.cat((targets, targets_mem), 0)
        outputs = self(inputs)

        outputs, self._class_mask = maybe_populate_mask_and_ignore_logits(
            self._mask_unused_classes, self._class_mask, self._classes_in_current_task, outputs
        )
        losses = self._loss_fn(outputs, targets)
        # Just for logging.
        self._update_metrics(outputs, targets, "train")
        loss_current = losses[:batch_size_current].mean()
        loss_memory = losses[batch_size_current:].mean() if "memory" in batch else 0.0
        self._loss_collections["train_losses"]["base_loss"](loss_current)
        self._loss_collections["train_losses"]["memory_loss"](loss_memory)
        # This is used for backprop.
        if self._subsampling_strategy == "loss_topk":
            loss = torch.topk(losses, min(len(losses), self._effective_batch_size)).values.mean()
        else:
            raise ValueError(f"Unknown strategy: {self._strategy}.")
        return {"loss": loss}


class SelectiveExperienceReplayModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        batch_memory_frac: int = defaults.BATCH_MEMORY_FRAC,
        subsampling_ratio: float = 0.5,
        subsampling_strategy: str = "loss_topk",
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
            "subsampling_ratio": subsampling_ratio,
            "subsampling_strategy": subsampling_strategy,
            "batch_size": batch_size,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=SelectiveExperienceReplayLearner,
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
