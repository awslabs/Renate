# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from renate import defaults
from renate.benchmark.models.l2p import LearningToPromptTransformer
from renate.models import RenateModule
from renate.types import NestedTensors
from renate.updaters.experimental.offline_er import OfflineExperienceReplayLearner
from renate.updaters.learner import Learner
from renate.updaters.model_updater import SingleTrainingLoopUpdater
from renate.utils.misc import maybe_populate_mask_and_ignore_logits

logger = logging.getLogger(__name__)


class LearningToPromptLearner(Learner):
    """Learner for learning to prompt

    This is identical to the base learner with an addition of loss term.
    TODO: Make this loss a component.

    Args:
        prompt_sim_loss_weight: Loss weight for the prompt key - image representation similarity
    """

    def __init__(
        self,
        prompt_sim_loss_weight: float = defaults.PROMPT_SIM_LOSS_WEIGHT,
        **kwargs,
    ) -> None:
        assert isinstance(
            kwargs["model"], LearningToPromptTransformer
        ), f"{self.__class__.__name__} can only train a LearningToPromptTransformer model"
        f"but got {type(kwargs['model'])}"
        super().__init__(
            **kwargs,
        )
        self.prompt_sim_loss_weight = prompt_sim_loss_weight
        self._loss_collections["train_losses"].update({"key_sim": torchmetrics.MeanMetric()})

    def training_step(
        self, batch: Tuple[NestedTensors, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        loss_dict = super().training_step(batch, batch_idx=batch_idx)
        key_similarity = -1 * self.prompt_sim_loss_weight * self._model.similarity_score
        loss_dict["loss"] += key_similarity
        self._loss_collections["train_losses"]["key_sim"](-key_similarity)
        return loss_dict


class LearningToPromptReplayLearner(OfflineExperienceReplayLearner):
    """L2P with an off-line ER learner.

    The model will be trained on weighted mixture of losses computed on the new data and a replay
    buffer. In contrast to the online version, the buffer will only be updated after training has
    terminated.

    Args:
        prompt_sim_loss_weight: Loss weight for the prompt key - image representation similarity
    """

    def __init__(
        self,
        prompt_sim_loss_weight: float = defaults.PROMPT_SIM_LOSS_WEIGHT,
        **kwargs,
    ) -> None:
        assert isinstance(
            kwargs["model"], LearningToPromptTransformer
        ), f"{self.__class__.__name__}  can only train a LearningToPromptTransformer model"
        f"but got {type(kwargs['model'])}"

        super().__init__(**kwargs)
        self.prompt_sim_loss_weight = prompt_sim_loss_weight
        self._loss_collections["train_losses"].update({"key_sim_loss": torchmetrics.MeanMetric()})

    def training_step(
        self, batch: Tuple[NestedTensors, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """PyTorch Lightning function to return the training loss."""
        # The reason for rewriting is to ensure two independent forward props of inputs and memory
        # samples. LearningToPromptTransformer uses per_batch_prompt which uses a single prompt
        # repeated across the batch. Hence, the separate processing of memory and input samples.
        if self._loss_weight_new_data is None:
            alpha = self._num_points_current_task / (
                self._num_points_current_task + self._num_points_previous_tasks
            )
        else:
            alpha = self._loss_weight_new_data
        alpha = torch.tensor(alpha, device=batch["current_task"][0].device)
        inputs, targets = batch["current_task"]
        outputs = self(inputs)

        outputs, self._class_mask = maybe_populate_mask_and_ignore_logits(
            self._mask_unused_classes,
            self._class_mask,
            self._classes_in_current_task,
            outputs,
        )

        if "memory" in batch:
            (inputs_mem, targets_mem), _ = batch["memory"]
            outputs_mem = self(inputs_mem)

            outputs_mem, self._class_mask = maybe_populate_mask_and_ignore_logits(
                self._mask_unused_classes,
                self._class_mask,
                self._classes_in_current_task,
                outputs_mem,
            )

        loss_current = self._loss_fn(outputs, targets).mean()
        if "memory" in batch:
            loss_memory = self._loss_fn(outputs_mem, targets_mem).mean()
            self._loss_collections["train_losses"]["base_loss"](loss_current)
            self._loss_collections["train_losses"]["memory_loss"](loss_memory)
            loss = alpha * loss_current + (1.0 - alpha) * loss_memory
        else:
            loss = loss_current.mean()
            self._loss_collections["train_losses"]["base_loss"](loss)
        self._update_metrics(outputs, targets, "train")

        key_similarity = -1 * self.prompt_sim_loss_weight * self._model.similarity_score
        loss += key_similarity
        self._loss_collections["train_losses"]["key_sim"](-key_similarity)
        return {"loss": loss}


class LearningToPromptModelUpdater(SingleTrainingLoopUpdater):
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
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {
            "batch_size": batch_size,
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
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            mask_unused_classes=mask_unused_classes,
        )


class LearningToPromptReplayModelUpdater(SingleTrainingLoopUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        batch_memory_frac: float = defaults.BATCH_MEMORY_FRAC,
        loss_weight_new_data: Optional[float] = None,
        learning_rate_scheduler: Optional[partial] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
        prompt_sim_loss_weight: float = defaults.PROMPT_SIM_LOSS_WEIGHT,
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
            "loss_weight_new_data": loss_weight_new_data,
            "batch_size": batch_size,
            "seed": seed,
            "prompt_sim_loss_weight": prompt_sim_loss_weight,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=LearningToPromptReplayLearner,
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
