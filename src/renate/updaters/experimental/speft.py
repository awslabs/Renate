# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.loggers.logger import Logger
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset

from renate import defaults
from renate.benchmark.models.spromptmodel import SPromptTransformer
from renate.models import RenateModule
from renate.updaters.learner import Learner
from renate.updaters.model_updater import SingleTrainingLoopUpdater


class SPeftLearner(Learner):
    """Learner to implement S-Prompts from
        ```Wang, Yabin, et.al .
        "S-prompts learning with pre-trained transformers: An occam’s razor for domain incremental learning."  # noqa: E501
        Advances in Neural Information Processing Systems 35 (2022): 5682-5695.```


    Args:
        model: The SPromptTransformer model to be trained.
        loss_fn: Loss function to be trained with.
        optimizer: Partial optimizer used to create an optimizer by passing the model parameters.
        learning_rate_scheduler: Partial object of learning rate scheduler that will be created by
            passing the optimizer.
        learning_rate_scheduler_interval: When to update the learning rate scheduler.
            Options: `epoch` and `step`.
        batch_size: Training batch size.
        train_transform: The transformation applied during training.
        train_target_transform: The target transformation applied during testing.
        test_transform: The transformation at test time.
        test_target_transform: The target transformation at test time.
        logged_metrics: Metrics logged additional to the default ones.
        seed: See :func:`renate.models.utils.get_generator`.
        mask_unused_classes: Masking logits corresponding to unused classes. Useful only for class
            incremental problems. Defaults to defaults.MASK_UNUSED_CLASSES.
    """

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

    def on_model_update_start(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_dataset_collate_fn: Optional[Callable] = None,
        val_dataset_collate_fn: Optional[Callable] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """A custom on_model_update_start hook for S-Peft methods.

        Here, we iterate oer the train data set and extract features. These features used to compute
        the task prototypes by the `update_task_identifier` call. Having this function in the model
        update start instead of end results in val metrics being reflective of test accuracy.
        """
        super().on_model_update_start(
            train_dataset, val_dataset, train_dataset_collate_fn, val_dataset_collate_fn, task_id
        )
        ## k-means
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self._model.to(device)
        features, labels = [], []
        with torch.inference_mode():
            for x, y in self.train_dataloader():
                features.append(self._model.features(x.to(device)).cpu())
                labels.append(y)
        features = torch.cat(features)
        labels = torch.cat(labels)
        self._model.update_task_identifier(features=features, labels=labels)

    def setup(self, stage: str) -> None:
        # We dont support distributed
        assert (
            self.trainer.world_size == 1
        ), "SPrompt learner does not support Multi-GPU training yet."
        if stage == "fit":
            # This needs to run before configure optimizers is called. The only hook is setup("fit")
            self._model.increment_task()

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ) -> None:
        """Explicitly setting grads to None instead of zero."""
        optimizer.zero_grad(set_to_none=True)


class SPeftModelUpdater(SingleTrainingLoopUpdater):
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
    ):
        learner_kwargs = {
            "batch_size": batch_size,
            "seed": seed,
            "loss_fn": loss_fn,
        }
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=SPeftLearner,
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
