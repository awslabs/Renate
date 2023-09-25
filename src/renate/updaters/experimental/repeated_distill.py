# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchmetrics
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from renate import defaults
from renate.memory import DataBuffer
from renate.models import RenateModule
from renate.types import NestedTensors
from renate.updaters.learner import Learner, ReplayLearner
from renate.updaters.model_updater import ModelUpdater
from renate.utils.pytorch import move_tensors_to_device, reinitialize_model_parameters


def double_distillation_loss(
    predicted_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    """Double distillation loss, where target logits are normalized across the class-dimension.

    This normalization is useful when distilling from multiple teachers and was proposed in

        TODO: Fix citation once we agreed on a format.
        Zhang, Junting, et al. "Class-incremental learning via deep model consolidation."
        Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2020.

    Args:
        predicted_logits: Logit predictions of the student model, size `(B, C)`, where `B` is the
            batch size and `C` is the number of classes.
        target_logits: Logits obtained from the teacher model(s), same size `(B, C)`.

    Returns:
        A tensor of size `(B,)` containing the loss values for each datapoint in the batch.
    """
    target_logits_normalized = target_logits - target_logits.mean(dim=1, keepdim=True)
    return 0.5 * (predicted_logits - target_logits_normalized).pow(2).mean(dim=1)


@torch.no_grad()
def extract_logits(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    task_id: Optional[str] = defaults.TASK_ID,
) -> torch.Tensor:
    """Extracts logits from a model for each point in a dataset.

    Args:
        model: The model. `model.get_logits(X)` is assumed to return logits.
        dataset: The dataset.
        batch_size: Batch size used to iterate over the dataset.
        task_id: Task id to be used, e.g., to select the output head.

    Returns:
        A tensor `logits` of shape `(N, C)` where `N` is the length of the dataset and `C` is
        the output dimension of `model`, i.e., the number of classes.
    """
    logits = []
    loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False, pin_memory=True)
    for batch in loader:
        inputs = batch[0][0] if isinstance(dataset, DataBuffer) else batch[0]
        device = next(model.parameters()).device
        inputs = move_tensors_to_device(inputs, device)
        logits.append(model.get_logits(inputs, task_id))
    return torch.cat(logits, dim=0)


class RepeatedDistillationModelUpdater(ModelUpdater):
    """Repeated Distillation (RD) is inspired by Deep Model Consolidation (DMC), which was proposed
    in

        TODO: Fix citation once we agreed on a format.
        Zhang, Junting, et al. "Class-incremental learning via deep model consolidation."
        Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2020.

    The idea underlying RD is the following: Given a new task/batch, a new model copy is trained
    from scratch on that data. Subsequently, this expert model is consolidated with the previous
    model state via knowledge distillation. The resulting consolidated model state is maintained,
    whereas the expert model is discarded.

    Our variant differs from the original algorithm in two ways:
        - The original algorithm is designed specifically for the class-incremental setting, where
          each new task introduces one or more novel classes. This variant is designed for the
          general continual learning setting with a pre-determined number of classes.
        - The original method is supposed to be memory-free and uses auxiliary data for the model
          consolidation phase. Our variant performs knowledge distillation over a memory
    """

    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        learning_rate_scheduler: Optional[partial] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
        batch_size: int = defaults.BATCH_SIZE,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
        input_state_folder: Optional[str] = None,
        output_state_folder: Optional[str] = None,
        max_epochs: int = defaults.MAX_EPOCHS,
        metric: Optional[str] = None,
        mode: defaults.SUPPORTED_TUNING_MODE_TYPE = "min",
        logger: Logger = defaults.LOGGER(**defaults.LOGGER_KWARGS),
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        strategy: str = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        seed: Optional[int] = None,
        early_stopping_enabled=False,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {"memory_size": memory_size, "batch_size": batch_size, "seed": seed}
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=RepeatedDistillationLearner,
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
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            early_stopping_enabled=early_stopping_enabled,
            logged_metrics=logged_metrics,
            deterministic_trainer=deterministic_trainer,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            mask_unused_classes=mask_unused_classes,
        )

    def update(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        train_dataset_collate_fn: Optional[Callable] = None,
        val_dataset_collate_fn: Optional[Callable] = None,
        task_id: Optional[str] = None,
    ) -> RenateModule:
        """Updates the model using the data passed as input.

        Args:
            train_dataset: The training data.
            val_dataset: The validation data.
            train_dataset_collate_fn: collate_fn used to merge a list of samples to form a
                mini-batch of Tensors for the training data.
            val_dataset_collate_fn: collate_fn used to merge a list of samples to form a
                mini-batch of Tensors for the validation data.
            task_id: The task id.
        """
        # First, train a copy of the model on the new data from scratch as an expert model. We use
        # the base `Learner` for that. The expert model and learner do not need to persist, we only
        # need it to extract logits.
        expert_model = copy.deepcopy(self._model)
        reinitialize_model_parameters(expert_model)
        expert_learner = Learner(
            model=expert_model,
            train_transform=self._train_transform,
            train_target_transform=self._train_target_transform,
            **{key: value for key, value in self._learner_kwargs.items() if key != "memory_size"},
        )
        expert_learner.on_model_update_start(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataset_collate_fn=train_dataset_collate_fn,
            val_dataset_collate_fn=val_dataset_collate_fn,
            task_id=task_id,
        )
        self._fit_learner(expert_learner)

        # Extract logits from the expert model and register them with the consolidation learner.
        expert_logits = extract_logits(
            expert_model,
            train_dataset,
            batch_size=self._learner_kwargs["batch_size"],
            task_id=task_id,
        )

        self._learner.update_expert_logits(expert_logits)
        del expert_model
        del expert_learner

        # Run consolidation.
        self._learner.on_model_update_start(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataset_collate_fn=train_dataset_collate_fn,
            val_dataset_collate_fn=val_dataset_collate_fn,
            task_id=task_id,
        )
        self._fit_learner(self._learner)
        return self._model


class RepeatedDistillationLearner(ReplayLearner):
    """A learner performing distillation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._expert_logits: Optional[torch.Tensor] = None

    def update_expert_logits(self, new_expert_logits: torch.Tensor) -> None:
        """Update expert logits."""
        self._expert_logits = new_expert_logits

    def on_model_update_start(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_dataset_collate_fn: Optional[Callable] = None,
        val_dataset_collate_fn: Optional[Callable] = None,
        task_id: Optional[int] = None,
    ) -> None:
        """Called before a model update starts."""
        super().on_model_update_start(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataset_collate_fn=train_dataset_collate_fn,
            val_dataset_collate_fn=val_dataset_collate_fn,
            task_id=task_id,
        )
        self._memory_buffer.update(train_dataset, metadata={"logits": self._expert_logits})
        reinitialize_model_parameters(self._model)
        self._expert_logits = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._memory_buffer,
            batch_size=self._batch_size,
            shuffle=True,
            generator=self._rng,
            pin_memory=True,
            collate_fn=self._train_collate_fn,
        )

    def on_model_update_end(self) -> None:
        """Called right before a model update terminates."""
        # Update the logits in memory using the newly consolidated model.
        logits = extract_logits(self._model, self._memory_buffer, self._batch_size)
        self._memory_buffer.set_metadata("logits", logits)

    def training_step(
        self,
        batch: Tuple[Tuple[NestedTensors, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """PyTorch Lightning function to return the training loss."""
        (inputs, targets), metadata = batch
        outputs = self(inputs)
        loss = double_distillation_loss(outputs, metadata["logits"]).mean()
        self._update_metrics(outputs, targets, prefix="train")
        self._loss_collections["train_losses"]["base_loss"](loss)
        return {"loss": loss, "outputs": outputs}
