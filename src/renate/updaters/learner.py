# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from renate import defaults
from renate.data.datasets import _TransformedDataset
from renate.memory import DataBuffer, InfiniteBuffer, ReservoirBuffer
from renate.models import RenateModule
from renate.types import NestedTensors
from renate.utils.misc import maybe_populate_mask_and_ignore_logits
from renate.utils.pytorch import get_generator, unique_classes


class RenateLightningModule(LightningModule, abc.ABC):
    """Base class for LightningModules, which implement metric logging and basic training logic.

    The `RenateLightningModule` is a `LightningModule`, but provides additional hook functions
    called by `ModelUpdater`. These hooks are:

    - `on_model_update_start`, which is called in the beginning of a
       model update. We expect this to return train and (optionally) validation
       data loader(s).
    - `on_model_update_end`, which is called in the end of a model update.

    Args:
        model: The model to be trained.
        optimizer: Partial optimizer used to create an optimizer by passing the model parameters.
        learning_rate_scheduler: Partial object of learning rate scheduler that will be created by
            passing the optimizer.
        learning_rate_scheduler_interval: When to update the learning rate scheduler.
            Options: `epoch` and `step`.
        batch_size: Training batch size.
        logged_metrics: Metrics logged additional to the default ones.
        seed: See :func:`renate.models.utils.get_generator`.
        mask_unused_classes: Flag to use if logits corresponding to unused classes are to be ignored
            in the loss computation. Possibly useful for class incremental learning.
    """

    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        learning_rate_scheduler: Optional[Optional[Callable[[Optimizer], _LRScheduler]]] = None,
        learning_rate_scheduler_interval: defaults.SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = defaults.LR_SCHEDULER_INTERVAL,  # noqa: E501
        batch_size: int = defaults.BATCH_SIZE,
        logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        seed: int = defaults.SEED,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ) -> None:
        super().__init__()
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._learning_rate_scheduler = learning_rate_scheduler
        self._learning_rate_scheduler_interval = learning_rate_scheduler_interval
        self._batch_size = batch_size
        self._seed = seed
        self._mask_unused_classes = mask_unused_classes

        self._class_mask = None
        self._classes_in_current_task = None

        self._task_id: str = defaults.TASK_ID
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self.val_enabled = False
        self._train_collate_fn: Optional[Callable] = None
        self._val_collate_fn: Optional[Callable] = None

        self._create_metrics_collections(logged_metrics)
        self._rng = get_generator(self._seed)
        self.save_hyperparameters(ignore=self._ignored_hyperparameters())

    def _ignored_hyperparameters(self):
        """Hyperparameters to be ignored in the ``save_hyperparameters`` call."""
        return [
            "model",
            "loss_fn",
            "optimizer",
            "learning_rate_scheduler",
            "logged_metrics",
        ]

    def _create_metrics_collections(
        self, logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None
    ) -> None:
        """Creates all logged metrics."""
        if logged_metrics is None:
            logged_metrics = {}
        metrics = torchmetrics.MetricCollection(logged_metrics)
        train_metrics = metrics.clone(prefix="train_")
        val_metrics = metrics.clone(prefix="val_")

        train_losses = nn.ModuleDict(
            {
                "base_loss": torchmetrics.MeanMetric(),
                "loss": torchmetrics.MeanMetric(),
            }
        )
        val_losses = nn.ModuleDict({"loss": torchmetrics.MeanMetric()})

        self._metric_collections = nn.ModuleDict(
            {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
        )
        self._loss_collections = nn.ModuleDict(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
        )

    def is_logged_metric(self, metric_name: str) -> bool:
        """Returns `True` if there is a metric with name `metric_name`."""
        if metric_name is None:
            return True
        logged_metrics = list()
        for prefix in ["train", "val"]:
            for collection, collection_name in zip(
                [self._metric_collections, self._loss_collections], ["metrics", "losses"]
            ):
                collection_key = f"{prefix}_{collection_name}"
                if collection_key in collection:
                    logged_metrics += [
                        f"{prefix}_{logged_metric_name}"
                        for logged_metric_name in collection[collection_key]
                    ]
        return metric_name in logged_metrics

    def on_model_update_start(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_dataset_collate_fn: Optional[Callable] = None,
        val_dataset_collate_fn: Optional[Callable] = None,
        task_id: Optional[str] = None,
    ) -> None:
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self.val_enabled = val_dataset is not None and len(val_dataset) > 0
        self._train_collate_fn = train_dataset_collate_fn
        self._val_collate_fn = val_dataset_collate_fn
        self._task_id = task_id
        self._model.add_task_params(task_id=self._task_id)
        if self._mask_unused_classes:
            # The first forward prop will populate the _class_mask with the following
            # unique classes
            self._classes_in_current_task = unique_classes(self._train_dataset)

    def train_dataloader(self) -> DataLoader:
        """Returns the dataloader for training the model."""
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            generator=self._rng,
            pin_memory=True,
            collate_fn=self._train_collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self._val_dataset is not None:
            return DataLoader(
                self._val_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                generator=self._rng,
                pin_memory=True,
                collate_fn=self._val_collate_fn,
            )

    def on_model_update_end(self) -> None:
        """Called right before a model update terminates."""
        pass

    def forward(self, inputs: NestedTensors, task_id: Optional[str] = None) -> torch.Tensor:
        """Forward pass of the model."""
        if task_id is None:
            task_id = self._task_id
        return self._model(inputs, task_id=task_id)

    def training_step_unpack_batch(self, batch: Tuple[Any, Any]) -> Tuple[Any, Any]:
        inputs, targets = batch
        return inputs, targets

    def training_step(
        self, batch: Tuple[NestedTensors, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """PyTorch Lightning function to return the training loss."""
        inputs, targets = self.training_step_unpack_batch(batch)
        outputs = self(inputs)
        outputs, self._class_mask = maybe_populate_mask_and_ignore_logits(
            self._mask_unused_classes, self._class_mask, self._classes_in_current_task, outputs
        )
        intermediate_representation = self._model.get_intermediate_representation()
        self._model.reset_intermediate_representation_cache()
        loss = self._loss_fn(outputs, targets).mean()
        self._update_metrics(outputs, targets, "train")
        self._loss_collections["train_losses"]["base_loss"](loss)
        return {
            "loss": loss,
            "outputs": outputs,
            "intermediate_representation": intermediate_representation,
        }

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        """PyTorch Lightning function to perform after the training step."""
        super().training_step_end(step_output)
        self._loss_collections["train_losses"]["loss"](step_output["loss"])
        return step_output

    def training_epoch_end(self, outputs: List[Union[Tensor, Dict[str, Any]]]) -> None:
        """PyTorch Lightning function to run at the end of training epoch."""
        super().training_epoch_end(outputs)
        if not self.val_enabled:
            self._log_metrics()

    def validation_step_unpack_batch(self, batch: Tuple[Tuple[Any, Any], Any]) -> Tuple[Any, Any]:
        (inputs, targets), _ = batch
        return inputs, targets

    def validation_step(self, batch: Tuple[NestedTensors, torch.Tensor], batch_idx: int) -> None:
        """PyTorch Lightning function to estimate validation metrics."""
        inputs, targets = self.validation_step_unpack_batch(batch)
        outputs = self(inputs)
        loss = self._loss_fn(outputs, targets)
        self._update_metrics(outputs, targets, "val")
        self._loss_collections["val_losses"]["loss"](loss)

    def validation_epoch_end(self, outputs: List[Union[Tensor, Dict[str, Any]]]) -> None:
        """PyTorch Lightning function to run at the end of validation epoch."""
        super().validation_epoch_end(outputs)
        self._log_metrics()

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[List[Optimizer], List[Dict[str, Any]]]]:
        """PyTorch Lightning function to create optimizers and learning rate schedulers."""
        optimizer = self._optimizer(self._model.get_params(self._task_id))
        if self._learning_rate_scheduler is None:
            return optimizer
        lr_scheduler_config = {
            "scheduler": self._learning_rate_scheduler(optimizer),
            "interval": self._learning_rate_scheduler_interval,
        }
        return [optimizer], [lr_scheduler_config]

    def _update_metrics(
        self,
        outputs: torch.Tensor,
        y: torch.Tensor,
        prefix: Literal["train", "val"],
    ) -> None:
        """Shared logic for updating metrics."""
        self._metric_collections[f"{prefix}_metrics"](outputs, y)

    def _log_metrics(
        self,
    ) -> None:
        """Shared logic for logging metrics, including the loss."""
        if self.trainer.sanity_checking:
            return
        prefixes = ["train", "val"] if self.val_enabled else ["train"]
        for prefix in prefixes:
            self.log_dict(
                self._metric_collections[f"{prefix}_metrics"].compute(),
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self._metric_collections[f"{prefix}_metrics"].reset()

            for loss_name, loss in self._loss_collections[f"{prefix}_losses"].items():
                self.log(
                    f"{prefix}_{loss_name}",
                    loss.compute(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
                loss.reset()


class Learner(RenateLightningModule, abc.ABC):
    """Base class for Learners, which encapsulate the core CL methodologies.

    The `Learner` is a `LightningModule`, but provides additional hook functions
    called by `ModelUpdater`. These hooks are:

    - `Learner.on_model_update_start`, which is called in the beginning of a
       model update. We expect this to return train and (optionally) validation
       data loader(s).
    - `Learner.on_model_update_end`, which is called in the end of a model update.

    This base class implements a basic training loop without any mechanism to
    counteract forgetting.

    Args:
        model: The model to be trained.
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
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_scheduler_interval=learning_rate_scheduler_interval,
            batch_size=batch_size,
            logged_metrics=logged_metrics,
            seed=seed,
            mask_unused_classes=mask_unused_classes,
        )
        self._train_transform = train_transform
        self._train_target_transform = train_target_transform
        self._test_transform = test_transform
        self._test_target_transform = test_target_transform
        self._val_memory_buffer: DataBuffer = InfiniteBuffer()

    def _ignored_hyperparameters(self):
        """Hyperparameters to be ignored in the ``save_hyperparameters`` call."""
        return super()._ignored_hyperparameters() + [
            "components",
            "train_transform",
            "train_target_transform",
            "test_transform",
            "test_target_transform",
            "buffer_transform",
            "buffer_target_transform",
        ]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        learner_state_dict = {
            "learner_class_name": self.__class__.__name__,
            "val_memory_buffer": self._val_memory_buffer.state_dict(),
        }
        checkpoint.update(learner_state_dict)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        self._val_memory_buffer.load_state_dict(checkpoint["val_memory_buffer"])

    def save(self, output_state_dir: str) -> None:
        val_buffer_dir = os.path.join(output_state_dir, "val_memory_buffer")
        os.makedirs(val_buffer_dir, exist_ok=True)
        self._val_memory_buffer.save(val_buffer_dir)

    def load(self, input_state_dir: str) -> None:
        self._val_memory_buffer.load(os.path.join(input_state_dir, "val_memory_buffer"))

    def on_model_update_start(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_dataset_collate_fn: Optional[Callable] = None,
        val_dataset_collate_fn: Optional[Callable] = None,
        task_id: Optional[str] = None,
    ) -> None:
        super().on_model_update_start(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_dataset_collate_fn=train_dataset_collate_fn,
            val_dataset_collate_fn=val_dataset_collate_fn,
            task_id=task_id,
        )
        self._model.add_task_params(task_id=self._task_id)

    def train_dataloader(self) -> DataLoader:
        """Returns the dataloader for training the model."""
        train_dataset = _TransformedDataset(
            self._train_dataset,
            transform=self._train_transform,
            target_transform=self._train_target_transform,
        )
        return DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            generator=self._rng,
            pin_memory=True,
            collate_fn=self._train_collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self._val_dataset is not None:
            val_dataset = _TransformedDataset(
                self._val_dataset,
                transform=self._test_transform,
                target_transform=self._test_target_transform,
            )
            self._val_memory_buffer.update(val_dataset)

        if len(self._val_memory_buffer):
            return DataLoader(
                self._val_memory_buffer,
                batch_size=self._batch_size,
                shuffle=False,
                generator=self._rng,
                pin_memory=True,
                collate_fn=self._val_collate_fn,
            )

    def validation_step_unpack_batch(
        self, batch: Tuple[NestedTensors, torch.Tensor]
    ) -> Tuple[NestedTensors, Any]:
        (inputs, targets), _ = batch
        return inputs, targets


class ReplayLearner(Learner, abc.ABC):
    """Base class for Learners which use a buffer to store data and reuse it in future updates.

    Args:
        memory_size: The maximum size of the memory.
        batch_memory_frac: Fraction of the batch that is sampled from rehearsal memory.
        buffer_transform: The transformation to be applied to the memory buffer data samples.
        buffer_target_transform: The target transformation to be applied to the memory buffer target
            samples.
        seed: See :func:`renate.models.utils.get_generator`.
    """

    def __init__(
        self,
        memory_size: int,
        batch_size: int = defaults.BATCH_SIZE,
        batch_memory_frac: float = defaults.BATCH_MEMORY_FRAC,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
        seed: int = defaults.SEED,
        **kwargs,
    ) -> None:
        if not (0 <= batch_memory_frac <= 1):
            raise ValueError(
                f"Expecting batch_memory_frac to be in [0, 1], received {batch_memory_frac}."
            )
        memory_batch_size = min(memory_size, int(batch_memory_frac * batch_size))
        batch_size = batch_size - memory_batch_size
        super().__init__(batch_size=batch_size, seed=seed, **kwargs)
        self._memory_batch_size = memory_batch_size
        self._memory_buffer = ReservoirBuffer(
            max_size=memory_size,
            seed=seed,
            transform=buffer_transform,
            target_transform=buffer_target_transform,
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["memory_buffer"] = self._memory_buffer.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        self._memory_buffer.load_state_dict(checkpoint["memory_buffer"])

    def save(self, output_state_dir: str) -> None:
        super().save(output_state_dir)
        buffer_dir = os.path.join(output_state_dir, "memory_buffer")
        os.makedirs(buffer_dir, exist_ok=True)
        self._memory_buffer.save(buffer_dir)

    def load(self, input_state_dir: str) -> None:
        super().load(input_state_dir)
        self._memory_buffer.load(os.path.join(input_state_dir, "memory_buffer"))

    def on_model_update_start(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_dataset_collate_fn: Optional[Callable] = None,
        val_dataset_collate_fn: Optional[Callable] = None,
        task_id: Optional[str] = None,
    ) -> None:
        super().on_model_update_start(
            train_dataset, val_dataset, train_dataset_collate_fn, val_dataset_collate_fn, task_id
        )
        if self._mask_unused_classes:
            self._classes_in_current_task = self._classes_in_current_task.union(
                unique_classes(self._memory_buffer)
            )
