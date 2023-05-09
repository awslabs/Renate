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
from torch.utils.data import DataLoader, Dataset

from renate import defaults
from renate.data.datasets import _TransformedDataset
from renate.evaluation.metrics.utils import create_metrics
from renate.memory import DataBuffer, InfiniteBuffer, ReservoirBuffer
from renate.models import RenateModule
from renate.types import NestedTensors
from renate.utils.optimizer import create_optimizer, create_scheduler
from renate.utils.pytorch import get_generator


class Learner(LightningModule, abc.ABC):
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
        optimizer: Optimizer used for training. Options: `Adam` or `SGD`.
        learning_rate: Initial learning rate used for training.
        learning_rate_scheduler: Learning rate scheduler used for training.
        learning_rate_scheduler_gamma: Learning rate scheduler gamma.
        learning_rate_scheduler_step_size: Learning rate scheduler step size.
        momentum: Momentum term (only relevant for optimizer `SGD`).
        weight_decay: L2 regularization applied to all model weights.
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
        optimizer: defaults.SUPPORTED_OPTIMIZERS_TYPE = defaults.OPTIMIZER,
        learning_rate: float = defaults.LEARNING_RATE,
        learning_rate_scheduler: defaults.SUPPORTED_LEARNING_RATE_SCHEDULERS_TYPE = defaults.LEARNING_RATE_SCHEDULER,  # noqa: E501
        learning_rate_scheduler_gamma: float = defaults.LEARNING_RATE_SCHEDULER_GAMMA,
        learning_rate_scheduler_step_size: int = defaults.LEARNING_RATE_SCHEDULER_STEP_SIZE,
        momentum: float = defaults.MOMENTUM,
        weight_decay: float = defaults.WEIGHT_DECAY,
        batch_size: int = defaults.BATCH_SIZE,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
        logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        seed: int = defaults.SEED,
    ) -> None:
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._learning_rate_scheduler = learning_rate_scheduler
        self._learning_rate_scheduler_gamma = learning_rate_scheduler_gamma
        self._learning_rate_scheduler_step_size = learning_rate_scheduler_step_size
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._batch_size = batch_size
        self._train_transform = train_transform
        self._train_target_transform = train_target_transform
        self._test_transform = test_transform
        self._test_target_transform = test_target_transform
        self._seed = seed
        self._task_id: str = defaults.TASK_ID

        self._val_memory_buffer: DataBuffer = InfiniteBuffer()
        self._create_metrics_collections(logged_metrics)
        self._post_init()

    def _post_init(self) -> None:
        self._rng = get_generator(self._seed)

    def _create_metrics_collections(
        self, logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None
    ) -> None:
        """Creates all logged metrics."""
        metrics = create_metrics(task="classification", additional_metrics=logged_metrics)
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

    def state_dict(self, **kwargs) -> Dict[str, Any]:
        """Returns the state of the learner."""
        return {
            "learner_class_name": self.__class__.__name__,
            "optimizer": self._optimizer,
            "learning_rate": self._learning_rate,
            "learning_rate_scheduler": self._learning_rate_scheduler,
            "learning_rate_scheduler_gamma": self._learning_rate_scheduler_gamma,
            "learning_rate_scheduler_step_size": self._learning_rate_scheduler_step_size,
            "momentum": self._momentum,
            "weight_decay": self._weight_decay,
            "batch_size": self._batch_size,
            "seed": self._seed,
            "task_id": self._task_id,
            "val_memory_buffer": self._val_memory_buffer.state_dict(),
        }

    def load_state_dict(self, model: RenateModule, state_dict: Dict[str, Any], **kwargs) -> None:
        """Restores the state of the learner.

        Even though this is a LightningModule, no modules are stored.

        Args:
            model: The model to be trained.
            state_dict: Dictionary containing the state.
        """
        if self.__class__.__name__ != state_dict["learner_class_name"]:
            raise RuntimeError(
                f"Learner of class {self.__class__} was used to load a state dict created by class "
                f"{state_dict['learner_class_name']}."
            )
        super().__init__()
        self._model = model
        self._optimizer = state_dict["optimizer"]
        self._learning_rate = state_dict["learning_rate"]
        self._learning_rate_scheduler = state_dict["learning_rate_scheduler"]
        self._learning_rate_scheduler_gamma = state_dict["learning_rate_scheduler_gamma"]
        self._learning_rate_scheduler_step_size = state_dict["learning_rate_scheduler_step_size"]
        self._momentum = state_dict["momentum"]
        self._weight_decay = state_dict["weight_decay"]
        self._batch_size = state_dict["batch_size"]
        self._seed = state_dict["seed"]
        self._task_id = state_dict["task_id"]
        if not hasattr(self, "_val_memory_buffer"):
            self._val_memory_buffer = InfiniteBuffer()
        self._val_memory_buffer.load_state_dict(state_dict["val_memory_buffer"])
        self._post_init()

    def save(self, output_state_dir: str) -> None:
        val_buffer_dir = os.path.join(output_state_dir, "val_memory_buffer")
        os.makedirs(val_buffer_dir, exist_ok=True)
        self._val_memory_buffer.save(val_buffer_dir)

    def load(self, input_state_dir: str) -> None:
        self._val_memory_buffer.load(os.path.join(input_state_dir, "val_memory_buffer"))

    def set_transforms(
        self,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
    ) -> None:
        """Update the transformations applied to the data."""
        self._train_transform = train_transform
        self._train_target_transform = train_target_transform
        self._test_transform = test_transform
        self._test_target_transform = test_target_transform

    def set_logged_metrics(
        self, logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None
    ) -> None:
        """Sets the additional metrics logged during training and evaluation."""
        self._create_metrics_collections(logged_metrics)

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

    def update_hyperparameters(self, args: Dict[str, Any]) -> None:
        """Update the hyperparameters of the learner."""
        if "optimizer" in args:
            self._optimizer = args["optimizer"]
        if "learning_rate" in args:
            self._learning_rate = args["learning_rate"]
        if "learning_rate_scheduler" in args:
            self._learning_rate_scheduler = args["learning_rate_scheduler"]
        if "learning_rate_scheduler_gamma" in args:
            self._learning_rate_scheduler_gamma = args["learning_rate_scheduler_gamma"]
        if "learning_rate_scheduler_step_size" in args:
            self._learning_rate_scheduler_step_size = args["learning_rate_scheduler_step_size"]
        if "momentum" in args:
            self._momentum = args["momentum"]
        if "weight_decay" in args:
            self._weight_decay = args["weight_decay"]
        if "batch_size" in args:
            self._batch_size = args["batch_size"]

    def on_model_update_start(
        self, train_dataset: Dataset, val_dataset: Dataset, task_id: Optional[str] = None
    ) -> None:
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self.val_enabled = val_dataset is not None and len(val_dataset)
        self._task_id = task_id
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
        )

    def val_dataloader(self) -> DataLoader:
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
            )
        else:
            return None

    def on_model_update_end(self) -> None:
        """Called right before a model update terminates."""
        pass

    def forward(self, inputs: NestedTensors, task_id: Optional[str] = None) -> torch.Tensor:
        """Forward pass of the model."""
        if task_id is None:
            task_id = self._task_id
        return self._model(inputs, task_id=task_id)

    def training_step(
        self, batch: Tuple[NestedTensors, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """PyTorch Lightning function to return the training loss."""
        inputs, targets = batch
        outputs = self(inputs)
        intermediate_representation = self._model.get_intermediate_representation()
        self._model.reset_intermediate_representation_cache()
        loss = self._model.loss_fn(outputs, targets)
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

    def validation_step(self, batch: Tuple[NestedTensors, torch.Tensor], batch_idx: int) -> None:
        """PyTorch Lightning function to estimate validation metrics."""
        (inputs, targets), _ = batch
        outputs = self(inputs)
        loss = self._model.loss_fn(outputs, targets)
        self._update_metrics(outputs, targets, "val")
        self._loss_collections["val_losses"]["loss"](loss)

    def validation_epoch_end(self, outputs: List[Union[Tensor, Dict[str, Any]]]) -> None:
        """PyTorch Lightning function to run at the end of validation epoch."""
        super().validation_epoch_end(outputs)
        self._log_metrics()

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """PyTorch Lightning function to create an optimizer."""
        optimizer = create_optimizer(
            params=self._model.get_params(self._task_id),
            optimizer=self._optimizer,
            lr=self._learning_rate,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
        )
        scheduler = create_scheduler(
            scheduler=self._learning_rate_scheduler,
            optimizer=optimizer,
            gamma=self._learning_rate_scheduler_gamma,
            step_size=self._learning_rate_scheduler_step_size,
        )
        return [optimizer], [scheduler]

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
            )
            self._metric_collections[f"{prefix}_metrics"].reset()

            for loss_name, loss in self._loss_collections[f"{prefix}_losses"].items():
                self.log(
                    f"{prefix}_{loss_name}",
                    loss.compute(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )
                loss.reset()


class ReplayLearner(Learner, abc.ABC):
    """Base class for Learners which use a buffer to store data and reuse it in future updates.

    Args:
        memory_size: The maximum size of the memory.
        memory_batch_size: Size of batches sampled from the memory. The memory batch will be
            appended to the batch sampled from the current dataset, leading to an effective batch
            size of `memory_batch_size + batch_size`.
        buffer_transform: The transformation to be applied to the memory buffer data samples.
        buffer_target_transform: The target transformation to be applied to the memory buffer target
            samples.
        seed: See :func:`renate.models.utils.get_generator`.
    """

    def __init__(
        self,
        memory_size: int,
        memory_batch_size: int = defaults.BATCH_SIZE,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
        seed: int = defaults.SEED,
        **kwargs,
    ) -> None:
        super().__init__(seed=seed, **kwargs)
        self._memory_batch_size = min(memory_size, memory_batch_size)
        self._memory_buffer = ReservoirBuffer(
            max_size=memory_size,
            seed=seed,
            transform=buffer_transform,
            target_transform=buffer_target_transform,
        )

    def state_dict(self, **kwargs) -> Dict[str, Any]:
        """Returns the state of the learner."""
        state_dict = super().state_dict(**kwargs)
        state_dict.update(
            {
                "memory_batch_size": self._memory_batch_size,
                "memory_buffer": self._memory_buffer.state_dict(),
            }
        )
        return state_dict

    def load_state_dict(self, model: RenateModule, state_dict: Dict[str, Any], **kwargs) -> None:
        """Restores the state of the learner."""
        super().load_state_dict(model, state_dict, **kwargs)
        self._memory_batch_size = state_dict["memory_batch_size"]
        if not hasattr(self, "_memory_buffer"):
            self._memory_buffer = ReservoirBuffer()
        self._memory_buffer.load_state_dict(state_dict["memory_buffer"])

    def save(self, output_state_dir: str) -> None:
        super().save(output_state_dir)
        buffer_dir = os.path.join(output_state_dir, "memory_buffer")
        os.makedirs(buffer_dir, exist_ok=True)
        self._memory_buffer.save(buffer_dir)

    def load(self, input_state_dir: str) -> None:
        super().load(input_state_dir)
        self._memory_buffer.load(os.path.join(input_state_dir, "memory_buffer"))

    def set_transforms(
        self,
        train_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
        buffer_transform: Optional[Callable] = None,
        buffer_target_transform: Optional[Callable] = None,
    ) -> None:
        """Update the transformations applied to the data."""
        super().set_transforms(
            train_transform=train_transform,
            train_target_transform=train_target_transform,
            test_transform=test_transform,
            test_target_transform=test_target_transform,
        )
        self._memory_buffer.set_transforms(buffer_transform, buffer_target_transform)
