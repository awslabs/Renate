# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torchmetrics
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from syne_tune import Reporter
from torch.utils.data import DataLoader, Dataset

from renate import defaults
from .learner import Learner, ReplayLearner
from ..models import RenateModule

logging_logger = logging.getLogger(__name__)


class SyneTuneCallback(Callback):
    """Callback to report metrics to Syne Tune.

    Args:
        val_enabled: Whether validation was enabled in the Learner.
    """

    def __init__(self, val_enabled: bool):
        super().__init__()
        self._report = Reporter()
        self._val_enabled = val_enabled

    def _log(self, trainer: Trainer, training: bool) -> None:
        """Report the current epoch's results to Syne Tune.

        If validation was run `_val_enabled` is True, the results are reported at the end of
        the validation epoch. Otherwise, they are reported at the end of the training epoch.
        """

        if trainer.sanity_checking or (training and self._val_enabled):
            return
        self._report(
            **{k: v.item() for k, v in trainer.logged_metrics.items()},
            step=trainer.current_epoch,
            epoch=trainer.current_epoch + 1,
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log(trainer=trainer, training=pl_module.training)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log(trainer=trainer, training=pl_module.training)


class RenateModelCheckpoint(ModelCheckpoint):
    """Callback to save Renate state after each epoch.

    Args:
        model: Model to be saved when creating a checkpoint.
        state_folder: Checkpoint folder location.
        val_enabled: Whether validation was enabled in the Learner. Forwarded to `SyneTuneCallback`.
        metric: Monitored metric to decide when to write a new checkpoint. If no metric is provided
            or validation is not enabled, the latest model will be stored.
        mode: `min` or `max`. Whether to minimize or maximize the monitored `metric`.
        use_syne_tune_callback: Whether to use `SyneTuneCallback`.
    """

    def __init__(
        self,
        model: RenateModule,
        state_folder: str,
        val_enabled: bool,
        metric: Optional[str] = None,
        mode: defaults.SUPPORTED_TUNING_MODE_TYPE = "min",
        use_syne_tune_callback: bool = True,
    ) -> None:
        every_n_epochs = 1
        save_last = False
        if metric is None or not val_enabled:
            every_n_epochs = 0
            save_last = True
        learner_checkpoint_filename = Path(defaults.learner_state_file("")).stem
        super().__init__(
            dirpath=state_folder,
            filename=learner_checkpoint_filename,
            every_n_epochs=every_n_epochs,
            monitor=metric,
            mode=mode,
            save_last=save_last,
            save_weights_only=True,
        )
        self._model = model
        self._state_folder = state_folder
        self.CHECKPOINT_NAME_LAST = learner_checkpoint_filename
        # Delete old checkpoint if exists
        Path(defaults.learner_state_file(self._state_folder)).unlink(missing_ok=True)
        # FIXME: Hack to make sure Syne Tune is called after checkpointing.
        # Details: https://github.com/Lightning-AI/lightning/issues/15026
        # If fixed, remove on_train_epoch_end, on_validation_epoch_end, val_enabled, remove line
        # below, and add in ModelUpdaterSyneTune callback.
        if use_syne_tune_callback:
            self._syne_tune_callback = SyneTuneCallback(val_enabled)
        else:
            self._syne_tune_callback = None

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), defaults.model_file(self.dirpath))
        super()._save_checkpoint(trainer=trainer, filepath=filepath)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_end(trainer=trainer, pl_module=pl_module)
        if self._syne_tune_callback is not None:
            self._syne_tune_callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer=trainer, pl_module=pl_module)
        if self._syne_tune_callback is not None:
            self._syne_tune_callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)


class ModelUpdater(abc.ABC):
    """Updates a learner using the data provided.

    Args:
        model: The potentially pretrained model to be updated with new data.
        learner_class: Class of the learner to be used for model update.
        learner_kwargs: Arguments either used for creating a new learner (no previous
            state available) or replace current arguments of the learner.
        current_state_folder: Folder used by Renate to store files for current state.
        next_state_folder: Folder used by Renate to store files for next state.
        max_epochs: The maximum number of epochs used to train the model.
        train_transform: The transformation applied during training.
        train_target_transform: The target transformation applied during testing.
        test_transform: The transformation at test time.
        test_target_transform: The target transformation at test time.
        buffer_transform: Augmentations applied to the input data coming from the memory. Not all
            updaters require this. If required but not passed, `transform` will be used.
        buffer_target_transform: Transformations applied to the target. Not all updaters require
            this. If required but not passed, `target_transform` will be used.
        metric: Monitored metric to decide when to write a new checkpoint or early-stop the
            optimization. If no metric is provided, the latest model will be stored.
        mode: `min` or `max`. Whether to minimize or maximize the monitored `metric`.
        logged_metrics: Metrics logged additional to the default ones.
        early_stopping_enabled: Enables the early stopping of the optimization.
        logger: Logger used by PyTorch Lightning to log intermediate results.
        accelerator: Accelerator used by PyTorch Lightning to train the model.
        devices: Devices used by PyTorch Lightning to train the model. If the devices flag is not
            defined, it will assume devices to be "auto" and fetch the `auto_device_count` from the
            `accelerator`.
    """

    def __init__(
        self,
        model: RenateModule,
        learner_class: Type[Learner],
        learner_kwargs: Optional[Dict[str, Any]] = None,
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
    ):
        self._learner_kwargs = learner_kwargs or {}
        self._model = model
        self._learner_state_file: Optional[str] = None
        if current_state_folder is not None:
            self._learner_state_file = defaults.learner_state_file(current_state_folder)
        else:
            logging_logger.info(
                "No location for current updater state provided. Updating will start from scratch."
            )
        if next_state_folder is None:
            logging_logger.info(
                "No location for next updater state provided. No state will be stored."
            )
        elif metric is None:
            logging_logger.info(
                "Metric or mode is not provided. Checkpoint is saved only after training."
            )
        if metric is None and early_stopping_enabled:
            warnings.warn(
                "Early stopping is enabled but no metric is specified. Early stopping will be "
                "ignored."
            )
            early_stopping_enabled = False

        self._next_state_folder = next_state_folder
        self._metric = metric
        self._mode = mode
        self._logged_metrics = logged_metrics
        self._early_stopping_enabled = early_stopping_enabled
        self._train_transform = train_transform
        self._train_target_transform = train_target_transform
        self._test_transform = test_transform
        self._test_target_transform = test_target_transform
        self._buffer_transform = buffer_transform or train_transform
        self._buffer_target_transform = buffer_target_transform or train_target_transform
        self._transforms_kwargs = {
            "train_transform": self._train_transform,
            "train_target_transform": self._train_target_transform,
            "test_transform": self._test_transform,
            "test_target_transform": self._test_target_transform,
        }
        if issubclass(learner_class, ReplayLearner):
            self._transforms_kwargs["buffer_transform"] = self._buffer_transform
            self._transforms_kwargs["buffer_target_transform"] = self._buffer_target_transform
        self._learner = self._load_learner(learner_class, self._learner_kwargs)
        assert self._learner.is_logged_metric(metric), f"Target metric `{metric}` is not logged."
        self._max_epochs = max_epochs
        self._logger = logger
        self._num_epochs_trained = 0
        if accelerator not in defaults.SUPPORTED_ACCELERATORS:
            raise ValueError(
                f"Accelerator {accelerator} not supported. "
                f"Supported accelerators are {defaults.SUPPORTED_ACCELERATORS}."
            )
        self._accelerator = accelerator
        self._devices = devices

    @abc.abstractmethod
    def update(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        task_id: Optional[str] = None,
    ) -> RenateModule:
        """Updates the model using the data passed as input.

        Args:
            train_dataset: The training data.
            val_dataset: The validation data.
            task_id: The task id.
        """

    def _load_learner(
        self,
        learner_class: Type[Learner],
        learner_kwargs: Dict[str, Any],
    ) -> Learner:
        if self._learner_state_file is None or not Path(self._learner_state_file).is_file():
            logging_logger.warning("No updater state available. Updating from scratch.")
            return learner_class(
                model=self._model,
                **learner_kwargs,
                logged_metrics=self._logged_metrics,
                **self._transforms_kwargs,
            )
        learner = learner_class.__new__(learner_class)
        learner.load_state_dict(self._model, torch.load(self._learner_state_file)["state_dict"])
        learner.set_transforms(**self._transforms_kwargs)
        learner.set_logged_metrics(self._logged_metrics)
        learner.update_hyperparameters(learner_kwargs)
        return learner

    def _fit_learner(
        self,
        learner: Learner,
        train_loader: DataLoader,
        val_loader: DataLoader,
        use_syne_tune_callback: bool = True,
    ) -> None:
        callbacks: List[Callback] = []
        if use_syne_tune_callback:
            callbacks.append(SyneTuneCallback(val_loader is not None))
        if self._next_state_folder is not None:
            model_checkpoint_callback = RenateModelCheckpoint(
                model=self._model,
                state_folder=self._next_state_folder,
                metric=self._metric,
                mode=self._mode,
                val_enabled=val_loader is not None,
                use_syne_tune_callback=use_syne_tune_callback,
            )
            callbacks = [model_checkpoint_callback]  # FIXME: insert at 0 as soon as PTL is fixed.

        if self._early_stopping_enabled:
            if val_loader is not None:
                callbacks.insert(
                    0,
                    EarlyStopping(
                        monitor=self._metric,
                        mode=self._mode,
                    ),
                )
            else:
                warnings.warn(
                    "Early stopping is currently not supported without a validation set. It will "
                    "be ignored."
                )

        trainer = Trainer(
            accelerator=self._accelerator,
            devices=self._devices,
            max_epochs=self._max_epochs,
            callbacks=callbacks,
            logger=self._logger,
            enable_progress_bar=False,
        )
        trainer.fit(learner, train_loader, val_loader)
        self._num_epochs_trained = trainer.current_epoch


class SingleTrainingLoopUpdater(ModelUpdater):
    """Simple ModelUpdater which requires a single learner only to update the model."""

    def update(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        task_id: Optional[str] = None,
    ) -> RenateModule:
        """Updates the model using the data passed as input.

        Args:
            train_dataset: The training data.
            val_dataset: The validation data.
            task_id: The task id.
        """
        train_loader, val_loader = self._learner.on_model_update_start(
            train_dataset, val_dataset, task_id
        )

        self._fit_learner(self._learner, train_loader, val_loader)

        return self._learner.on_model_update_end(train_dataset, val_dataset, task_id)
