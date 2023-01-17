# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

import torch
import torchmetrics
from avalanche.benchmarks import dataset_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training import Naive
from avalanche.training.plugins import (
    EvaluationPlugin,
    LRSchedulerPlugin,
)
from avalanche.training.templates import BaseSGDTemplate
from torch.optim import Optimizer
from torch.utils.data import Dataset, TensorDataset

from renate import defaults
from renate.data.datasets import _TransformedDataset
from renate.models import RenateModule
from renate.updaters.avalanche.learner import AvalancheEWCLearner, AvalancheReplayLearner
from renate.updaters.avalanche.plugins import (
    RenateCheckpointPlugin,
    RenateFileSystemCheckpointStorage,
    SyneTunePlugin,
    metrics_mapper,
)
from renate.updaters.learner import Learner
from renate.updaters.model_updater import SimpleModelUpdater

logger = logging.getLogger(__name__)


class AvalancheModelUpdater(SimpleModelUpdater):
    def _load_learner(
        self,
        learner_class: Type[Learner],
        learner_kwargs: Dict[str, Any],
    ) -> BaseSGDTemplate:
        if self._early_stopping_enabled:  # TODO: support it
            raise Exception("Early stopping is not supported yet.")
        logger.warning(
            "Avalanche updaters currently support only accuracy and loss."
        )  # TODO: make use of passed metrics
        self._dummy_learner = learner_class(
            model=self._model,
            **learner_kwargs,
            logged_metrics=self._logged_metrics,
            **self._transforms_kwargs,
        )
        optimizer, scheduler = self._dummy_learner.configure_optimizers()
        optimizer, scheduler = optimizer[0], scheduler[0]
        lr_scheduler_plugin = LRSchedulerPlugin(scheduler=scheduler)
        avalanche_learner = self._load_if_exists(
            self._current_state_folder, self._metric, self._mode
        )

        checkpoint_plugin = None
        if self._next_state_folder is not None:
            checkpoint_plugin = RenateCheckpointPlugin(
                RenateFileSystemCheckpointStorage(
                    directory=Path(self._next_state_folder),
                ),
                metric=self._metric,
                mode=self._mode,
                # map_location=self._devices TODO
            )
        evaluator = self._create_evaluator()

        if avalanche_learner is None:
            logger.warning("No updater state available. Updating from scratch.")
            return self._create_avalanche_learner(
                evaluator=evaluator,
                checkpoint_plugin=checkpoint_plugin,
                lr_scheduler_plugin=lr_scheduler_plugin,
                optimizer=optimizer,
            )

        self._dummy_learner.update_settings(
            avalanche_learner=avalanche_learner,
            plugins=[checkpoint_plugin, lr_scheduler_plugin],
            evaluator=evaluator,
            optimizer=optimizer,
            max_epochs=self._max_epochs,
            current_state_folder=self._current_state_folder,
        )
        return avalanche_learner

    @staticmethod
    def _create_evaluator() -> EvaluationPlugin:
        return EvaluationPlugin(
            accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
            loggers=[SyneTunePlugin()],
        )

    def _get_device(self) -> torch.device:
        """Returns the device according to the chosen accelerator."""
        if self._accelerator == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif self._accelerator == "gpu":
            return torch.device("cuda", self._devices or 0)
        elif self._accelerator == "tpu":
            raise Exception("Not supported accelerator `TPU`.")
        return torch.device("cpu")

    def _create_avalanche_learner(
        self,
        evaluator: EvaluationPlugin,
        checkpoint_plugin: RenateCheckpointPlugin,
        lr_scheduler_plugin: LRSchedulerPlugin,
        optimizer: Optimizer,
    ) -> BaseSGDTemplate:
        """Returns an Avalanche learner based on the arguments passed to the ModelUpdater.

        Args:
            evaluator: Evaluation plugin.
            checkpoint_plugin: Plugin to checkpoint regularly.
            lr_scheduler_plugin: Plugin to adapt the learning rate.
            optimizer: PyTorch optimizer object used for training the Avalanche learner.
        """
        plugins = [lr_scheduler_plugin]
        if checkpoint_plugin is not None:
            plugins.append(checkpoint_plugin)
        avalanche_learner = self._dummy_learner.create_avalanche_learner(
            optimizer=optimizer,
            train_epochs=self._max_epochs,
            plugins=plugins,
            evaluator=evaluator,
            device=self._get_device(),
            eval_every=1,
        )
        avalanche_learner.is_logged_metric = (
            lambda metric_name: metric_name is None or metric_name in metrics_mapper.keys()
        )
        return avalanche_learner

    @staticmethod
    def _load_if_exists(
        current_state_folder: Optional[str], metric: str, mode: defaults.SUPPORTED_TUNING_MODE_TYPE
    ) -> Optional[Naive]:
        """Loads the Avalanche strategy if a state exists."""

        if current_state_folder is None:
            return None
        checkpoint_plugin = RenateCheckpointPlugin(
            RenateFileSystemCheckpointStorage(
                directory=Path(current_state_folder),
            ),
            metric=metric,
            mode=mode,
            # map_location=self._devices TODO
        )
        avalanche_learner, _ = checkpoint_plugin.load_checkpoint_if_exists()
        return avalanche_learner

    def update(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        task_id: Optional[str] = None,
    ) -> RenateModule:
        if isinstance(train_dataset, _TransformedDataset):
            x_data, y_data = [], []
            for x, y in train_dataset:
                x_data.append(x)
                y_data.append(y)
            train_dataset = TensorDataset(torch.stack(x_data), torch.stack(y_data))
        val_memory_dataset = train_dataset
        val_dataset_exists = val_dataset is not None
        if val_dataset_exists:
            self._dummy_learner._val_memory_buffer.update(val_dataset)
            val_memory_dataset = TensorDataset(*self._dummy_learner._val_memory_buffer.to_tensors())
        self._learner.evaluator.loggers[0].report_val = val_dataset_exists
        benchmark = dataset_benchmark(
            [train_dataset],
            [val_memory_dataset],
            train_transform=self._train_transform,
            train_target_transform=self._train_target_transform,
            eval_transform=self._test_transform,
            eval_target_transform=self._test_target_transform,
        )
        train_exp = benchmark.train_stream[0]
        self._learner.train(train_exp, eval_streams=[benchmark.test_stream])
        if self._next_state_folder is not None:
            Path(self._next_state_folder).mkdir(exist_ok=True, parents=True)  # TODO: remove
            torch.save(self._model.state_dict(), defaults.model_file(self._next_state_folder))
        if val_dataset_exists and self._next_state_folder is not None:
            torch.save(
                self._dummy_learner._val_memory_buffer.state_dict(),
                Path(self._next_state_folder) / defaults.BUFFER_CHECKPOINT_NAME,
            )
        results = self._learner.eval(benchmark.test_stream)
        SyneTunePlugin.report(
            epoch=self._max_epochs, results=results, report_val=val_dataset_exists
        )

        return self._model


class ExperienceReplayAvalancheModelUpdater(AvalancheModelUpdater):
    def __init__(
        self,
        model: RenateModule,
        memory_size: int,
        memory_batch_size: int = defaults.BATCH_SIZE,
        optimizer: defaults.SUPPORTED_OPTIMIZERS_TYPE = defaults.OPTIMIZER,
        learning_rate: float = defaults.LEARNING_RATE,
        learning_rate_scheduler: defaults.SUPPORTED_LEARNING_RATE_SCHEDULERS_TYPE = defaults.LEARNING_RATE_SCHEDULER,  # noqa: E501
        learning_rate_scheduler_gamma: float = defaults.LEARNING_RATE_SCHEDULER_GAMMA,
        learning_rate_scheduler_step_size: int = defaults.LEARNING_RATE_SCHEDULER_STEP_SIZE,
        momentum: float = defaults.MOMENTUM,
        weight_decay: float = defaults.WEIGHT_DECAY,
        batch_size: int = defaults.BATCH_SIZE,
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
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        seed: int = defaults.SEED,
    ):
        learner_kwargs = {
            "memory_size": memory_size,
            "memory_batch_size": memory_batch_size,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "learning_rate_scheduler": learning_rate_scheduler,
            "learning_rate_scheduler_gamma": learning_rate_scheduler_gamma,
            "learning_rate_scheduler_step_size": learning_rate_scheduler_step_size,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "seed": seed,
        }
        super().__init__(
            model,
            learner_class=AvalancheReplayLearner,
            learner_kwargs=learner_kwargs,
            current_state_folder=current_state_folder,
            next_state_folder=next_state_folder,
            max_epochs=max_epochs,
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
            accelerator=accelerator,
            devices=devices,
        )


class ElasticWeightConsolidationModelUpdater(AvalancheModelUpdater):
    def __init__(
        self,
        model: RenateModule,
        ewc_lambda: float = defaults.EWC_LAMBDA,
        optimizer: defaults.SUPPORTED_OPTIMIZERS_TYPE = defaults.OPTIMIZER,
        learning_rate: float = defaults.LEARNING_RATE,
        learning_rate_scheduler: defaults.SUPPORTED_LEARNING_RATE_SCHEDULERS_TYPE = defaults.LEARNING_RATE_SCHEDULER,  # noqa: E501
        learning_rate_scheduler_gamma: float = defaults.LEARNING_RATE_SCHEDULER_GAMMA,
        learning_rate_scheduler_step_size: int = defaults.LEARNING_RATE_SCHEDULER_STEP_SIZE,
        momentum: float = defaults.MOMENTUM,
        weight_decay: float = defaults.WEIGHT_DECAY,
        batch_size: int = defaults.BATCH_SIZE,
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
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        seed: int = defaults.SEED,
    ):
        learner_kwargs = {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "learning_rate_scheduler": learning_rate_scheduler,
            "learning_rate_scheduler_gamma": learning_rate_scheduler_gamma,
            "learning_rate_scheduler_step_size": learning_rate_scheduler_step_size,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "ewc_lambda": ewc_lambda,
            "seed": seed,
        }
        super().__init__(
            model,
            learner_class=AvalancheEWCLearner,
            learner_kwargs=learner_kwargs,
            current_state_folder=current_state_folder,
            next_state_folder=next_state_folder,
            max_epochs=max_epochs,
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
            accelerator=accelerator,
            devices=devices,
        )
