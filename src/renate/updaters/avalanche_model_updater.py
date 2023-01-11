# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torchmetrics
from avalanche.benchmarks import dataset_benchmark
from avalanche.core import BasePlugin
from avalanche.training import Naive
from avalanche.training.plugins import LRSchedulerPlugin, ReplayPlugin
from avalanche.training.plugins.checkpoint import CheckpointPlugin, FileSystemCheckpointStorage
from pytorch_lightning.loggers import Logger
from syne_tune import Reporter
from torch.optim import Optimizer
from torch.utils.data import Dataset

from renate import defaults
from .learner import Learner, ReplayLearner
from .model_updater import SimpleModelUpdater
from ..models import RenateModule

logging_logger = logging.getLogger(__name__)


class RenateFileSystemCheckpointStorage(FileSystemCheckpointStorage):
    def _make_checkpoint_dir(self, checkpoint_name: str) -> Path:
        return self.directory

    def _make_checkpoint_file_path(self, checkpoint_name: str) -> Path:
        return Path(defaults.learner_state_file(str(self._make_checkpoint_dir(checkpoint_name))))

    def checkpoint_exists(self, checkpoint_name: str) -> bool:
        return self._make_checkpoint_file_path(checkpoint_name).exists()

    def load_checkpoint(self, checkpoint_name: str, checkpoint_loader) -> Any:
        checkpoint_file = self._make_checkpoint_file_path(checkpoint_name)
        with open(checkpoint_file, "rb") as f:
            return checkpoint_loader(f)


class RenateCheckpointPlugin(CheckpointPlugin):
    def load_checkpoint_if_exists(self):
        if not self.storage.checkpoint_exists(defaults.LEARNER_CHECKPOINT_NAME):
            return None, 0
        loaded_checkpoint = self.storage.load_checkpoint(
            defaults.LEARNER_CHECKPOINT_NAME, self.load_checkpoint
        )
        return loaded_checkpoint["strategy"], loaded_checkpoint["exp_counter"]


class AvalancheReplayLearner(ReplayLearner):
    """Dummy class that enables consistent access and handling of inputs."""


class AvalancheModelUpdater(SimpleModelUpdater):
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
        logger: Logger = defaults.LOGGER(**defaults.LOGGER_KWARGS),
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
        self._current_state_folder = current_state_folder
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
            logged_metrics=logged_metrics,  # TODO: not used
            early_stopping_enabled=early_stopping_enabled,  # TODO: not used
            logger=logger,  # TODO: not used
            accelerator=accelerator,  # TODO: not used
            devices=devices,  # TODO: not used
        )
        self._report = Reporter()

    def _load_learner(
        self,
        learner_class: Type[Learner],
        learner_kwargs: Dict[str, Any],
    ) -> Naive:
        self._dummy_learner = learner_class(
            model=self._model,
            **learner_kwargs,
            logged_metrics=self._logged_metrics,
            **self._transforms_kwargs,
        )
        optimizer, scheduler = self._dummy_learner.configure_optimizers()
        optimizer, scheduler = optimizer[0], scheduler[0]
        lr_scheduler_plugin = LRSchedulerPlugin(scheduler=scheduler)
        avalanche_learner = self._load_if_exists(self._current_state_folder)

        checkpoint_plugin = None
        if self._next_state_folder is not None:
            checkpoint_plugin = RenateCheckpointPlugin(
                RenateFileSystemCheckpointStorage(
                    directory=Path(self._next_state_folder),
                ),
                # map_location=self._devices TODO
            )
        if avalanche_learner is None:
            logging_logger.warning("No updater state available. Updating from scratch.")
            return self._create_avalanche_learner(
                learner=self._dummy_learner,
                checkpoint_plugin=checkpoint_plugin,
                lr_scheduler_plugin=lr_scheduler_plugin,
                optimizer=optimizer,
            )
        avalanche_learner.plugins = self._replace_plugin(
            checkpoint_plugin, avalanche_learner.plugins
        )
        avalanche_learner.plugins = self._replace_plugin(
            lr_scheduler_plugin, avalanche_learner.plugins
        )
        avalanche_learner.model = self._model
        avalanche_learner.optimizer = optimizer
        avalanche_learner._criterion = self._model.loss_fn
        avalanche_learner.train_epochs = self._max_epochs
        avalanche_learner.train_mb_size = self._dummy_learner._batch_size
        avalanche_learner.eval_mb_size = self._dummy_learner._batch_size
        return avalanche_learner

    def _create_avalanche_learner(
        self,
        learner: Learner,
        checkpoint_plugin: CheckpointPlugin,
        lr_scheduler_plugin: LRSchedulerPlugin,
        optimizer: Optimizer,
    ) -> Naive:
        replay_plugin = ReplayPlugin(
            mem_size=learner._memory_buffer._max_size,
            batch_size_mem=learner._memory_batch_size,
        )
        plugins = [replay_plugin, lr_scheduler_plugin]
        if checkpoint_plugin is not None:
            plugins.append(checkpoint_plugin)
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # TODO: respect selected devices
        avalanche_learner = Naive(
            model=self._model,
            optimizer=optimizer,
            criterion=self._model.loss_fn,
            train_mb_size=learner._batch_size,
            eval_mb_size=learner._batch_size,
            train_epochs=self._max_epochs,
            plugins=[replay_plugin, checkpoint_plugin],
            device=device,
        )
        avalanche_learner.is_logged_metric = lambda metric_name: metric_name in [
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ]
        return avalanche_learner

    @staticmethod
    def _load_if_exists(current_state_folder) -> Optional[Naive]:
        """Loads the Avalanche strategy if a state exists."""

        if current_state_folder is None:
            return None
        checkpoint_plugin = RenateCheckpointPlugin(
            RenateFileSystemCheckpointStorage(
                directory=Path(current_state_folder),
            ),
            # map_location=self._devices TODO
        )
        avalanche_learner, _ = checkpoint_plugin.load_checkpoint_if_exists()
        return avalanche_learner

    @staticmethod
    def _replace_plugin(
        plugin: Optional[BasePlugin], plugins: List[BasePlugin]
    ) -> List[BasePlugin]:
        """Replaces a plugin if already exists and appends otherwise.

        Args:
            plugin: New plugin that replaces existing one.
            plugins: List of current plugins.
        """
        plugins_types = [type(p) for p in plugins]
        assert max(Counter(plugins_types).values()) <= 1, "Duplicate plugins are not supported"
        if plugin is None:
            return plugins
        if type(plugin) in plugins_types:
            plugins[plugins_types.index(type(plugin))] = plugin
        else:
            plugins.append(plugin)
        return plugins

    def update(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        task_id: Optional[str] = None,
    ) -> RenateModule:
        print(vars(val_dataset))
        self._dummy_learner._val_memory_buffer.update(val_dataset)
        print(vars(self._dummy_learner._val_memory_buffer))
        benchmark = dataset_benchmark(
            [train_dataset],
            [self._dummy_learner._val_memory_buffer],
            train_transform=self._train_transform,
            train_target_transform=self._train_target_transform,
            eval_transform=self._test_transform,
            eval_target_transform=self._test_target_transform,
        )
        train_exp = benchmark.train_stream[0]
        self._learner.train(train_exp)
        results = self._learner.eval(benchmark.test_stream)
        torch.save(self._model.state_dict(), defaults.model_file(self._next_state_folder))
        self._report(
            train_loss=results["Loss_Epoch/train_phase/train_stream/Task000"],
            train_accuracy=results["Top1_Acc_Epoch/train_phase/train_stream/Task000"],
            val_loss=results["Loss_Stream/eval_phase/test_stream/Task000"],
            val_accuracy=results["Top1_Acc_Stream/eval_phase/test_stream/Task000"],
            step=self._max_epochs,
            epoch=self._max_epochs + 1,
        )
        return self._model
