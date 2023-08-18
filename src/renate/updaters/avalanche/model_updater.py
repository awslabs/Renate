# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torchmetrics
from avalanche.training import Naive
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.supervised.icarl import _ICaRLPlugin
from avalanche.training.templates import BaseSGDTemplate
from syne_tune import Reporter
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset

from renate import defaults
from renate.models import RenateModule
from renate.updaters.avalanche.learner import (
    AvalancheEWCLearner,
    AvalancheICaRLLearner,
    AvalancheLwFLearner,
    AvalancheReplayLearner,
    ICaRL,
    plugin_by_class,
)
from renate.updaters.avalanche.plugins import (
    RenateCheckpointPlugin,
    RenateFileSystemCheckpointStorage,
)
from renate.updaters.learner import Learner
from renate.updaters.model_updater import SingleTrainingLoopUpdater
from renate.utils.avalanche import AvalancheBenchmarkWrapper, to_avalanche_dataset

logger = logging.getLogger(__name__)

metrics_mapper = {
    "train_loss": "Loss_Epoch/train_phase/train_stream/Task000",
    "train_accuracy": "Top1_Acc_Epoch/train_phase/train_stream/Task000",
    "val_loss": "Loss_Stream/eval_phase/test_stream/Task000",
    "val_accuracy": "Top1_Acc_Stream/eval_phase/test_stream/Task000",
}


class AvalancheModelUpdater(SingleTrainingLoopUpdater):
    _report = Reporter()

    def __init__(self, *args, **kwargs):
        if kwargs.get("mask_unused_classes", False) is True:
            warnings.warn(
                "Avalanche model updaters do not support mask_unused_classes. Ignoring it."
            )
        super().__init__(*args, **kwargs)

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
        plugins = []
        lr_scheduler_plugin = None
        optimizers_scheduler = self._dummy_learner.configure_optimizers()
        if isinstance(optimizers_scheduler, tuple):
            optimizer, scheduler_config = optimizers_scheduler
            optimizer, scheduler_config = optimizer[0], scheduler_config[0]
            lr_scheduler_plugin = LRSchedulerPlugin(
                scheduler=scheduler_config["scheduler"],
                step_granularity="iteration"
                if scheduler_config["interval"] == "step"
                else scheduler_config["interval"],
            )
            plugins.append(lr_scheduler_plugin)
        else:
            optimizer = optimizers_scheduler
        avalanche_learner = self._load_if_exists(self._input_state_folder)

        checkpoint_plugin = None
        if self._output_state_folder is not None:
            checkpoint_plugin = RenateCheckpointPlugin(
                RenateFileSystemCheckpointStorage(
                    directory=Path(self._output_state_folder),
                ),
            )
            plugins.append(checkpoint_plugin)

        if avalanche_learner is None:
            logger.warning("No updater state available. Updating from scratch.")
            return self._create_avalanche_learner(
                checkpoint_plugin=checkpoint_plugin,
                lr_scheduler_plugin=lr_scheduler_plugin,
                optimizer=optimizer,
            )

        self._dummy_learner.update_settings(
            avalanche_learner=avalanche_learner,
            plugins=plugins,
            optimizer=optimizer,
            max_epochs=self._max_epochs,
            device=self._get_device(),
            eval_every=1,
        )
        return avalanche_learner

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
        checkpoint_plugin: RenateCheckpointPlugin,
        lr_scheduler_plugin: Optional[LRSchedulerPlugin],
        optimizer: Optimizer,
    ) -> BaseSGDTemplate:
        """Returns an Avalanche learner based on the arguments passed to the ModelUpdater.

        Args:
            checkpoint_plugin: Plugin to checkpoint regularly.
            lr_scheduler_plugin: Plugin to adapt the learning rate.
            optimizer: PyTorch optimizer object used for training the Avalanche learner.
        """
        plugins = []
        if lr_scheduler_plugin is not None:
            plugins.append(lr_scheduler_plugin)
        if checkpoint_plugin is not None:
            plugins.append(checkpoint_plugin)
        avalanche_learner = self._dummy_learner.create_avalanche_learner(
            optimizer=optimizer,
            train_epochs=self._max_epochs,
            plugins=plugins,
            device=self._get_device(),
            eval_every=1,
        )
        avalanche_learner.is_logged_metric = (
            lambda metric_name: metric_name is None or metric_name in metrics_mapper.keys()
        )
        return avalanche_learner

    @staticmethod
    def _load_if_exists(input_state_folder: Optional[str]) -> Optional[Naive]:
        """Loads the Avalanche strategy if a state exists."""

        if input_state_folder is None:
            return None
        checkpoint_plugin = RenateCheckpointPlugin(
            RenateFileSystemCheckpointStorage(
                directory=Path(input_state_folder),
            ),
        )
        avalanche_learner, _ = checkpoint_plugin.load_checkpoint_if_exists()
        return avalanche_learner

    def update(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        train_dataset_collate_fn: Optional[Callable] = None,
        val_dataset_collate_fn: Optional[Callable] = None,
        task_id: Optional[str] = None,
    ) -> RenateModule:
        val_dataset_exists = val_dataset is not None
        benchmark = self._load_benchmark_if_exists(
            train_dataset, val_dataset, train_dataset_collate_fn, val_dataset_collate_fn
        )
        train_exp = benchmark.train_stream[0]
        self._learner.train(train_exp, eval_streams=[benchmark.test_stream])
        results = self._learner.eval(benchmark.test_stream)
        if isinstance(self._learner, ICaRL):
            class_means = plugin_by_class(_ICaRLPlugin, self._learner.plugins).class_means
            self._model.class_means.data[:, :] = class_means
        if self._output_state_folder is not None:
            Path(self._output_state_folder).mkdir(
                exist_ok=True, parents=True
            )  # TODO: remove when checkpointing is active
            torch.save(self._model.state_dict(), defaults.model_file(self._output_state_folder))
            self._save_avalanche_state(benchmark, val_dataset_exists)
        self._report(
            **{
                metric_name: results[metric_internal_name]
                for metric_name, metric_internal_name in metrics_mapper.items()
                if val_dataset_exists or not metric_name.startswith("val")
            },
            step=self._max_epochs - 1,
            epoch=self._max_epochs,
        )
        return self._model

    def _load_benchmark_if_exists(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        train_dataset_collate_fn: Optional[Callable] = None,
        val_dataset_collate_fn: Optional[Callable] = None,
    ) -> AvalancheBenchmarkWrapper:
        train_dataset = to_avalanche_dataset(train_dataset, train_dataset_collate_fn)

        avalanche_state = None
        if self._input_state_folder is not None:
            avalanche_state_file = defaults.avalanche_state_file(self._input_state_folder)
            if Path(avalanche_state_file).exists():
                avalanche_state = torch.load(avalanche_state_file)
                if "val_memory_buffer" in avalanche_state:
                    self._dummy_learner._val_memory_buffer.load_state_dict(
                        avalanche_state["val_memory_buffer"]
                    )
                    self._dummy_learner.load(self._input_state_folder)
        if val_dataset is not None:
            self._dummy_learner._val_memory_buffer.update(val_dataset)
            val_memory_dataset = to_avalanche_dataset(
                self._dummy_learner._val_memory_buffer, val_dataset_collate_fn
            )
        else:
            val_memory_dataset = to_avalanche_dataset(train_dataset, val_dataset_collate_fn)

        benchmark = AvalancheBenchmarkWrapper(
            train_dataset=train_dataset,
            val_dataset=val_memory_dataset,
            train_transform=self._train_transform,
            train_target_transform=self._train_target_transform,
            test_transform=self._test_transform,
            test_target_transform=self._test_target_transform,
        )
        if avalanche_state is not None:
            benchmark.load_state_dict(avalanche_state)
        benchmark.update_benchmark_properties()
        return benchmark

    def _save_avalanche_state(self, benchmark: AvalancheBenchmarkWrapper, val_dataset_exists: bool):
        state = benchmark.state_dict()
        if val_dataset_exists:
            self._dummy_learner.save(self._output_state_folder)
            state["val_memory_buffer"] = self._dummy_learner._val_memory_buffer.state_dict()
        torch.save(state, defaults.avalanche_state_file(self._output_state_folder))


class ExperienceReplayAvalancheModelUpdater(AvalancheModelUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        batch_memory_frac: int = defaults.BATCH_MEMORY_FRAC,
        learning_rate_scheduler: Optional[Callable[[Optimizer], _LRScheduler]] = None,
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
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        strategy: Optional[str] = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        seed: int = defaults.SEED,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {
            "batch_size": batch_size,
            "memory_size": memory_size,
            "batch_memory_frac": batch_memory_frac,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=AvalancheReplayLearner,
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
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            mask_unused_classes=mask_unused_classes,
        )


class ElasticWeightConsolidationModelUpdater(AvalancheModelUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        ewc_lambda: float = defaults.EWC_LAMBDA,
        learning_rate_scheduler: Optional[Callable[[Optimizer], _LRScheduler]] = None,
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
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        strategy: Optional[str] = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        seed: int = defaults.SEED,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {
            "batch_size": batch_size,
            "ewc_lambda": ewc_lambda,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=AvalancheEWCLearner,
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
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            mask_unused_classes=mask_unused_classes,
        )


class LearningWithoutForgettingModelUpdater(AvalancheModelUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        alpha: float = defaults.LWF_ALPHA,
        temperature: float = defaults.LWF_TEMPERATURE,
        learning_rate_scheduler: Optional[Callable[[Optimizer], _LRScheduler]] = None,
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
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        seed: int = defaults.SEED,
        strategy: Optional[str] = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {
            "batch_size": batch_size,
            "alpha": alpha,
            "temperature": temperature,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=AvalancheLwFLearner,
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
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            mask_unused_classes=mask_unused_classes,
        )


class ICaRLModelUpdater(AvalancheModelUpdater):
    def __init__(
        self,
        model: RenateModule,
        loss_fn: torch.nn.Module,
        optimizer: Callable[[List[Parameter]], Optimizer],
        memory_size: int,
        learning_rate_scheduler: Optional[Callable[[Optimizer], _LRScheduler]] = None,
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
        accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
        devices: Optional[int] = None,
        strategy: Optional[str] = defaults.DISTRIBUTED_STRATEGY,
        precision: str = defaults.PRECISION,
        seed: int = defaults.SEED,
        deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
        gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
        mask_unused_classes: bool = defaults.MASK_UNUSED_CLASSES,
    ):
        learner_kwargs = {
            "memory_size": memory_size,
            "batch_size": batch_size,
            "seed": seed,
        }
        super().__init__(
            model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learner_class=AvalancheICaRLLearner,
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
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            mask_unused_classes=mask_unused_classes,
        )
