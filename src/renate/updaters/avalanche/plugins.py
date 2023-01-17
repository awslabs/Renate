# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from avalanche.core import SupervisedPlugin
from avalanche.logging import BaseLogger
from avalanche.training.determinism import RNGManager
from avalanche.training.plugins.checkpoint import CheckpointPlugin, FileSystemCheckpointStorage
from avalanche.training.templates import BaseSGDTemplate
from syne_tune import Reporter

from renate import defaults

metrics_mapper = {
    "train_loss": "Loss_Epoch/train_phase/train_stream/Task000",
    "train_accuracy": "Top1_Acc_Epoch/train_phase/train_stream/Task000",
    "val_loss": "Loss_Stream/eval_phase/test_stream/Task000",
    "val_accuracy": "Top1_Acc_Stream/eval_phase/test_stream/Task000",
}

"""
class RenateFileSystemCheckpointStorage(FileSystemCheckpointStorage):
    def _make_checkpoint_dir(self, checkpoint_name: str) -> Path:
        return self.directory

    def _make_checkpoint_file_path(self, checkpoint_name: str) -> Path:
        return Path(defaults.learner_state_file(str(self._make_checkpoint_dir(checkpoint_name))))

    def checkpoint_exists(self, checkpoint_name: str) -> bool:
        return self._make_checkpoint_file_path(checkpoint_name).exists()

    def checkpoint_delete(self, checkpoint_name: str) -> None:
        self._make_checkpoint_file_path(checkpoint_name).unlink(missing_ok=True)

    def load_checkpoint(self, checkpoint_name: str, checkpoint_loader) -> Any:
        checkpoint_file = self._make_checkpoint_file_path(checkpoint_name)
        with open(checkpoint_file, "rb") as f:
            return checkpoint_loader(f)


class RenateCheckpointPlugin(CheckpointPlugin):
    def __init__(
        self,
        storage: RenateFileSystemCheckpointStorage,
        metric: str,
        mode: defaults.SUPPORTED_TUNING_MODE_TYPE,
        map_location: Optional[Union[str, torch.device, Dict[str, str]]] = None,
    ):
        super().__init__(storage=storage, map_location=map_location)
        if metric is not None and metric not in metrics_mapper:
            raise ValueError(
                f"Unknown metric `{metric}`. Supported metrics: `{metrics_mapper.keys()}`."
            )
        self._metric = metric if metric is None else metrics_mapper[metric]
        self._mode = mode
        self._best_metric = None

    def load_checkpoint_if_exists(self):
        if not self.storage.checkpoint_exists(defaults.LEARNER_CHECKPOINT_NAME):
            return None, 0
        loaded_checkpoint = self.storage.load_checkpoint(
            defaults.LEARNER_CHECKPOINT_NAME, self.load_checkpoint
        )
        return loaded_checkpoint["strategy"], loaded_checkpoint["exp_counter"]

    def after_eval(self, strategy: BaseSGDTemplate, *args, **kwargs):
        if self._metric is not None:
            results = strategy.evaluator.get_last_metrics()
            if self._metric not in results:
                return
            if self._best_metric is not None and (
                self._mode == "min"
                and results[self._metric] >= self._best_metric
                or self._mode == "max"
                and results[self._metric] <= self._best_metric
            ):
                return

            self._best_metric = results[self._metric]

        ended_experience_counter = strategy.clock.train_exp_counter + (1 if self._training else 0)

        checkpoint_data = {
            "strategy": strategy,
            "rng_manager": RNGManager,
            "exp_counter": ended_experience_counter,
        }

        self.storage.checkpoint_delete(defaults.LEARNER_CHECKPOINT_NAME)

        self.storage.store_checkpoint(
            str(ended_experience_counter),
            partial(RenateCheckpointPlugin.save_checkpoint, checkpoint_data),
        )
"""


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
    def __init__(
        self,
        storage: RenateFileSystemCheckpointStorage,
        metric: str,
        mode: defaults.SUPPORTED_TUNING_MODE_TYPE,
        map_location: Optional[Union[str, torch.device, Dict[str, str]]] = None,
    ):
        super().__init__(storage=storage, map_location=map_location)

    def load_checkpoint_if_exists(self):
        if not self.storage.checkpoint_exists(defaults.LEARNER_CHECKPOINT_NAME):
            return None, 0
        loaded_checkpoint = self.storage.load_checkpoint(
            defaults.LEARNER_CHECKPOINT_NAME, self.load_checkpoint
        )
        return loaded_checkpoint["strategy"], loaded_checkpoint["exp_counter"]


class SyneTunePlugin(BaseLogger, SupervisedPlugin):
    _report = Reporter()

    def __init__(self) -> None:
        super().__init__()
        self._report_val = True

    @staticmethod
    def report(epoch: int, results: Dict[str, float], report_val: bool) -> None:
        SyneTunePlugin._report(
            **{
                metric_name: results[metric_internal_name]
                for metric_name, metric_internal_name in metrics_mapper.items()
                if report_val or not metric_name.startswith("val")
            },
            step=epoch - 1,
            epoch=epoch,
        )

    def before_training_epoch(self, strategy: BaseSGDTemplate, metric_values: Any, **kwargs):
        if strategy.clock.train_exp_epochs == 0:
            return
        self.report(
            epoch=strategy.clock.train_exp_epochs,
            results=strategy.evaluator.get_last_metrics(),
            report_val=self._report_val,
        )

    @property
    def report_val(self) -> bool:
        return self._report_val

    @report_val.setter
    def report_val(self, value: bool):
        self._report_val = value
