# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from pathlib import Path
from typing import List, Optional

import torch
from avalanche.core import BasePlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import BaseSGDTemplate
from torch.nn import Module
from torch.optim import Optimizer

from renate import defaults
from renate.updaters.learner import Learner, ReplayLearner


def replace_plugin(plugin: Optional[BasePlugin], plugins: List[BasePlugin]) -> List[BasePlugin]:
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


class AvalancheLoaderMixing:
    def update_settings(
        self,
        avalanche_learner: BaseSGDTemplate,
        plugins: List[BasePlugin],
        evaluator: EvaluationPlugin,
        optimizer: Optimizer,
        max_epochs: int,
        current_state_folder: Optional[str] = None,
    ):
        for plugin in plugins + [evaluator]:
            avalanche_learner.plugins = replace_plugin(plugin, avalanche_learner.plugins)
        avalanche_learner.evaluator = evaluator
        avalanche_learner.model = self._model
        avalanche_learner.optimizer = optimizer
        avalanche_learner._criterion = self._model.loss_fn
        avalanche_learner.train_epochs = max_epochs
        avalanche_learner.train_mb_size = self._batch_size
        avalanche_learner.eval_mb_size = self._batch_size


class AvalancheLearner(Learner, AvalancheLoaderMixing):
    """"""


class AvalancheReplayLearner(ReplayLearner, AvalancheLoaderMixing):
    """Dummy class that enables consistent access and handling of inputs."""

    def update_settings(
        self,
        avalanche_learner: BaseSGDTemplate,
        plugins: List[BasePlugin],
        evaluator: EvaluationPlugin,
        optimizer: Optimizer,
        max_epochs: int,
        current_state_folder: Optional[str] = None,
    ):
        super().update_settings(
            avalanche_learner=avalanche_learner,
            plugins=plugins,
            evaluator=evaluator,
            optimizer=optimizer,
            max_epochs=max_epochs,
            current_state_folder=current_state_folder,
        )
        self._val_memory_buffer.load_state_dict(
            torch.load(Path(current_state_folder) / defaults.BUFFER_CHECKPOINT_NAME)
        )
