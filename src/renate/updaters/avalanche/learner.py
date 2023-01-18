# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Type

import torch
from avalanche.core import BasePlugin, SupervisedPlugin
from avalanche.training.plugins import EWCPlugin, LwFPlugin, ReplayPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import BaseSGDTemplate, SupervisedTemplate
from torch.optim import Optimizer

from renate import defaults
from renate.updaters.learner import Learner, ReplayLearner


def replace_plugin(plugin: Optional[BasePlugin], plugins: List[BasePlugin]) -> List[BasePlugin]:
    """Replaces a plugin if already exists and appends otherwise.

    Args:
        plugin: New plugin that replaces existing one.
        plugins: List of current plugins.
    """
    idx = _plugin_index(type(plugin), plugins)
    if idx >= 0:
        plugins[idx] = plugin
    else:
        plugins.append(plugin)
    return plugins


def plugin_by_class(
    plugin_class: Type[BasePlugin], plugins: List[BasePlugin]
) -> Optional[BasePlugin]:
    idx = _plugin_index(plugin_class, plugins)
    if idx >= 0:
        return plugins[idx]
    return None


def _plugin_index(plugin_class: Type[BasePlugin], plugins: List[BasePlugin]) -> int:
    """Returns index at which a plugin of that type is located in the list.

    Returns:
        Returns location of plugin and ``-1`` if it does not exist.
    """
    plugins_types = [type(p) for p in plugins]
    assert max(Counter(plugins_types).values()) <= 1, "Duplicate plugins are not supported"
    if plugin_class in plugins_types:
        return plugins_types.index(plugin_class)
    return -1


class AvalancheLoaderMixing:
    def update_settings(
        self,
        avalanche_learner: BaseSGDTemplate,
        plugins: List[BasePlugin],
        # evaluator: EvaluationPlugin,
        optimizer: Optimizer,
        max_epochs: int,
        current_state_folder: Optional[str] = None,
    ):
        for plugin in plugins:  # + [evaluator]:
            avalanche_learner.plugins = replace_plugin(plugin, avalanche_learner.plugins)
        # avalanche_learner.evaluator = evaluator
        avalanche_learner.model = self._model
        avalanche_learner.optimizer = optimizer
        avalanche_learner._criterion = self._model.loss_fn
        avalanche_learner.train_epochs = max_epochs
        avalanche_learner.train_mb_size = self._batch_size
        avalanche_learner.eval_mb_size = self._batch_size
        if current_state_folder is not None:
            buffer_checkpoint_file = Path(current_state_folder) / defaults.BUFFER_CHECKPOINT_NAME
            if buffer_checkpoint_file.exists():
                self._val_memory_buffer.load_state_dict(
                    torch.load(Path(current_state_folder) / defaults.BUFFER_CHECKPOINT_NAME)
                )

    def _create_avalanche_learner(
        self,
        optimizer: Optimizer,
        train_epochs: int,
        plugins: List[SupervisedPlugin],
        # evaluator: EvaluationPlugin,
        device: torch.device,
        eval_every: int,
        **kwargs: Any,
    ) -> BaseSGDTemplate:
        return SupervisedTemplate(
            model=self._model,
            optimizer=optimizer,
            criterion=self._model.loss_fn,
            train_mb_size=self._batch_size,
            eval_mb_size=self._batch_size,
            train_epochs=train_epochs,
            plugins=plugins,
            evaluator=default_evaluator(),
            device=device,
            eval_every=eval_every,
            **kwargs,
        )


class AvalancheReplayLearner(ReplayLearner, AvalancheLoaderMixing):
    """Dummy class that enables consistent access and handling of inputs."""

    def create_avalanche_learner(
        self, plugins: List[SupervisedPlugin], **kwargs: Any
    ) -> BaseSGDTemplate:
        replay_plugin = ReplayPlugin(
            mem_size=self._memory_buffer._max_size,
            batch_size=self._batch_size,
            batch_size_mem=self._memory_batch_size,
        )
        plugins.append(replay_plugin)
        return self._create_avalanche_learner(plugins=plugins, **kwargs)


class AvalancheEWCLearner(Learner, AvalancheLoaderMixing):
    """"""

    def __init__(self, ewc_lambda: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ewc_lambda = ewc_lambda

    def update_settings(self, avalanche_learner: BaseSGDTemplate, **kwargs: Any):
        super().update_settings(avalanche_learner=avalanche_learner, **kwargs)
        plugin_by_class(EWCPlugin, avalanche_learner.plugins).ewc_lambda = self._ewc_lambda

    def create_avalanche_learner(
        self, plugins: List[SupervisedPlugin], **kwargs
    ) -> BaseSGDTemplate:
        ewc_plugin = EWCPlugin(ewc_lambda=self._ewc_lambda)
        plugins.append(ewc_plugin)
        return self._create_avalanche_learner(plugins=plugins, **kwargs)


class AvalancheLwFLearner(Learner, AvalancheLoaderMixing):
    """"""

    def __init__(self, alpha: float, temperature: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._alpha = alpha
        self._temperature = temperature

    def update_settings(self, avalanche_learner: BaseSGDTemplate, **kwargs: Any):
        super().update_settings(avalanche_learner=avalanche_learner, **kwargs)
        lwf_plugin = plugin_by_class(LwFPlugin, avalanche_learner.plugins)
        lwf_plugin.lwf.alpha = self._alpha
        lwf_plugin.lwf.temperature = self._temperature

    def create_avalanche_learner(
        self, plugins: List[SupervisedPlugin], **kwargs
    ) -> BaseSGDTemplate:
        lwf_plugin = LwFPlugin(alpha=self._alpha, temperature=self._temperature)
        plugins.append(lwf_plugin)
        return self._create_avalanche_learner(plugins=plugins, **kwargs)
