# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, List

import torch
from avalanche.core import BasePlugin, SupervisedPlugin
from avalanche.models import NCMClassifier, TrainEvalModel
from avalanche.training import ICaRL, ICaRLLossPlugin
from avalanche.training.plugins import EWCPlugin, LwFPlugin, ReplayPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.supervised.icarl import _ICaRLPlugin
from avalanche.training.templates import BaseSGDTemplate, SupervisedTemplate
from torch.optim import Optimizer

from renate.updaters.avalanche.plugins import RenateCheckpointPlugin
from renate.updaters.learner import Learner, ReplayLearner
from renate.utils.avalanche import plugin_by_class, remove_plugin, replace_plugin


class AvalancheLoaderMixin:
    """Mixin for Avalanche dummy learner classes."""

    def update_settings(
        self,
        avalanche_learner: BaseSGDTemplate,
        plugins: List[BasePlugin],
        optimizer: Optimizer,
        max_epochs: int,
        device: torch.device,
        eval_every: int,
    ) -> None:
        """Updates settings of Avalanche learner after reloading."""
        avalanche_learner.plugins = remove_plugin(RenateCheckpointPlugin, avalanche_learner.plugins)
        for plugin in plugins:
            avalanche_learner.plugins = replace_plugin(plugin, avalanche_learner.plugins)
        avalanche_learner.model = self._model
        avalanche_learner.optimizer = optimizer
        avalanche_learner._criterion = self._loss_fn
        avalanche_learner.train_epochs = max_epochs
        avalanche_learner.train_mb_size = self._batch_size
        avalanche_learner.eval_mb_size = self._batch_size + getattr(self, "_memory_batch_size", 0)
        avalanche_learner.device = device
        avalanche_learner.eval_every = eval_every

    def _create_avalanche_learner(
        self,
        optimizer: Optimizer,
        train_epochs: int,
        plugins: List[SupervisedPlugin],
        device: torch.device,
        eval_every: int,
        **kwargs: Any,
    ) -> BaseSGDTemplate:
        """Returns Avalanche object that this dummy learner wraps around."""
        return SupervisedTemplate(
            model=self._model,
            optimizer=optimizer,
            criterion=self._loss_fn,
            train_mb_size=self._batch_size,
            eval_mb_size=self._batch_size + getattr(self, "_memory_batch_size", 0),
            train_epochs=train_epochs,
            plugins=plugins,
            evaluator=default_evaluator(),
            device=device,
            eval_every=eval_every,
            **kwargs,
        )


class AvalancheReplayLearner(ReplayLearner, AvalancheLoaderMixin):
    """Renate wrapper around Avalanche Experience Replay."""

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


class AvalancheEWCLearner(Learner, AvalancheLoaderMixin):
    """Renate wrapper around Avalanche EWC."""

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


class AvalancheLwFLearner(Learner, AvalancheLoaderMixin):
    """Renate wrapper around Avalanche LwF"""

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


class AvalancheICaRLLearner(Learner, AvalancheLoaderMixin):
    """Renate wrapper around Avalanche ICaRL."""

    def __init__(self, memory_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._memory_size = memory_size

    def create_avalanche_learner(
        self,
        optimizer: Optimizer,
        train_epochs: int,
        plugins: List[SupervisedPlugin],
        device: torch.device,
        eval_every: int,
    ) -> BaseSGDTemplate:
        if not hasattr(self._model, "class_means"):
            raise RuntimeError(
                """The RenateModule must contain an attribute `class_means`.
                Please add something like

                self.class_means = torch.nn.Parameter(
                    torch.zeros((embedding_size, num_outputs)), requires_grad=False
                )
                """
            )
        if not hasattr(self._model, "get_backbone") or not hasattr(self._model, "get_predictor"):
            raise RuntimeError(
                "The RenateModule must be explicitly split into backbone and predictor module. "
                "Please implement functions `get_backbone()` and `get_predictor()` to return these "
                "modules."
            )
        icarl = ICaRL(
            feature_extractor=self._model.get_backbone(),
            classifier=self._model.get_predictor(),
            optimizer=optimizer,
            memory_size=self._memory_size,
            buffer_transform=None,  # TODO
            fixed_memory=True,
            train_mb_size=self._batch_size,
            train_epochs=train_epochs,
            eval_mb_size=self._batch_size,
            device=device,
            plugins=plugins,
            eval_every=-1,  # TODO: https://github.com/ContinualAI/avalanche/issues/1281
        )
        plugin_by_class(_ICaRLPlugin, icarl.plugins).class_means = self._model.class_means

        return icarl

    def update_settings(self, avalanche_learner: BaseSGDTemplate, **kwargs) -> None:
        super().update_settings(avalanche_learner=avalanche_learner, **kwargs)
        avalanche_learner.model = TrainEvalModel(
            feature_extractor=self._model.get_backbone(),
            train_classifier=self._model.get_predictor(),
            eval_classifier=NCMClassifier(),
        )
        icarl_loss_plugin = plugin_by_class(ICaRLLossPlugin, avalanche_learner.plugins)
        avalanche_learner._criterion = icarl_loss_plugin
        avalanche_learner.eval_every = (
            -1
        )  # TODO: https://github.com/ContinualAI/avalanche/issues/1281
