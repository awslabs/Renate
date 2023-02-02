# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import itertools
from collections import Counter
from math import ceil
from pathlib import Path
from typing import Any, List, Optional, Sequence, Type, Union

import torch
from avalanche.benchmarks import CLExperience, classification_subset
from avalanche.benchmarks.utils import concat_datasets, make_tensor_classification_dataset
from avalanche.core import BasePlugin, SupervisedPlugin
from avalanche.models import NCMClassifier, TrainEvalModel
from avalanche.training import ICaRLLossPlugin
from avalanche.training.plugins import EWCPlugin, LwFPlugin, ReplayPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator
from avalanche.training.templates import BaseSGDTemplate, SupervisedTemplate
from avalanche.training.templates.base import ExpSequence
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

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
    ) -> None:
        for plugin in plugins:  # + [evaluator]:
            avalanche_learner.plugins = replace_plugin(plugin, avalanche_learner.plugins)
        # avalanche_learner.evaluator = evaluator
        avalanche_learner.model = self._model
        avalanche_learner.optimizer = optimizer
        avalanche_learner._criterion = self._model.loss_fn
        avalanche_learner.train_epochs = max_epochs
        avalanche_learner.train_mb_size = self._batch_size
        avalanche_learner.eval_mb_size = self._batch_size

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


class AvalancheICaRLLearner(ReplayLearner, AvalancheLoaderMixing):
    """"""

    def create_avalanche_learner(
        self,
        optimizer: Optimizer,
        train_epochs: int,
        plugins: List[SupervisedPlugin],
        # evaluator: EvaluationPlugin,
        device: torch.device,
        eval_every: int,
    ) -> BaseSGDTemplate:
        return ICaRL(
            feature_extractor=self._model._model,
            classifier=self._model._tasks_params[defaults.TASK_ID],
            optimizer=optimizer,
            memory_size=self._memory_buffer._max_size,
            buffer_transform=None,  # TODO
            fixed_memory=True,
            train_mb_size=self._batch_size,
            train_epochs=train_epochs,
            eval_mb_size=self._batch_size,
            device=device,
            plugins=plugins,
            # evaluator=evaluator,
            eval_every=-1,  # TODO
        )

    def update_settings(self, avalanche_learner: BaseSGDTemplate, **kwargs) -> None:
        super().update_settings(avalanche_learner=avalanche_learner, **kwargs)
        avalanche_learner.model = TrainEvalModel(
            feature_extractor=self._model._model,
            train_classifier=self._model._tasks_params[defaults.TASK_ID],
            eval_classifier=NCMClassifier(),
        )
        icarl_loss_plugin = plugin_by_class(ICaRLLossPlugin, avalanche_learner.plugins)
        avalanche_learner._criterion = icarl_loss_plugin


class ICaRL(SupervisedTemplate):
    """iCaRL Strategy.

    This strategy does not use task identities.
    """

    def __init__(
        self,
        feature_extractor: Module,
        classifier: Module,
        optimizer: Optimizer,
        memory_size,
        buffer_transform,
        fixed_memory,
        criterion=ICaRLLossPlugin(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
    ):
        """Init.

        :param feature_extractor: The feature extractor.
        :param classifier: The differentiable classifier that takes as input
            the output of the feature extractor.
        :param optimizer: The optimizer to use.
        :param memory_size: The nuber of patterns saved in the memory.
        :param buffer_transform: transform applied on buffer elements already
            modified by test_transform (if specified) before being used for
            replay
        :param fixed_memory: If True a memory of size memory_size is
            allocated and partitioned between samples from the observed
            experiences. If False every time a new class is observed
            memory_size samples of that class are added to the memory.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """
        model = TrainEvalModel(
            feature_extractor,
            train_classifier=classifier,
            eval_classifier=NCMClassifier(),
        )

        icarl = _ICaRLPlugin(memory_size, buffer_transform, fixed_memory)

        if plugins is None:
            plugins = [icarl]
        else:
            plugins += [icarl]

        if isinstance(criterion, SupervisedPlugin):
            plugins += [criterion]

        super().__init__(
            model,
            optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        print(vars(self.dataloader))
        # print(vars(self.dataloader.data))
        print(vars(self.dataloader._dl))
        for self.mbatch in self.dataloader:
            # print(self.mbatch)
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)


class _ICaRLPlugin(SupervisedPlugin):
    """
    iCaRL Plugin.
    iCaRL uses nearest class exemplar classification to prevent
    forgetting to occur at the classification layer. The feature extractor
    is continually learned using replay and distillation. The exemplars
    used for replay and classification are selected through herding.
    This plugin does not use task identities.
    """

    def __init__(self, memory_size, buffer_transform=None, fixed_memory=True):
        """
        :param memory_size: amount of patterns saved in the memory.
        :param buffer_transform: transform applied on buffer elements already
            modified by test_transform (if specified) before being used for
             replay
        :param fixed_memory: If True a memory of size memory_size is
            allocated and partitioned between samples from the observed
            experiences. If False every time a new class is observed
            memory_size samples of that class are added to the memory.
        """
        super().__init__()

        self.memory_size = memory_size
        self.buffer_transform = buffer_transform
        self.fixed_memory = fixed_memory

        self.x_memory = []
        self.y_memory = []
        self.order = []

        self.old_model = None
        self.observed_classes = []
        self.class_means = None
        self.embedding_size = None
        self.output_size = None
        self.input_size = None

    def after_train_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        if strategy.clock.train_exp_counter != 0:
            memory = make_tensor_classification_dataset(
                torch.cat(self.x_memory).cpu(),
                torch.tensor(list(itertools.chain.from_iterable(self.y_memory))),
                transform=self.buffer_transform,
                target_transform=None,
            )

            strategy.adapted_dataset = concat_datasets((strategy.adapted_dataset, memory))

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        tid = strategy.clock.train_exp_counter
        benchmark = strategy.experience.benchmark
        nb_cl = benchmark.n_classes_per_exp[tid]
        previous_seen_classes = sum(benchmark.n_classes_per_exp[:tid])

        self.observed_classes.extend(
            benchmark.classes_order[previous_seen_classes : previous_seen_classes + nb_cl]
        )

    def before_forward(self, strategy: "SupervisedTemplate", **kwargs):
        if self.input_size is None:
            with torch.no_grad():
                self.input_size = strategy.mb_x.shape[1:]
                self.output_size = strategy.model(strategy.mb_x).shape[1]
                self.embedding_size = strategy.model.feature_extractor(strategy.mb_x).shape[1]

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.model.eval()

        self.construct_exemplar_set(strategy)
        self.reduce_exemplar_set(strategy)
        self.compute_class_means(strategy)

    def compute_class_means(self, strategy):
        print("class_means", self.class_means)
        if self.class_means is None:
            print(strategy.experience.benchmark.n_classes_per_exp)
            n_classes = sum(strategy.experience.benchmark.n_classes_per_exp)
            self.class_means = torch.zeros((self.embedding_size, 10)).to(
                strategy.device
            )  # TODO 10->n_classes
        print("class_means.shape", self.class_means.shape)

        for i, class_samples in enumerate(self.x_memory):
            label = self.y_memory[i][0]
            class_samples = class_samples.to(strategy.device)

            with torch.no_grad():
                mapped_prototypes = strategy.model.feature_extractor(class_samples).detach()
            D = mapped_prototypes.T
            D = D / torch.norm(D, dim=0)

            if len(class_samples.shape) == 4:
                class_samples = torch.flip(class_samples, [3])

            with torch.no_grad():
                mapped_prototypes2 = strategy.model.feature_extractor(class_samples).detach()

            D2 = mapped_prototypes2.T
            D2 = D2 / torch.norm(D2, dim=0)

            div = torch.ones(class_samples.shape[0], device=strategy.device)
            div = div / class_samples.shape[0]

            m1 = torch.mm(D, div.unsqueeze(1)).squeeze(1)
            m2 = torch.mm(D2, div.unsqueeze(1)).squeeze(1)
            print("class_means.shape", self.class_means.shape)
            print("class_means[:, label].shape", self.class_means[:, label].shape)
            print(((m1 + m2) / 2).shape, m1.shape, m2.shape)
            self.class_means[:, label] = (m1 + m2) / 2
            self.class_means[:, label] /= torch.norm(self.class_means[:, label])

            strategy.model.eval_classifier.class_means = self.class_means

    def construct_exemplar_set(self, strategy: SupervisedTemplate):
        tid = strategy.clock.train_exp_counter
        benchmark = strategy.experience.benchmark
        nb_cl = benchmark.n_classes_per_exp[tid]
        previous_seen_classes = sum(benchmark.n_classes_per_exp[:tid])

        if self.fixed_memory:
            nb_protos_cl = int(ceil(self.memory_size / len(self.observed_classes)))
        else:
            nb_protos_cl = self.memory_size
        new_classes = self.observed_classes[previous_seen_classes : previous_seen_classes + nb_cl]

        dataset = strategy.experience.dataset
        targets = torch.tensor(dataset.targets)
        for iter_dico in range(nb_cl):
            cd = classification_subset(dataset, torch.where(targets == new_classes[iter_dico])[0])
            collate_fn = cd.collate_fn if hasattr(cd, "collate_fn") else None

            eval_dataloader = DataLoader(
                cd.eval(), collate_fn=collate_fn, batch_size=strategy.eval_mb_size
            )

            class_patterns = []
            mapped_prototypes = []
            for idx, (class_pt, _, _) in enumerate(eval_dataloader):
                class_pt = class_pt.to(strategy.device)
                class_patterns.append(class_pt)
                with torch.no_grad():
                    mapped_pttp = strategy.model.feature_extractor(class_pt).detach()
                mapped_prototypes.append(mapped_pttp)

            class_patterns = torch.cat(class_patterns, dim=0)
            mapped_prototypes = torch.cat(mapped_prototypes, dim=0)

            D = mapped_prototypes.T
            D = D / torch.norm(D, dim=0)

            mu = torch.mean(D, dim=1)
            order = torch.zeros(class_patterns.shape[0])
            w_t = mu

            i, added, selected = 0, 0, []
            while not added == nb_protos_cl and i < 1000:
                tmp_t = torch.mm(w_t.unsqueeze(0), D)
                ind_max = torch.argmax(tmp_t)

                if ind_max not in selected:
                    order[ind_max] = 1 + added
                    added += 1
                    selected.append(ind_max.item())

                w_t = w_t + mu - D[:, ind_max]
                i += 1

            pick = (order > 0) * (order < nb_protos_cl + 1) * 1.0
            self.x_memory.append(class_patterns[torch.where(pick == 1)[0]])
            self.y_memory.append([new_classes[iter_dico]] * len(torch.where(pick == 1)[0]))
            self.order.append(order[torch.where(pick == 1)[0]])

    def reduce_exemplar_set(self, strategy: SupervisedTemplate):
        tid = strategy.clock.train_exp_counter
        nb_cl = strategy.experience.benchmark.n_classes_per_exp

        if self.fixed_memory:
            nb_protos_cl = int(ceil(self.memory_size / len(self.observed_classes)))
        else:
            nb_protos_cl = self.memory_size

        for i in range(len(self.x_memory) - nb_cl[tid]):
            pick = (self.order[i] < nb_protos_cl + 1) * 1.0
            self.x_memory[i] = self.x_memory[i][torch.where(pick == 1)[0]]
            self.y_memory[i] = self.y_memory[i][: len(torch.where(pick == 1)[0])]
            self.order[i] = self.order[i][torch.where(pick == 1)[0]]
