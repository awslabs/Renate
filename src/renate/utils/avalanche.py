# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from avalanche.benchmarks import dataset_benchmark
from avalanche.core import BasePlugin
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from renate.data.datasets import _TransformedDataset


class AvalancheSubset(Dataset):
    """Helper class to use dataset subsets with Avalanche.

    Since Avalanche does not support the Subset class, this class is used to convert subsets.
    Furthermore, some Avalanche updaters (e.g. iCaRL) directly access the targets attribute.

    Args:
        subset: The subset dataset that is converted.
    """

    def __init__(self, subset: Subset):
        super().__init__()
        x_data, y_data = [], []
        data_loader = DataLoader(subset)
        for x, y in data_loader:
            x_data.append(x)
            y_data.append(y.item())
        self.x: Tensor = torch.cat(x_data)
        self.y: List[int] = y_data
        self._targets: Optional[Tensor] = None

    @property
    def targets(self) -> Tensor:
        """Access to the dataset targets.

        Since not every Avalanche updater requires this attribute, we only create it when
        requested.
        """
        if self._targets is not None:
            return self._targets
        self._targets = torch.tensor(self.y, dtype=torch.long)
        return self._targets

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.y)


class AvalancheBenchmarkWrapper:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        train_transform,
        train_target_transform,
        test_transform,
        test_target_transform,
    ):
        self._n_classes_per_exp = None
        self._classes_order = None
        self._n_classes = 0
        self._train_dataset = train_dataset
        self._train_target_transform = train_target_transform
        self._benchmark = dataset_benchmark(
            [train_dataset],
            [val_dataset],
            train_transform=train_transform,
            train_target_transform=train_target_transform,
            eval_transform=test_transform,
            eval_target_transform=test_target_transform,
        )
        self.train_stream = self._benchmark.train_stream
        self.test_stream = self._benchmark.test_stream

    def update_benchmark_properties(self):
        dataset = _TransformedDataset(
            dataset=self._train_dataset, target_transform=self._train_target_transform
        )
        dataloader = DataLoader(dataset)
        unique_classes = set()
        for batch in dataloader:
            unique_classes.add(batch[1].item())
        if self._n_classes_per_exp is None:
            self._n_classes_per_exp = [len(unique_classes)]
            self._classes_order = list(sorted(unique_classes))
        else:
            self._n_classes_per_exp.append(len(unique_classes))
            self._classes_order += list(sorted(unique_classes))
        self._n_classes = sum(self._classes_order)
        self._benchmark.n_classes_per_exp = self._n_classes_per_exp
        self._benchmark.classes_order = self._classes_order

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the benchmark."""
        state_dict = {
            "n_classes_per_exp": self._n_classes_per_exp,
            "classes_order": self._classes_order,
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores the state of the benchmark."""
        self._n_classes_per_exp = state_dict["n_classes_per_exp"]
        self._classes_order = state_dict["classes_order"]
        self._benchmark.n_classes_per_exp = self._n_classes_per_exp
        self._benchmark.classes_order = self._classes_order


def replace_plugin(plugin: Optional[BasePlugin], plugins: List[BasePlugin]) -> List[BasePlugin]:
    """Replaces a plugin if already exists and appends otherwise.

    Args:
        plugin: New plugin that replaces existing one.
        plugins: List of current plugins.
    Returns:
        Reference to ``plugins``.
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
    """Returns plugin with respective class from a list of plugins.

    Args:
        plugin_class: Class type of interest in ``plugins``.
        plugins: List of plugins we search for an object of type ``plugin_class``.
    Returns:
        ``None`` if class does not exist, otherwise the respective object.
    """
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
    if max(Counter(plugins_types).values()) > 1:
        raise ValueError("Multiple occurrences of same type in `plugins` are not supported.")
    if plugin_class in plugins_types:
        return plugins_types.index(plugin_class)
    return -1
