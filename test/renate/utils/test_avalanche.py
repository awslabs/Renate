# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from avalanche.training.plugins import EWCPlugin, ReplayPlugin
from torch import Tensor
from torch.utils.data import Subset, TensorDataset

from renate.utils.avalanche import (
    AvalancheBenchmarkWrapper,
    _plugin_index,
    plugin_by_class,
    remove_plugin,
    replace_plugin,
    to_avalanche_dataset,
)


def test_to_avalanche_dataset():
    expected_x = 6
    expected_y = 1
    tensor_dataset = TensorDataset(
        torch.tensor([5, expected_x, 7]), torch.tensor([0, expected_y, 2])
    )
    dataset = to_avalanche_dataset(Subset(tensor_dataset, [1]))
    assert dataset._inputs[0].item() == expected_x
    assert type(dataset._targets) == list
    assert len(dataset._targets) == 1
    assert dataset._targets[0] == expected_y
    assert type(dataset._targets[0]) == int
    assert dataset.targets.item() == dataset._targets[0]
    assert dataset.targets == expected_y
    assert type(dataset.targets) == Tensor
    x, y = dataset[0]
    assert x == expected_x and y == expected_y
    assert len(dataset) == 1


def test_avalanche_benchmark_wrapper_correctly_tracks_and_saves_state():
    """Check if the state we need to track for Avalanche works correctly."""

    def get_benchmark(dataset):
        return AvalancheBenchmarkWrapper(
            train_dataset=dataset,
            val_dataset=dataset,
            train_transform=None,
            train_target_transform=None,
            test_transform=None,
            test_target_transform=None,
        )

    dataset = TensorDataset(Tensor([5, 5]), Tensor([0, 1]))
    benchmark = get_benchmark(dataset)
    benchmark.update_benchmark_properties()
    assert benchmark._n_classes_per_exp == [2]
    assert benchmark._classes_order == [0, 1]
    benchmark_state = benchmark.state_dict()
    dataset = TensorDataset(Tensor([5, 5]), Tensor([3, 2]))
    benchmark = get_benchmark(dataset)
    benchmark.load_state_dict(benchmark_state)
    assert benchmark._n_classes_per_exp == [2]
    assert benchmark._classes_order == [0, 1]
    benchmark.update_benchmark_properties()
    assert benchmark._n_classes_per_exp == [2, 2]
    assert benchmark._classes_order == [0, 1, 2, 3]


def test_given_plugin_exists_in_list_replace_it():
    """Tests if `replace_plugin` replaces a plugin if a plugin of this class already exists."""
    to_be_replaced = ReplayPlugin()
    to_be_added = ReplayPlugin()
    plugins = [to_be_replaced, EWCPlugin(ewc_lambda=1)]
    plugins = replace_plugin(to_be_added, plugins)
    assert to_be_added == plugins[0]
    assert to_be_replaced not in plugins


def test_given_plugin_not_exists_in_list_append_it():
    """Tests if `replace_plugin` appends a plugin if a plugin of this class does not exist."""
    to_be_added = ReplayPlugin()
    plugins = [EWCPlugin(ewc_lambda=1)]
    plugins = replace_plugin(to_be_added, plugins)
    assert to_be_added == plugins[-1]


def test_given_type_and_object_exists_in_list_returns_it():
    """``plugin_by_class`` returns object by type from list if exists."""
    expected = ReplayPlugin()
    assert expected == plugin_by_class(ReplayPlugin, [expected, EWCPlugin(ewc_lambda=1)])


def test_given_plugin_exists_in_list_remove_it():
    """Tests if `remove_plugin` removes a plugin if a plugin of this class already exists."""
    to_be_removed = ReplayPlugin()
    expected_plugin = EWCPlugin(ewc_lambda=1)
    plugins = [to_be_removed, expected_plugin]
    plugins = remove_plugin(ReplayPlugin, plugins)
    assert expected_plugin == plugins[0]
    assert len(plugins) == 1


def test_given_type_and_object_exists_not_in_list_returns_none():
    """``plugin_by_class`` returns None if type is not in list."""
    assert None is plugin_by_class(ReplayPlugin, [EWCPlugin(ewc_lambda=1)])


def test_duplicate_plugins_raise_exception():
    """Helper function do not allow for duplicate plugin types."""
    with pytest.raises(ValueError):
        _plugin_index(ReplayPlugin, [ReplayPlugin(), ReplayPlugin()])
