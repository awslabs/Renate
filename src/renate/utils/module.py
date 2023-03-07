# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Union

import torchmetrics

from renate import defaults
from renate.benchmark.scenarios import Scenario
from renate.data.data_module import RenateDataModule
from renate.evaluation.evaluator import evaluate
from renate.models import RenateModule


def evaluate_and_record_results(
    results: Dict[str, List[List[float]]],
    model: RenateModule,
    data_module: Union[Scenario, RenateDataModule],
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    task_id: str = defaults.TASK_ID,
    logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    metric_postfix: str = "",
    batch_size: int = defaults.BATCH_SIZE,
    accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
    devices: Optional[int] = None,
) -> Dict[str, List[List[float]]]:
    """A helper function that performs the evaluation on test data and records quantitative metrics
    in a dictionary.

    Args:
        results: The results dictionary to which the results should be saved.
        model: A RenateModule to be evaluated.
        data_module: A Scenario or RenateDataModule from which the test data is queried.
        transform: The transformation applied for evaluation.
        target_transform: The target transformation applied for evaluation.
        task_id: The task ID for which the evaluation should be performed.
        logged_metrics: Metrics logged additional to the default ones.
        metric_postfix: The postfix for the metric names.
        batch_size: A batch size for the test loader.
        accelerator: Accelerator used by PyTorch Lightning to train the model.
        devices: Devices used by PyTorch Lightning to train the model. If the devices flag is not
            defined, it will assume devices to be "auto" and fetch the `auto_device_count` from the
            `accelerator`.
    """

    data_module.setup()

    update_results = evaluate(
        model=model,
        test_dataset=data_module.test_data(),
        task_id=task_id,
        batch_size=batch_size,
        transform=transform,
        target_transform=target_transform,
        logged_metrics=logged_metrics,
        accelerator=accelerator,
        devices=devices,
    )
    for key, value in update_results.items():
        if key not in results:
            results[key + metric_postfix] = []
        results[key + metric_postfix].append(value)
    return results


def get_model(config_module: ModuleType, **kwargs: Any) -> RenateModule:
    """Creates and returns a model instance."""
    return getattr(config_module, "model_fn")(**kwargs)


def get_data_module(config_module: ModuleType, **kwargs: Any) -> RenateDataModule:
    """Creates and returns a data module instance."""
    return getattr(config_module, "data_module_fn")(**kwargs)


def get_metrics(config_module: ModuleType) -> Dict[str, torchmetrics.Metric]:
    """Creates and returns a dictionary of metrics."""
    metrics_fn_name = "metrics_fn"
    if metrics_fn_name in vars(config_module):
        return getattr(config_module, metrics_fn_name)()


def get_and_prepare_data_module(config_module: ModuleType, **kwargs: Any) -> RenateDataModule:
    """Prepares data."""
    data_module = get_data_module(config_module, **kwargs)
    data_module.prepare_data()
    return data_module


def get_and_setup_data_module(
    config_module: ModuleType,
    prepare_data: bool,
    **kwargs: Any,
) -> RenateDataModule:
    """Creates data module and possibly calls the prepare_data function needed for setup"""
    data_module = get_data_module(config_module, **kwargs)
    if prepare_data:
        data_module.prepare_data()
    data_module.setup()
    return data_module


def import_module(module_name: str, location: str) -> ModuleType:
    """Imports Python module from file location."""
    spec = importlib.util.spec_from_file_location(name=module_name, location=location)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
