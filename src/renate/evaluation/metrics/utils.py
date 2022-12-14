# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional

import torchmetrics

import renate.defaults as defaults


def create_metrics(
    task: defaults.SUPPORTED_TASKS_TYPE,
    additional_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
) -> torchmetrics.MetricCollection:
    """Creates task-specific metrics including all additional metrics.

    Args:
        task: Whether classification or regression, for now.
        additional_metrics: Dictionary of additionally metrics to be added to the returned
            `MetricCollection`.
    """
    if task == "classification":
        metric_collection = {
            "accuracy": torchmetrics.Accuracy(),
        }
    elif task == "regression":
        metric_collection = {"mean_squared_error": torchmetrics.MeanSquaredError()}
    else:
        raise NotImplementedError(f"Task {task} not implemented.")
    if additional_metrics:
        assert set(metric_collection).isdisjoint(set(additional_metrics)), (
            "Use a different name for your custom metrics. Following names are reserved for the "
            f"default metrics: {set(metric_collection)}."
        )
        metric_collection.update(additional_metrics)
    return torchmetrics.MetricCollection(metric_collection)
