# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torchmetrics

from renate import defaults
from renate.evaluation.metrics.utils import create_metrics


@pytest.mark.parametrize(
    "task",
    defaults.SUPPORTED_TASKS_TYPE.__args__ + ("unsupported_task",),
    ids=defaults.SUPPORTED_TASKS_TYPE.__args__ + ("unsupported_task",),
)
def test_create_metrics_without_additional_metrics(task):
    """Tests all allowed cases and all edge cases when no additional metrics is provided.

    Edge cases: not supported task.
    """
    if task == "unsupported_task":
        with pytest.raises(NotImplementedError):
            create_metrics(task=task)
    else:
        metric_collection = create_metrics(task=task)
        assert metric_collection.prefix is None


@pytest.mark.parametrize(
    "additional_metrics, is_duplicate",
    [
        (
            {
                "auc": torchmetrics.AUC(),
                "calibration_error": torchmetrics.CalibrationError(),
            },
            False,
        ),
        (
            {
                "accuracy": torchmetrics.Accuracy(),
            },
            True,
        ),
    ],
    ids=["valid_metric_names", "duplicate_metric_names"],
)
def test_given_additional_metrics_create_metrics_adds_them_to_metric_collection(
    additional_metrics, is_duplicate
):
    """Passes if duplicate metric names raise an exception and additional metrics are added to the
    metric collection.

    Case 1: Valid case.
    Case 2: Metric name already used as part of the default metric collection.
    """
    if is_duplicate:
        with pytest.raises(AssertionError):
            create_metrics(task="classification", additional_metrics=additional_metrics)
    else:
        metric_collection = create_metrics(
            task="classification", additional_metrics=additional_metrics
        )
        assert metric_collection.prefix is None
        assert set(additional_metrics).issubset(set(metric_collection))
