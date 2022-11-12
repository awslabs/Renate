# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from conftest import SAMPLE_CLASSIFICATION_RESULTS
from renate.evaluation.metrics.classification import (
    average_accuracy,
    backward_transfer,
    forgetting,
    forward_transfer,
)


@pytest.mark.parametrize(
    "task_id,result",
    [
        [0, 0.9362000226974487],
        [1, sum([0.8284000158309937, 0.9506999850273132]) / 2],
        [2, sum([0.4377000033855438, 0.48260000348091125, 0.9438999891281128]) / 3],
    ],
)
def test_average_accuracy(task_id, result):
    results = SAMPLE_CLASSIFICATION_RESULTS
    assert pytest.approx(average_accuracy(results, task_id)) == result


@pytest.mark.parametrize(
    "task_id,result",
    [
        [0, 0.0],
        [1, (1 / 1) * (0.9362000226974487 - 0.8284000158937)],
        [
            2,
            (1 / 2)
            * sum(
                [0.9362000226974487 - 0.4377000033855438, 0.9506999850273132 - 0.48260000348091125]
            ),
        ],
    ],
)
def test_forgetting(task_id, result):
    results = SAMPLE_CLASSIFICATION_RESULTS
    assert pytest.approx(forgetting(results, task_id)) == result


@pytest.mark.parametrize(
    "task_id,result",
    [
        [0, 0.0],
        [1, (1 / 1) * (0.8284000158309937 - 0.9362000226974487)],
        [
            2,
            (1 / 2)
            * sum(
                [0.4377000033855438 - 0.9362000226974487, 0.48260000348091125 - 0.9506999850273132]
            ),
        ],
    ],
)
def test_backward_transfer(task_id, result):
    results = SAMPLE_CLASSIFICATION_RESULTS
    assert pytest.approx(backward_transfer(results, task_id)) == result


@pytest.mark.parametrize(
    "task_id,result",
    [
        [0, 0.0],
        [1, (1 / 1) * (0.6093000173568726 - 0.1)],
        [
            2,
            (1 / 2) * sum([0.6093000173568726 - 0.1, 0.3382999897003174 - 0.09]),
        ],
    ],
)
def test_forward_transfer(task_id, result):
    results = SAMPLE_CLASSIFICATION_RESULTS
    assert pytest.approx(forward_transfer(results, task_id)) == result
