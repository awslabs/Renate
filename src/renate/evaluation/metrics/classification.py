# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List


def average_accuracy(
    results: Dict[str, List[List[float]]], task_id: int, num_instances: List[int]
) -> float:
    """Compute the average accuracy of a model.

    This measure is defined by:

    .. math::
        \\frac{1}{T}  sum_{i=1}^T  a_{T,i}

    where :math:`T` is the number of tasks, :math:`a_{T,i}` is the accuracy
    of the model on task :math:`i`, while having learned all tasks up to :math:`T`.

    Args:
        results: The results dictionary holding all the results with respect to all recorded
            metrics.
        task_id: The task index.
        num_instances: Count of test data points for each update.
    """
    return sum(results["accuracy"][task_id][: task_id + 1]) / (task_id + 1)


def micro_average_accuracy(
    results: Dict[str, List[List[float]]], task_id: int, num_instances: List[int]
) -> float:
    """Compute the micro average accuracy of a model.

    This measure is defined by the number of correctly classified data points divided by the
    total number of data points. If the number of data points is the same in each update step,
    this is the same as ``average_accuracy``.

    Args:
        results: The results dictionary holding all the results with respect to all recorded
            metrics.
        task_id: The task index.
        num_instances: Count of test data points for each update.
    """
    total_num_instances = sum(num_instances[: task_id + 1])
    return sum(
        [
            num_instances[i] / total_num_instances * results["accuracy"][task_id][i]
            for i in range(task_id + 1)
        ]
    )


def forgetting(
    results: Dict[str, List[List[float]]], task_id: int, num_instances: List[int]
) -> float:
    """Compute the forgetting measure of the model.

    This measure is defined by:

    .. math::
        \\frac{1}{T-1}  sum_{i=1}^{T-1}  f_{T,i}

    where :math:`f_{j,i}` is defined as:

    .. math::
        f_{j,i} = \\max_{k\\in\\{1, \\ldots, j=i\\}} a_{k,i} - a_{j,i}

    where :math:`T` is the final task index, :math:`a_{n,i}` is the test classification accuracy
    on task :math:`i` after sequentially learning the nth task and :math:`f_{j,i}` is a measure of
    forgetting on task :math:`i` after training up to task :math:`j`.

    Args:
        results: The results dictionary holding all the results with respect to all recorded
            metrics.
        task_id: The task index.
        num_instances: Count of test data points for each update.
    """
    if task_id == 0:
        return 0.0

    def f(results: List[List[float]], j: int, i: int) -> float:
        """A Helper function to compute the: math:`f_{j,i}`."""
        accuracy_ji = results[j][i]
        max_accuracy_ki = max([results[k][i] for k in range(j)])
        return max_accuracy_ki - accuracy_ji

    sum_f = 0.0
    for i in range(task_id):
        sum_f += f(results["accuracy"], task_id, i)
    return sum_f / task_id


def backward_transfer(
    results: Dict[str, List[List[float]]], task_id: int, num_instances: List[int]
) -> float:
    """Compute the backward transfer measure of the model.

    This measure is defined by:

    .. math::
        \\frac{1}{T-1}  sum_{i=1}^{T-1}  a_{T,i} - a_{i,i}

    where :math:`T` is the final task index, :math:`a_{n,i}` is the test classification accuracy
    on task :math:`i` after sequentially learning the nth task.

    Args:
        results: The results dictionary holding all the results with respect to all recorded
            metrics.
        task_id: The task index.
        num_instances: Count of test data points for each update.
    """
    if task_id == 0:
        return 0.0
    return (
        sum([results["accuracy"][task_id][i] - results["accuracy"][i][i] for i in range(task_id)])
        / task_id
    )


def forward_transfer(
    results: Dict[str, List[List[float]]], task_id: int, num_instances: List[int]
) -> float:
    """Compute the forward transfer measure of the model.

    This measure is defined by:

    .. math::
        \\frac{1}{T-1}  sum_{i=2}^{T}  a_{i-1,i} - b_{i}

    where :math:`T` is the final task index, :math:`a_{n,i}` is the test classification accuracy
    on task :math:`i` after sequentially learning the nth task. :math:`b_{i}` is the accuracy for
    all :math:`T` tasks recorded at initialisation prior to observing any task.

    Args:
        results: The results dictionary holding all the results with respect to all recorded
            metrics.
        task_id: The task index.
        num_instances: Count of test data points for each update.
    """
    if task_id == 0:
        return 0.0
    return (
        sum(
            [
                results["accuracy"][i - 1][i] - results["accuracy_init"][0][i]
                for i in range(1, task_id + 1)
            ]
        )
        / task_id
    )
