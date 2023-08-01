# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pandas as pd
import pytest

from renate.benchmark.experimentation import (
    cumulative_metrics_summary,
    execute_experiment_job,
    individual_metrics_summary,
)
from renate.evaluation.metrics.classification import average_accuracy


@pytest.fixture
def experiment_job_kwargs():
    return {
        "backend": "local",
        "config_file": str(
            Path(__file__).parent.parent / "renate_config_files" / "config_scenario.py"
        ),
        "config_space": {"updater": "ER", "max_epochs": 5},
        "mode": "max",
        "metric": "val_accuracy",
        "num_updates": 2,
        "max_time": 30,
        "seed": 0,
        "accelerator": "cpu",
        "devices": 1,
    }


@pytest.mark.parametrize("save_state", (True, False))
def test_execute_experiment_job(tmpdir, experiment_job_kwargs, save_state):
    """Only checking if things run, not testing anything besides that."""
    expected_columns = [
        "Task ID",
        "Average Accuracy",
        "Micro Average Accuracy",
        "Forgetting",
        "Forward Transfer",
        "Backward Transfer",
    ]
    expected_num_updates = experiment_job_kwargs["num_updates"]
    experiment_job_kwargs["save_state"] = save_state
    execute_experiment_job(experiment_outputs_url=tmpdir, **experiment_job_kwargs)
    results_df = pd.read_csv(str(Path(tmpdir) / "logs" / "metrics_summary.csv"))
    assert all(results_df.columns == expected_columns)
    if save_state:
        hpo_file = Path(tmpdir) / f"update_{expected_num_updates - 1}" / "hpo.csv"
        for update_id in range(expected_num_updates):
            assert (Path(tmpdir) / f"update_{update_id}" / "learner.ckpt").is_file()
            assert (Path(tmpdir) / f"update_{update_id}" / "model.ckpt").is_file()
    else:
        hpo_file = Path(tmpdir) / "hpo.csv"
    assert len(pd.read_csv(str(hpo_file))["update_id"].unique()) == expected_num_updates


@pytest.mark.parametrize(
    "update_dict,regex",
    [
        ({"backend": "UNKNOWN_BACKEND"}, r"Backend*"),
        ({"mode": "UNKNOWN_MODE"}, r"Mode*"),
        ({"num_updates": 3}, r"The dataset has*"),
    ],
    ids=["unknown backend", "unknown mode", "wrong number of updates"],
)
def test_execute_experiment_job_edge_cases(tmpdir, experiment_job_kwargs, update_dict, regex):
    """Check if basic input errors raise an exception."""
    experiment_job_kwargs.update(update_dict)
    with pytest.raises(AssertionError, match=regex):
        execute_experiment_job(experiment_outputs_url=tmpdir, **experiment_job_kwargs)


def test_cumulative_metrics_summary():
    results = {
        "accuracy": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        "accuracy_init": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    }
    metrics = [("Average Accuracy", average_accuracy)]
    df = cumulative_metrics_summary(
        results=results,
        cumulative_metrics=metrics,
        num_tasks=3,
        num_instances=[10, 20, 30],
    )
    assert list(df.columns) == ["Task ID", "Average Accuracy"]
    assert pytest.approx(list(df["Average Accuracy"])) == [0.1, 0.45, 0.8]
    assert df.shape == (3, 2)


def test_individual_metrics_summary():
    results = {
        "accuracy": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        "accuracy_init": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    }
    df = individual_metrics_summary(results=results, current_task=2, num_tasks=3)
    assert list(df.columns) == [("Task ID", "")] + [("accuracy", f"Task {i}") for i in range(1, 4)]
    assert list(df.iloc[0]) == [1.0, 0.1, 0.2, 0.3]
    assert list(df.iloc[1]) == [2.0, 0.4, 0.5, 0.6]
    assert df.shape == (2, 4)
