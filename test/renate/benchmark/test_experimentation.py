# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pandas as pd
import pytest

from renate.benchmark.experimentation import execute_experiment_job


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
        "max_time": 15,
        "seed": 0,
    }


def test_execute_experiment_job(tmpdir, experiment_job_kwargs):
    """Only checking if things run, not testing anything besides that."""
    expected_columns = [
        "Task ID",
        "Average Accuracy",
        "Forgetting",
        "Forward Transfer",
        "Backward Transfer",
    ]
    expected_num_updates = experiment_job_kwargs["num_updates"]
    execute_experiment_job(experiment_outputs_url=tmpdir, **experiment_job_kwargs)
    results_df = pd.read_csv(str(Path(tmpdir) / "logs" / "metrics_summary.csv"))
    assert all(results_df.columns == expected_columns)
    for update_id in range(expected_num_updates):
        assert (Path(tmpdir) / f"update_{update_id}" / "learner.ckpt").is_file()
        assert (Path(tmpdir) / f"update_{update_id}" / "model.ckpt").is_file()
    assert (
        len(
            pd.read_csv(str(Path(tmpdir) / f"update_{expected_num_updates - 1}" / "hpo.csv"))[
                "update_id"
            ].unique()
        )
        == expected_num_updates
    )


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
