# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from renate.benchmark.experimentation import execute_experiment_job


@pytest.fixture
def experiment_job_kwargs():
    return {
        "backend": "local",
        "config_file": str(Path(__file__).parent.parent / "renate_config_files" / "config.py"),
        "config_space": {"updater": "ER", "data_module_fn_use_scenario": "True", "max_epochs": 5},
        "mode": "max",
        "metric": "val_accuracy",
        "num_updates": 2,
        "max_time": 15,
        "seed": 0,
    }


def test_execute_experiment_job(tmpdir, experiment_job_kwargs):
    """Only checking if things run, not testing anything besides that."""
    execute_experiment_job(experiment_outputs_url=tmpdir, **experiment_job_kwargs)


@pytest.mark.parametrize(
    "update_dict,regex",
    [
        ({"backend": "UNKNOWN_BACKEND"}, r"Backend*"),
        ({"mode": "UNKNOWN_MODE"}, r"Mode*"),
        ({"num_updates": 3}, r"The dataset has*"),
    ],
)
def test_execute_experiment_job_edge_cases(tmpdir, experiment_job_kwargs, update_dict, regex):
    """Check if basic input errors raise an exception."""
    experiment_job_kwargs.update(update_dict)
    with pytest.raises(AssertionError, match=regex):
        execute_experiment_job(experiment_outputs_url=tmpdir, **experiment_job_kwargs)
