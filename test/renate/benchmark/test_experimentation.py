# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from renate.benchmark.experimentation import execute_experiment_job


def test_execute_experiment_job(tmpdir):
    execute_experiment_job(
        backend="local",
        config_file=str(Path(__file__).parent.parent / "renate_config_files" / "config.py"),
        config_space={"updater": "ER", "data_module_fn_use_scenario": "True", "max_epochs": 5},
        experiment_outputs_url=tmpdir,
        mode="max",
        metric="val_accuracy",
        num_updates=2,
        max_time=15,
        seed=0,
    )
