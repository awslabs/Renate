# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import subprocess
from pathlib import Path
from sys import platform

import pandas as pd
import pytest

from renate import defaults

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        f"--test-file",
        type=str,
        required=True,
        help="Test suite to run.",
    )
    args = parser.parse_args()
    test_suite = "quick"
    current_folder = Path(os.path.dirname(__file__))
    configs_folder = current_folder / "configs"
    test_file = configs_folder / "suites" / test_suite / args.test_file
    if not test_file.is_file():
        raise FileNotFoundError(f"Unknown test file '{test_file}'.")
    with open(test_file) as f:
        test_config = json.load(f)
    job_name = f"{test_config['job_name']}-{defaults.current_timestamp()}"
    process = subprocess.Popen(
        [
            "python",
            current_folder / "run_experiment.py",
            "--scenario-file",
            configs_folder / "scenarios" / test_config["scenario"],
            "--model-file",
            configs_folder / "models" / test_config["model"],
            "--updater-file",
            configs_folder / "updaters" / test_config["updater"],
            "--dataset-file",
            configs_folder / "datasets" / test_config["dataset"],
            "--backend",
            test_config["backend"],
            "--job-name",
            job_name,
            "--test-suite",
            test_suite,
        ]
    )
    process.wait()
    expected_accuracy = test_config[f"expected_accuracy_{platform}"]
    num_updates = len(test_config["expected_accuracy_darwin"][0])
    result_file = (
        Path("tmp")
        / "renate-integration-tests"
        / test_suite
        / job_name
        / "0"
        / "logs"
        / f"metrics_summary_update_{num_updates - 1}.csv"
    )
    if result_file.exists():
        df = pd.read_csv(result_file)
        accuracies = [float(acc) for acc in list(df.iloc[-1])[1:]]
    else:
        accuracies = []
    assert any([pytest.approx(acc) == accuracies for acc in expected_accuracy]), accuracies
