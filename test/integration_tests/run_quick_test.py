# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

import boto3
import pandas as pd
import pytest

from renate.utils.file import upload_file_to_s3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        f"--test-file",
        type=str,
        required=True,
        help="Test suite to run.",
    )
    parser.add_argument(
        f"--commit-id",
        type=str,
        required=True,
        help="Provide the commit ID you are testing for reference.",
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
    job_name = f"{test_config['job_name']}-{args.commit_id}"
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
    num_updates = len(test_config["expected_accuracy"])
    result_file = (
        Path("tmp")
        / "renate-integration-tests"
        / test_suite
        / job_name
        / "logs"
        / f"metrics_summary_update_{num_updates - 1}.csv"
    )
    if result_file.exists():
        df = pd.read_csv(result_file)
        accuracies = [float(acc) for acc in list(df.iloc[-1])[1:]]
        results = pd.DataFrame(data={f"Update {i+1}": [acc] for i, acc in enumerate(accuracies)})
        results["All Results"] = None
        results["All Results"].astype("object")
        results.at[0, "All Results"] = df.iloc[:, 1:].to_numpy().tolist()
    else:
        accuracies = []
        results = pd.DataFrame(data={f"Update {i+1}": [pd.NA] for i in range(num_updates)})
    results["Timestamp"] = datetime.datetime.now()
    results["Commit"] = args.commit_id
    results["Test Suite"] = test_suite
    results["Test Name"] = test_config["job_name"]

    local_results_file = (
        Path("tmp") / "renate-integration-tests" / test_suite / job_name / f"{job_name}.csv"
    )
    local_results_file.parent.mkdir(exist_ok=True, parents=True)
    results.to_csv(local_results_file, index=False)
    aws_account_id = boto3.client("sts").get_caller_identity().get("Account")
    upload_file_to_s3(
        local_results_file,
        f"sagemaker-us-west-2-{aws_account_id}",
        f"renate-integration-tests/{test_suite}/{job_name}.csv",
    )
    assert pytest.approx(test_config["expected_accuracy"]) == accuracies
