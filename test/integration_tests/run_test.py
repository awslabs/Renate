# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Test suite to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed.",
    )
    parser.add_argument(
        f"--requirements-file",
        type=str,
        required=False,
        help="Path to requirements file",
    )
    args = parser.parse_args()
    test_suite = "main"
    current_folder = Path(os.path.dirname(__file__))
    configs_folder = current_folder / "configs"
    test_file = configs_folder / "suites" / test_suite / args.test_file
    if not test_file.is_file():
        raise FileNotFoundError(f"Unknown test file '{test_file}'.")
    with open(test_file) as f:
        test_config = json.load(f)
    job_name = f"{test_config['job_name']}"
    requirements_file = args.requirements_file
    if not requirements_file:
        requirements_file = current_folder.parent.parent / "requirements.txt"
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
            "--seed",
            str(args.seed),
            "--requirements-file",
            requirements_file,
        ]
    )
    process.wait()
