# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
import os

import boto3
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role

from renate.benchmark.experimentation import execute_experiment_job, experiment_config_file


def load_config(scenario_file, model_file, updater_file, dataset_file):
    cs = {}
    for file in [scenario_file, model_file, updater_file, dataset_file]:
        with open(file) as f:
            cs.update(json.load(f))
    return cs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for file_name in ["scenario", "model", "updater", "dataset"]:
        parser.add_argument(
            f"--{file_name}-file",
            type=str,
            required=True,
            help=f"File containing the path to the {file_name} JSON.",
        )
    parser.add_argument(
        f"--backend",
        type=str,
        required=True,
        choices=["local", "sagemaker"],
        help="Whether to run locally or on SageMaker.",
    )
    parser.add_argument(
        f"--job-name",
        type=str,
        required=True,
        help="Name of the job.",
    )
    parser.add_argument(
        f"--test-suite",
        type=str,
        required=True,
        choices=["quick", "main"],
        help="Test suite that is run.",
    )
    parser.add_argument(
        f"--seed",
        type=int,
        default=0,
        help="Seed.",
    )
    parser.add_argument(
        f"--max-time",
        type=int,
        default=12 * 3600,
        help="Maximum execution time.",
    )
    parser.add_argument(
        f"--requirements-file",
        type=str,
        required=False,
        help="Path to requirements file",
    )
    args = parser.parse_args()
    config_space = load_config(
        args.scenario_file, args.model_file, args.updater_file, args.dataset_file
    )
    current_folder = Path(os.path.dirname(__file__))
    requirements_file = args.requirements_file
    if not requirements_file:
        requirements_file = current_folder.parent.parent / "requirements.txt"

    if args.backend == "local":
        experiment_outputs_url = (
            Path("tmp")
            / "renate-integration-tests"
            / args.test_suite
            / args.job_name
            / str(args.seed)
        )
        role = None
    else:
        AWS_ACCOUNT_ID = boto3.client("sts").get_caller_identity().get("Account")
        experiment_outputs_url = (
            f"s3://sagemaker-us-west-2-{AWS_ACCOUNT_ID}/renate-integration-tests/"
            f"{args.test_suite}/{args.job_name}/{args.seed}"
        )
        role = get_execution_role()
    execute_experiment_job(
        backend=args.backend,
        config_file=experiment_config_file(),
        config_space=config_space,
        experiment_outputs_url=experiment_outputs_url,
        mode="max",
        metric="val_accuracy",
        num_updates=config_space["num_tasks"],
        role=role,
        instance_type="ml.g4dn.xlarge",
        max_time=args.max_time,
        seed=args.seed,
        job_name=args.job_name[:36],
        devices=1,
        strategy="ddp",
        requirements_file=args.requirements_file,
    )
