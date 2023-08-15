# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
from pathlib import Path

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
    parser.add_argument(
        "--benchmark-file",
        type=str,
        required=True,
        help="Test suite to run.",
    )
    parser.add_argument(
        f"--backend",
        type=str,
        required=True,
        choices=["local", "sagemaker"],
        help="Whether to run locally or on SageMaker.",
    )
    parser.add_argument(
        f"--budget-factor",
        type=float,
        required=True,
        help="Use budget_factor * N many fine-tuning epochs, "
        "where N is the default number of epochs.",
    )
    parser.add_argument(
        f"--job-name",
        type=str,
        required=True,
        help="Name of the job.",
    )
    parser.add_argument(
        f"--num-repetitions",
        type=int,
        default=1,
        help="Number of runs with different seeds.",
    )
    parser.add_argument(
        f"--max-time",
        type=int,
        default=5 * 24 * 3600,
        help="Maximum execution time.",
    )
    parser.add_argument(
        "--use-prior-task-weights",
        action="store_true",
        help="Do not reset model weights (Joint and GDumb only)",
    )

    args = parser.parse_args()
    current_folder = Path(os.path.dirname(__file__))
    configs_folder = current_folder / "experiment_configs"
    benchmark_file = configs_folder / args.benchmark_file
    if not benchmark_file.is_file():
        raise FileNotFoundError(f"Unknown benchmark file '{benchmark_file}'.")
    with open(benchmark_file) as f:
        benchmark_config = json.load(f)
    config_space = load_config(
        os.path.join(configs_folder, "scenarios", benchmark_config["scenario"]),
        os.path.join(configs_folder, "models", benchmark_config["model"]),
        os.path.join(configs_folder, "updaters", benchmark_config["updater"]),
        os.path.join(configs_folder, "datasets", benchmark_config["dataset"]),
    )
    if args.use_prior_task_weights:
        if config_space["updater"] in ["Joint", "GDumb"]:
            config_space["reset"] = False
        else:
            raise ValueError("Please use `use-prior-task-weights` only for Joint or GDumb.")
    config_space["max_epochs"] = int(args.budget_factor * config_space["max_epochs"])
    if "learning_rate_scheduler_step_size" in config_space:
        config_space["learning_rate_scheduler_step_size"] = int(
            args.budget_factor * config_space["learning_rate_scheduler_step_size"]
        )
    if "learning_rate_scheduler_t_max" in config_space:
        config_space["learning_rate_scheduler_t_max"] = int(
            args.budget_factor * config_space["learning_rate_scheduler_t_max"]
        )
    current_folder = Path(os.path.dirname(__file__))

    role = None if args.backend == "local" else get_execution_role()

    for seed in range(args.num_repetitions):
        if args.backend == "local":
            experiment_outputs_url = (
                Path("tmp") / "renate-integration-tests" / args.job_name / str(seed)
            )
            working_directory = str(Path("tmp") / "renate_working_dir")
        else:
            AWS_ACCOUNT_ID = boto3.client("sts").get_caller_identity().get("Account")
            experiment_outputs_url = (
                f"s3://sagemaker-us-west-2-{AWS_ACCOUNT_ID}/renate-domain-incremental/"
                f"{args.job_name}/{seed}"
            )
            working_directory = "/tmp/renate_working_dir"
        execute_experiment_job(
            backend=args.backend,
            config_file=experiment_config_file(),
            config_space=config_space,
            experiment_outputs_url=experiment_outputs_url,
            working_directory=working_directory,
            mode="max",
            metric="val_accuracy",
            num_updates=config_space["num_tasks"],
            role=role,
            instance_type="ml.g4dn.xlarge",
            n_workers=1,
            max_time=args.max_time,
            instance_max_time=args.max_time,
            seed=seed,
            job_name=args.job_name[:36],
            devices=1,
            strategy="ddp",
            save_state=False,
            requirements_file=str(current_folder / "requirements.txt"),
        )
