# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This script is used to launch an experiment on SageMaker. Previously passed arguments on a different
machine are loaded from S3 and the update process is started with these parameters. Arguments
expected are described in renate.benchmark.execute_experiment_job.
"""
import argparse

import renate.defaults as defaults
from renate.benchmark.experimentation import execute_experiment_job


class ExperimentCLI:
    """Entry point to perform an experiment from start to end including evaluation.

    Given a dataset, optionally wrapped in a scenario, a model and a training configuration,
    this script will find the best hyper-parameters per each data chunk, update the model and
    evaluate it on all the provided test data.
    """

    def run(self):
        parser = argparse.ArgumentParser()
        argument_group = parser.add_argument_group("Required Parameters")

        argument_group.add_argument(
            "--config_file",
            type=str,
            required=True,
            help="File containing the definition of model_fn, data_module_fn, config_space_fn and "
            "scheduler_fn.",
        )
        argument_group.add_argument(
            "--experiment_outputs_url",
            type=str,
            required=True,
            help="Location where to store the experimental results and the final model and state.",
        )
        argument_group.add_argument(
            "--num_updates",
            type=int,
            required=True,
            help="How many updates or chunk IDs should be passed to the updater to update the "
            "model.",
        )
        argument_group.add_argument(
            "--mode",
            choices=defaults.SUPPORTED_TUNING_MODE,
            required=True,
            help=f"Declares the type of optimization problem: {defaults.SUPPORTED_TUNING_MODE}",
        )
        argument_group.add_argument(
            "--metric",
            type=str,
            required=True,
            help="Name of metric to optimize.",
        )

        argument_group = parser.add_argument_group("Syne Tune Parameters")
        argument_group.add_argument(
            "--max_time",
            type=float,
            default=None,
            help="Stopping criterion: wall clock time.",
        )
        argument_group.add_argument(
            "--max_num_trials_started",
            type=int,
            default=None,
            help="Stopping criterion: trials started.",
        )
        argument_group.add_argument(
            "--max_num_trials_completed",
            type=int,
            default=None,
            help="Stopping criterion: trials completed.",
        )
        argument_group.add_argument(
            "--max_num_trials_finished",
            type=int,
            default=None,
            help="Stopping criterion: trials finished.",
        )
        argument_group.add_argument(
            "--n_workers",
            type=int,
            default=defaults.N_WORKERS,
            help=f"Number of workers running in parallel. Default: {defaults.N_WORKERS}.",
        )
        parser.add_argument(
            "--working_directory",
            type=str,
            default=defaults.WORKING_DIRECTORY,
            help="Folder used by Renate to store files temporarily. Default: "
            f"{defaults.WORKING_DIRECTORY}.",
        )

        argument_group = parser.add_argument_group("Optional Parameters")
        argument_group.add_argument(
            "--max_epochs",
            type=int,
            default=defaults.MAX_EPOCHS,
            help=f"Maximum number of (finetuning-equiv.) epochs. Default: {defaults.MAX_EPOCHS}",
        )
        argument_group.add_argument(
            "--seed",
            type=int,
            default=defaults.SEED,
            help=f"Seed used for this job. Default: {defaults.SEED}.",
        )
        argument_group.add_argument(
            "--accelerator",
            type=str,
            default=defaults.ACCELERATOR,
            help=f"Accelerator used for this job. Default: {defaults.ACCELERATOR}.",
        )
        argument_group.add_argument(
            "--devices",
            type=int,
            default=defaults.DEVICES,
            help=f"Devices used for this job. Default: {defaults.DEVICES} device.",
        )
        argument_group.add_argument(
            "--backend",
            type=str,
            default="local",
            help=f"Backend to use for the experiment. Can be either {defaults.SUPPORTED_BACKEND}.",
        )
        argument_group = parser.add_argument_group("SageMaker Parameters")
        argument_group.add_argument(
            "--role",
            type=str,
            help="An AWS IAM role (either name or full ARN)",
        )
        argument_group.add_argument(
            "--instance_type",
            type=str,
            default=defaults.INSTANCE_TYPE,
            help="Type of SageMaker instance to use for training, for example, 'ml.c4.xlarge'.",
        )
        argument_group.add_argument(
            "--instance_count",
            type=int,
            default=defaults.INSTANCE_COUNT,
            help="Number of SageMaker instances to use for training.",
        )
        argument_group.add_argument(
            "--instance_max_time",
            type=float,
            default=defaults.INSTANCE_MAX_TIME,
            help="Maximum time in seconds to wait for an instance to be available.",
        )
        argument_group.add_argument(
            "--requirements_file",
            type=str,
            default="./requirements.txt",
            help="File containing the requirements to install Renate.",
        )
        argument_group.add_argument(
            "--job_name",
            type=str,
            default=None,
            help="Name of the SageMaker job.",
        )

        argument_group.add_argument(
            "--deterministic_trainer",
            type=str,
            default=str(defaults.DETERMINISTIC_TRAINER),
            choices=["True", "False"],
            help="When True forces the trainer to be deterministic. Default: "
            f"{defaults.DETERMINISTIC_TRAINER}.",
        )

        args = parser.parse_args()

        execute_experiment_job(
            **vars(args),
        )


if __name__ == "__main__":
    ExperimentCLI().run()
