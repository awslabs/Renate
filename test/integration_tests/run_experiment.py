# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json


def load_config(scenario_file, model_file, updater_file):
    config_space = {}
    for file in [scenario_file, model_file, updater_file]:
        with open(file) as f:
            config_space.update(json.load(f))
    return config_space


config_space, num_tasks = load_config(benchmark_type, benchmark, updater)
execute_experiment_job(
    backend="sagemaker",
    config_file=experiment_config_file(),
    config_space=config_space,
    experiment_outputs_url=f"s3://sagemaker-us-west-2-{AWS_ACCOUNT_ID}/{S3_OUTPUT_PATH}/{benchmark_type}-{benchmark}-{updater}/{seed}",
    mode="max",
    metric="val_accuracy",
    num_updates=num_tasks,
    role=f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/AmazonSageMakerServiceCatalogProductsUseRole",
    instance_type="ml.g4dn.{}xlarge".format("12" if "hpo" in updater else ""),
    max_time=6 * 3600,
    n_workers=4,
    seed=seed,
    job_name=f"{benchmark}-{updater}-{seed}",
)
