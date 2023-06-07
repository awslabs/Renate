# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from renate.benchmark.experimentation import execute_experiment_job, experiment_config_file


config_space = {
    "updater": "Joint",
    "optimizer": "SGD",
    "alpha": 0.2,
    "beta": 0.5,
    "momentum": 0.9,
    "weight_decay": 1e-5,
    "learning_rate": 0.01,
    "batch_size": 256,
    "learning_rate_scheduler": "StepLR",
    "learning_rate_scheduler_step_size": 30,
    "learning_rate_scheduler_gamma": 0.1,
    "max_epochs": 100,
    "model_name": "ResNet18",
    "scenario_name": "BenchmarkScenario",
    "dataset_name": "CIFAR10",
    "num_tasks": 10,
    "val_size": 0.05,
}
num_seeds = 1
for seed in range(num_seeds):
    execute_experiment_job(
        backend="sagemaker",
        config_file=experiment_config_file(),
        config_space=config_space,
        experiment_outputs_url=f"results/{seed}/",
        mode="max",
        metric="val_accuracy",
        num_updates=config_space["num_tasks"],
        seed=seed,
    )
