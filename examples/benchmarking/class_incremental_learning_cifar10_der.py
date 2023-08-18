# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from renate.benchmark.experimentation import execute_experiment_job, experiment_config_file


config_space = {
    "updater": "DER",
    "optimizer": "SGD",
    "momentum": 0.0,
    "weight_decay": 0.0,
    "learning_rate": 0.03,
    "alpha": 0.2,
    "beta": 0.5,
    "batch_size": 64,
    "batch_memory_frac": 0.5,
    "memory_size": 500,
    "max_epochs": 50,
    "loss_normalization": 0,
    "loss_weight": 1.0,
    "model_name": "ResNet18CIFAR",
    "scenario_name": "ClassIncrementalScenario",
    "dataset_name": "CIFAR10",
    "val_size": 0,
    "groupings": ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
    "num_outputs": 10,
}

for seed in range(10):
    execute_experiment_job(
        backend="local",
        config_file=experiment_config_file(),
        config_space=config_space,
        experiment_outputs_url=f"results/{seed}/",
        mode="max",
        metric="val_accuracy",
        num_updates=5,
        seed=seed,
    )
