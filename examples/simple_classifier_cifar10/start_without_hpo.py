# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role

import renate
from renate.tuning import execute_tuning_job

config_space = {
    "optimizer": "SGD",
    "momentum": 0.0,
    "weight_decay": 0.0,
    "learning_rate": 0.1,
    "alpha": 0.5,
    "batch_size": 32,
    "memory_batch_size": 32,
    "memory_size": 300,
    "loss_normalization": 0,
    "loss_weight": 0.5,
}

if __name__ == "__main__":

    execute_tuning_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="ER",
        max_epochs=50,
        chunk_id=1,
        config_file="./split_cifar10.py",
        requirements_file=str(Path(renate.__path__[0]).resolve().parents[1] / "requirements.txt"),
        backend="sagemaker",
        role=get_execution_role(),
        instance_type="ml.g4dn.2xlarge",
        job_name="testjob",
    )
