# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role

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

    AWSID = "387922178948"
    AWSREGION = "us-west-2"

    execute_tuning_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="ER",
        max_epochs=50,
        chunk_id=0,
        config_file="renate_config.py",
        requirements_file="requirements.txt",
        next_state_url=f"s3://sagemaker-{AWSREGION}-{AWSID}/renate-training-test-cifar10/",
        backend="sagemaker",
        role=get_execution_role(),
        instance_type="ml.g4dn.2xlarge",
        job_name="testjob",
    )
