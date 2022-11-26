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

    AWS_ID = "0123456789"  # use your AWS account id here
    AWS_REGION = "us-west-2"  # use your AWS preferred region here

    execute_tuning_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="ER",  # we train with Experience Replay
        max_epochs=50,
        chunk_id=0,  # we select the first chunk of our dataset, you will probably not need this in practice
        config_file="renate_config.py",
        requirements_file="requirements.txt",
        # replace the url below with a different one if you already ran it and you want to avoid overwriting
        next_state_url=f"s3://sagemaker-{AWS_REGION}-{AWS_ID}/renate-training-cifar10-1st-model/",
        # uncomment the line below only if you already created a model with this script and you want to update it
        # state_url=f"s3://sagemaker-{AWS_REGION}-{AWS_ID}/renate-training-cifar10-1st-model/",
        backend="sagemaker",  # we will run this on SageMaker, but you can select "local" to run this locally
        role=get_execution_role(),
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        job_name="job_name",
    )
