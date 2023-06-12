# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import boto3
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role

from renate.training import run_training_job

training_config = {
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 0.0,
    "learning_rate": 0.01,
    "batch_size": 32,
}
model_config = {
    "pretrained_model_name": "bert-base-uncased",
    "peft_type": "lora",
    "lora_alpha": 8,
    "lora_dropout": 0.1,
}
data_config = {
    "dataset_name": "rotten_tomatoes",
    "input_column": "text",
    "target_column": "label",
    "num_outputs": 2,
}

config_space = {**training_config, **model_config, **data_config}

if __name__ == "__main__":
    AWS_ID = boto3.client("sts").get_caller_identity().get("Account")
    AWS_REGION = "us-west-2"  # use your AWS preferred region here

    run_training_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="FineTuning",
        max_epochs=5,
        config_file="renate_config.py",
        output_state_url=f"s3://sagemaker-{AWS_REGION}-{AWS_ID}/renate-peft/",
        backend="local",
        role=get_execution_role(),
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        job_name="renate-peft",
    )
