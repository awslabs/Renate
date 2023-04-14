# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import boto3
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role

from renate.training import run_training_job

config_space = {
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 0.0,
    "learning_rate": 0.001,
    "batch_size": 32,
}

if __name__ == "__main__":
    AWS_ID = boto3.client("sts").get_caller_identity().get("Account")
    AWS_REGION = "us-west-2"  # use your AWS preferred region here

    run_training_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="ER",  # we train with Experience Replay
        max_epochs=10,
        config_file="examples/nlp_peft/renate_peft_config.py",
        # For this example, we can train on two binary movie review datasets: "rotten_tomatoes" and
        # "imdb". Set chunk_id to [0, 1] to switch between the two.
        chunk_id=0,
        output_state_url = "testout",
        backend="local",  # run on SageMaker, select "local" to run this locally
        devices=4,
        strategy="deepspeed_stage_3_offload"
    )