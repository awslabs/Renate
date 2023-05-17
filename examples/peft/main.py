# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from renate.training import run_training_job

config_space = {
    # Training hyperparameters
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 0.0,
    "learning_rate": 0.01,
    "batch_size": 32,
    # Model
    "pretrained_model_name": "bert-base-uncased",
    "peft_type": "lora",
    # Dataset
    "dataset_name": "rotten_tomatoes",
    "input_column": "text",
    "target_column": "label",
    "num_outputs": 2,
}

if __name__ == "__main__":
    # AWS_ID = boto3.client("sts").get_caller_identity().get("Account")
    # AWS_REGION = "us-west-2"  # use your AWS preferred region here

    run_training_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="FineTuning",  # we train with Experience Replay
        max_epochs=5,
        config_file="renate_config.py",
        output_state_url="tmp",  # f"s3://sagemaker-{AWS_REGION}-{AWS_ID}/renate-peft/",
        backend="local",
        # role=get_execution_role(),
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        job_name="renate-peft",
    )
