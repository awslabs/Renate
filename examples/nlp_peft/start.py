# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from renate.training import run_training_job

### THIS ARE JUST  HYPERPARAMETERS. 
config_space = {
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 0.0,
    "learning_rate": 0.001,
    "batch_size": 32,
}
#####################


if __name__ == "__main__":
    ### IGNORE THE FIRST LINES 19-30
    run_training_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="ER",  # we train with Experience Replay
        max_epochs=10,
        config_file="examples/nlp_peft/renate_peft_config.py",
        chunk_id=0,
        output_state_url = "testout",
        backend="local",  # run on SageMaker, select "local" to run this locally
        devices=4,
        strategy="deepspeed_stage_3"
        # strategy="fsdp_native"
    )