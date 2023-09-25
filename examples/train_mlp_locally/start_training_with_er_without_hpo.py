# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from renate.training import run_training_job

configuration = {
    "optimizer": "SGD",
    "momentum": 0.0,
    "weight_decay": 1e-2,
    "learning_rate": 0.05,
    "batch_size": 64,
    "batch_memory_frac": 0.5,
    "max_epochs": 50,
    "memory_size": 500,
}

if __name__ == "__main__":
    run_training_job(
        config_space=configuration,
        mode="max",
        metric="val_accuracy",
        updater="ER",
        max_epochs=50,
        chunk_id=0,
        config_file="renate_config.py",
        output_state_url="./output_folder/",  # this is where the model will be stored
        backend="local",  # the training job will run on the local machine
    )
