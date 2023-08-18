# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from renate.training import run_training_job


config_space = {
    "optimizer": "SGD",
    "momentum": 0.0,
    "weight_decay": 0.0,
    "learning_rate": 0.1,
    "alpha": 0.5,
    "batch_size": 64,
    "batch_memory_frac": 0.5,
    "memory_size": 500,
    "loss_normalization": 0,
    "loss_weight": 0.5,
    "early_stopping": True,
}

if __name__ == "__main__":
    # we run the first training job on the MNIST classes [0-4]
    run_training_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="ER",
        max_epochs=50,
        chunk_id=0,  # this selects the first chunk of the dataset
        config_file="renate_config.py",
        # this is where the model will be stored
        output_state_url="./state_dump_first_model/",
        # the training job will run on the local machine
        backend="local",
    )

    # retrieve the model from `./state_dump_first_model/` if you want
    # do not delete the model, we are going to use it below

    run_training_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="ER",
        max_epochs=50,
        chunk_id=1,  # this time we use the second chunk of the dataset
        config_file="renate_config.py",
        # the output of the first training job is loaded
        input_state_url="./state_dump_first_model/",
        # the new model will be stored in this folder
        output_state_url="./state_dump_second_model/",
        backend="local",
    )
