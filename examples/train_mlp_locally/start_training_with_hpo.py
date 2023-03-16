# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from renate.training import run_training_job
from renate.utils.config_spaces import config_space

if __name__ == "__main__":
    # we run the first training job on the MNIST classes [0-4]
    run_training_job(
        config_space=config_space("RD"),  # getting the default search space
        mode="max",
        metric="val_accuracy",
        updater="RD",  # use the RepeatedDistillationModelUpdater
        max_epochs=50,
        chunk_id=0,  # this selects the first chunk of the dataset
        config_file="renate_config.py",
        # location where the new model is saved to
        output_state_url="./state_dump_first_model_rd/",
        backend="local",  # the training job will run on the local machine
        scheduler="asha",
        # using only 5 trials will not give great performance -- just an example
        max_num_trials_finished=5,
    )

    # retrieve the model from `./state_dump_first_model_rd/` if you want
    # do not delete it, we will need it in the remaining of this example

    run_training_job(
        config_space=config_space("RD"),
        mode="max",
        metric="val_accuracy",
        updater="RD",
        max_epochs=50,
        chunk_id=1,  # this time we use the second chunk of the dataset
        config_file="renate_config.py",
        # the output of the first training job is loaded
        input_state_url="./state_dump_first_model_rd/",
        # location where the new model is saved to
        output_state_url="./state_dump_second_model_rd/",
        backend="local",
        scheduler="asha",
        max_num_trials_finished=5,
    )
