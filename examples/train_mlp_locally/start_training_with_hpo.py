from renate.tuning import execute_tuning_job
from renate.tuning.config_spaces import config_space

if __name__ == "__main__":
    # we run the first training job on the MNIST classes [0-4]
    execute_tuning_job(
        config_space=config_space("RD"),  # getting the default search space
        mode="max",
        metric="val_accuracy",
        updater="RD",  # use the RepeatedDistillationModelUpdater
        max_epochs=50,
        chunk_id=0,  # this selects the first chunk of the dataset
        config_file="renate_config.py",
        next_state_url="./state_dump_first_model_rd/",  # this is where the model will be stored
        backend="local",  # the training job will run on the local machine
        scheduler="asha",
        # using only 5 trials will not give great performance but this is just an example
        max_num_trials_finished=5,
    )

    # retrieve the model from `./state_dump_first_model_rd/` if you want -- don't delete it

    execute_tuning_job(
        config_space=config_space("RD"),
        mode="max",
        metric="val_accuracy",
        updater="RD",
        max_epochs=50,
        chunk_id=1,  # this time we use the second chunk of the dataset
        config_file="renate_config.py",
        state_url="./state_dump_first_model_rd/",  # the output of the first training job is loaded
        next_state_url="./state_dump_second_model_rd/",  # location where the new model is saved to
        backend="local",
        scheduler="asha",
        max_num_trials_finished=5,
    )
