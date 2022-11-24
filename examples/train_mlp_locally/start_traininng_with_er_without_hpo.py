from renate.tuning import execute_tuning_job

configuration = {
    "optimizer": "SGD",
    "momentum": 0.0,
    "weight_decay": 1e-2,
    "learning_rate": 0.05,
    "batch_size": 32,
    "max_epochs": 50,
    "memory_batch_size": 32,
    "memory_size": 500,
}

if __name__ == "__main__":

    execute_tuning_job(
        config_space=configuration,  # getting the default search space
        mode="max",
        metric="val_accuracy",
        updater="ER",  # use the RepeatedDistillationModelUpdater
        max_epochs=50,
        chunk_id=0,
        config_file="renate_config.py",
        next_state_url="./output_folder/",  # this is where the model will be stored
        backend="local",  # the training job will run on the local machine
    )
