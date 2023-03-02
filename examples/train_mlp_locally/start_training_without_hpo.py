from renate.tuning import execute_tuning_job


config_space = {
    "optimizer": "SGD",
    "momentum": 0.0,
    "weight_decay": 0.0,
    "learning_rate": 0.1,
    "alpha": 0.5,
    "batch_size": 32,
    "memory_batch_size": 32,
    "memory_size": 500,
    "loss_normalization": 0,
    "loss_weight": 0.5,
    "enable_early_stopping": True,
}

if __name__ == "__main__":
    # we run the first training job on the MNIST classes [0-4]
    execute_tuning_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="ER",
        max_epochs=50,
        chunk_id=0,  # this selects the first chunk of the dataset
        config_file="renate_config.py",
        next_state_url="./state_dump_first_model/",  # this is where the model will be stored
        backend="local",  # the training job will run on the local machine
    )

    # retrieve the model from `./state_dump_first_model/` if you want -- don't delete it

    execute_tuning_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="ER",
        max_epochs=50,
        chunk_id=1,  # this time we use the second chunk of the dataset
        config_file="renate_config.py",
        state_url="./state_dump_first_model/",  # the output of the first training job is loaded
        next_state_url="./state_dump_second_model/",  # the new model will be stored in this folder
        backend="local",
    )
