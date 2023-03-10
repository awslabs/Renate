# before merging, pull @marwistu's updated version and use that instead of this file

from renate.benchmark.experimentation import execute_experiment_job, experiment_config_file


def to_dense_str(value):
    return str(value).replace(" ", "")


config_space = {
    "updater": "DER",
    "optimizer": "SGD",
    "momentum": 0.0,
    "weight_decay": 0.0,
    "learning_rate": 0.03,
    "alpha": 0.2,
    "beta": 0.5,
    "batch_size": 32,
    "memory_batch_size": 32,
    "memory_size": 500,
    "max_epochs": 5,
    "loss_normalization": 0,
    "loss_weight": 1.0,
    "model_name": "ResNet18CIFAR",
    "scenario_name": "ClassIncrementalScenario",
    "dataset_name": "CIFAR10",
    "val_size": 0.95,
    "class_groupings": to_dense_str([[0, 1], [2, 3]]),
}

execute_experiment_job(
    backend="local",
    config_file=experiment_config_file(),
    config_space=config_space,
    experiment_outputs_url=f"results/",
    mode="max",
    metric="val_accuracy",
    num_updates=2,
    # seed=seed,
)
