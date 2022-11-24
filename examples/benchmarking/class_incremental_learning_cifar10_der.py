from renate.benchmark.experimentation import execute_experiment_job, experiment_config_file


def to_dense_str(value):
    return str(value).replace(" ", "")


dataset_name = "CIFAR10"
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
    "max_epochs": 50,
    "loss_normalization": 0,
    "loss_weight": 1.0,
    "model_fn_model_name": "ResNet18CIFAR",
    "data_module_fn_scenario_name": "class_incremental",
    "data_module_fn_dataset_name": dataset_name,
    "data_module_fn_val_size": 0,
    "data_module_fn_class_groupings": to_dense_str([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
    "transform_dataset_name": dataset_name,
}

execute_experiment_job(
    backend="local",
    config_file=experiment_config_file(),
    config_space=config_space,
    experiment_outputs_url="results/",
    mode="max",
    metric="val_accuracy",
    num_updates=5,
    seed=0,
)
