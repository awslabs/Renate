# Renate - Automatic Neural Networks retraining and Continual Learning in Python

Renate is a Python package for automatic neural networks models retraining using
continual learning and lifelong learning algorithms. The package is based on [PyTorch](https://pytorch.org),
and [PyTorch Lightning](https://www.pytorchlightning.ai/).
It also leverages [SyneTune](https://github.com/awslabs/syne-tune) for hyperparameters optimization (HPO) and neural architecture search (NAS).


## Who needs Renate?
In many applications data are made available over time and retraining from scratch for
every new batch of data is prohibitively expensive. In these cases we would like to use
the new batch of data provided to update our previous model.
Unfortunately, since data in different chunks are not sampled according to the same distribution,
this creates a number of problems (e.g., the so-called "catastrophic forgetting").

This is a simple example taking ...

IMAGE

...

If you **can** afford retraining your model from scratch but you cannot afford a complete
retuning of all the hyperparamters and network architecture, we got you covered!
You can either follow [this example]() using Renate or leverage directly the algorithms
we put in [Syne Tune](https://github.com/awslabs/syne-tune).


## Installation

Renate requires Python 3.9 or newer, and the easiest way to install it is via `pip`:

```bash
pip install renate
```
## A Simple Example

In this example we will update a ResNet model trained on the first five classes of CIFAR10 using the
remaining ones. For this purpose we will split the dataset in two chunks, grouping the first five
classes in the first and the remaining in the second. This will not be necessarily in real-world
scenarios since in most cases the training will happen on all the data available in different points
in time, but you can simplify the example code following the comments.

Crete a file defining your model and dataset.
For this example we created one for you called `split_cifar10.py` in the folder `examples/simple_classifier_cifar10/`.

The main blocks are:

1. The **model function**. This function should be able to instantiate a new model or load
a model from the URL provided. In this cases we will use a simple ResNet model.
```py
def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    """Returns a model instance."""
    if model_state_url is None:
        model = ResNet18CIFAR()
    else:
        state_dict = torch.load(str(model_state_url))
        model = ResNet18CIFAR.from_state_dict(state_dict)
    return model
```

2. The **data module function**. This function is loading the data and creating the dataset. For this example we will
use CIFAR10 from torchvision, but you can load any dataset you like. We also add the `ClassIncrementalScenario` to
split the dataset in to chunks using the class id, but this is only for demonstrational purposes and can be removed.
```py
def data_module_fn(
    data_path: Union[Path, str], chunk_id: int, seed: int = defaults.SEED
) -> ClassIncrementalScenario:
    """Returns a class-incremental scenario instance.

    The transformations passed to prepare the input data are required to convert the data to
    PyTorch tensors.
    """
    data_module = TorchVisionDataModule(
        str(data_path),
        dataset_name="CIFAR10",
        download=True,
        val_size=0.2,
        seed=seed,
    )
    class_incremental_scenario = ClassIncrementalScenario(
        data_module=data_module,
        num_tasks=2,
        class_groupings=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        chunk_id=chunk_id,
    )
    return class_incremental_scenario
```

3. The **transformations** to be applied to the dataset are defined in ad-hoc functions. One function for the training set,
one for the test set (since some information may not be available at inference time) and an optional one to be used on the
memory buffer for the methods that actually have one (e.g., Experience Replay).
```py
def train_transform() -> transforms.Compose:
    """Returns a transform function to be used in the training."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
        ]
    )


def test_transform() -> transforms.Compose:
    """Returns a transform function to be used for validation or testing."""
    return transforms.Compose(
        [
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
        ]
    )
```

We also created a script to launch our training job. You can see some examples in the folder `examples/simple_classifier_cifar10/`.
The scripts have two main components: a configuration, eventually containing also the hyperparameters ranges for HPO,
and a main function launching the training or job.

The following is a simple search space for HPO:
```py
config_space = {
    "updater": "ER",
    "optimizer": "SGD",
    "momentum": uniform(0.1, 0.9),
    "weight_decay": 0.0,
    "learning_rate": loguniform(1e-4, 1e-1),
    "alpha": uniform(0.0, 1.0),
    "batch_size": choice([32, 64, 128, 256]),
    "memory_batch_size": 32,
    "memory_size": 1000,
    "max_epochs": 50,
    "loss_normalization": 0,
    "loss_weight": uniform(0.0, 1.0),
}
```
If HPO is not required, the config space can be specified using exact values instead of range. For example,
`"momentum: 0.25"` instead of `"momentum": uniform(0.1, 0.9)`.

While the main function can be similar to the following one:
```py
if __name__ == "__main__":

    execute_tuning_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="ER",  # we train with Experience Replay
        chunk_id=0,  # we select the first chunk of our dataset, you will probably not need this in practice
        source_dir="../../cli/",
        model_data_definition="./split_cifar10.py",
        requirements_file="../../../../requirements.txt",
        backend="sagemaker",  # we will run this on SageMaker, but you can select "local" to run this locally
        role="arn:aws:iam::MYAWSID:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole",
        instance_count=1,
        instance_type="ml.g4dn.2xlarge",
        max_num_trials_finished=100,
        scheduler="asha",  # we will run ASHA to optimize our hyerparameters
        n_workers=4,
        job_name="testjob",
    )
```

### More Examples
More examples are available in the `/examples` folder.

## Contributing

If you wish to contribute to the project, please refer to our
[contribution guidelines](https://github.com/awslabs/renate/tree/master/CONTRIBUTING.md).


## Documentation
> TODO: fix link below
* [API documentation](https://X)
