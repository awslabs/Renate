# How to Write a Config File

User input is passed to Renate via a config file.
It contains the definition of the model you want to train, code to load and preprocess the data,
and (optionally) a set of transforms to be applied to that data.
These components are provided by implementing functions with a fixed name.
When accessing your config file, Renate will inspect it for these functions.

## Model: `model_fn`

This function takes a path to a model state and returns a model in the form of a `RenateModule`.
A `RenateModule` is a `torch.nn.Module` with some additional functionality relevant to continual learning;
for detailed information, see [here](TODO).
If no path is given (i.e., when we first train a model) the model should be created from scratch,
otherwise it should be reloaded from the stored state, for which `RenateModule` provides a
`from_state_dict` method.

**Signature**

`model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule`

**Example**

```python
import torch


class MyMNISTMLP(RenateModule):

    def __init__(self, num_hidden: int):
        # Model hyperparameters as well as the loss function need to registered via RenateModule's
        # constructor, see documentation. Otherwise, this is a standard torch model.
        super().__init__(
            hyperparameters={"num_hidden": num_hidden}
            loss_fn=torch.nn.CrossEntropyLoss()
        )
        self._fc1 = torch.nn.Linear(28*28, num_hidden)
        self._fc2 = torch.nn.Linear(num_hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._fc1(x)
        x = torch.nn.functional.relu(x)
        return self._fc2(x)


def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    if model_state_url is None:
        # If no model state is given, we create the model from scratch with initial model hyperparams.
        model = MyMNISTMLP(num_hidden=100)
    else:
        # If a model state is passed, we reload the model using RenateModule's load_state_dict.
        # In this case, model hyperparameters are restored from the saved state.
        state_dict = torch.load(str(model_state_url))
        model = MyMNISTMLP.from_state_dict(model)
```


## Data: `data_module_fn`

This function takes a path to a data folder and returns data in the form of a `RenateDataModule`.
`RenateDataModule` provides a structured interface to download, set up, and access train/val/test datasets; for detailed information, see [here](TODO).
The function also accepts a `seed`, which should be used for any randomized operations, such as data subsampling or splitting.

**Signature**

`data_module_fn(data_path: Union[Path, str], seed: int = defaults.SEED) -> RenateDataModule`

**Example**

```python
class MyMNISTDataModule(RenateDataModule):

    def __init__(self, data_path: Union[Path, str], val_size: float, seed: int = 42):
        super().__init__(data_path, val_size=val_size, seed=seed)

    def prepare_data(self):
        # This is only to download the data. We separate downloading from the remaining set-up to
        # streamline data loading when using multiple training jobs during HPO.
        _ = torchvision.datasets.MNIST(self._data_path, download=True)

    def setup(self, stage):
        # This sets up train/val/test datasets, assuming data has already been downloaded.
        if stage in ["train", "val"] or stage is None:
            train_data = torchvision.datasets.MNIST(
                self._data_path,
                train=True,
                transform=transforms.ToTensor(),
                target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)),
            )
            self._train_data, self._val_data = self._split_train_val_data(train_data)

        if stage == "test" or stage is None:
            self._test_data = torchvision.datasets.MNIST(
                self._data_path,
                train=False,
                transform=transforms.ToTensor(),
                target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)),
            )


def data_module_fn(data_path: Union[Path, str], seed) -> RenateDataModule:
    return MyMNISTDataModule(val_size=0.2, seed=seed)
```


## Transforms

Transforms for data preprocessing or augmentation are often applied "inside" of torch datasets.
That is, `x, y = dataset[i]` returns a fully-preprocessed and potentially augmented data point,
ready to be passed to a torch model.

In Renate, transforms should, to some extent, be handled _outside_ of the dataset object.
This is because many continual learning methods maintain a memory of previously-encountered data
points.
Having access to the _raw_, _untransformed_ data points allows us to store this data in a
memory-efficient way and ensures that data augmentation operations do not cumulate over time.
Explicit access to the preprocessing transforms is also useful when deploying a trained model.

It is on the user to decide which transforms to apply inside the dataset and which to pass to
Renate explicitly. As a general rule, `dataset[i]` should return `torch.Tensor`s of fixed size and data
type. Randomized data augmentation operations should be passed explicitly.

Transforms are specified in the config file via four functions
- `def train_transform() -> Callable`
- `def train_target_transform() -> Callable`
- `def test_transform() -> Callable`
- `def test_target_transform() -> Callable`

which return a transform in the form of a (single) callable.
These are applied to train and test data, as well as inputs (`x`) and targets (`y`), respectively.
The transform functions are optional and each of them can be ommitted if no respective transform
should be applied.

Some methods perform a separate set of transforms to data kept in a memory buffer, e.g., for
enhanced augmentation.
These can be set via two addition transform functions
- `def buffer_transform() -> Callable`
- `def buffer_target_transform() -> Callable`

These are optional as well but, if ommitted, Renate will use `train_transform` and
`train_target_transform`, respectively.

**Example**

```python
def train_transform():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop((28, 28), padding=4),
            torch.nn.Flatten()
        ]
    )


def test_transform()
    return torch.nn.Flatten()


def buffer_transform():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop((28, 28), padding=4),
            torchvision.transforms.RandomRotation(degrees=15),
            torch.nn.Flatten()
        ]
    )
```


