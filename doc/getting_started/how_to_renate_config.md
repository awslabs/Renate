# How to Write a Config File

User input is passed to `renate` via a config file.
It contains the definition of the model you want to train, code to load and preprocess the data,
and (optionally) a set of transforms to be applied to that data.
These components are provided by implementing functions with a fixed name.
When accessing your config file, `renate` will inspect it for these functions.

## Model: `model_fn`

This function takes a path to a model state and returns a model in the form of a `RenateModule`.
For more information on `RenateModule`, see [here](TODO).
If no path is given (i.e., when we first train a model) the model should be created from scratch.

**Signature**

`model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule`

**Example**

```python
import torch


class MyMNISTMLP(RenateModule):

    def __init__(self, num_hidden: int):
        super().__init__(
            constructor_args={"num_hidden": num_hidden}
            loss_fn=torch.nn.CrossEntropyLoss()
        )
        self._fc1 = torch.nn.Linear(num_inputs, num_hidden)
        self._fc2 = torch.nn.Linear(num_hidden, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._fc1(x)
        x = torch.nn.functional.relu(x)
        return self._fc2(x)


def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    if model_state_url is None:
        model = MyMLP(num_hidden=100)
    else:
        state_dict = torch.load(str(model_state_url))
        model = MyMLP.from_state_dict(model)
```


## Data: `data_module_fn`

This function takes a path to a data folder and returns data in the form of a `RenateDataModule`.
For more information on the `RenateDataModule`, see [here](TODO).
The function also accepts a `seed`, which should be used for any randomized operations used when
providing the data, such as data subsampling or splitting.

**Signature**

`data_module_fn(data_path: Union[Path, str], seed: int = defaults.SEED) -> RenateDataModule`

**Example**

```python
class MyMNISTDataModule(RenateDataModule):

    def __init__(self, data_path: Union[Path, str], val_size: float, seed: int = 42):
        super().__init__(data_path, val_size=val_size, seed=seed)

    def prepare_data(self):
        _ = torchvision.datasets.MNIST(self._data_path, download=True)

    def setup(self, stage):
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
That is, `x, y = dataset[i]` returns a fully-preprocessed and potentially augmented data point, ready to
be passed to a torch model.

In `renate`, transforms should, to some extent, be handled _outside_ of the dataset object.
This is because many continual learning methods maintain a memory of previously-encountered data points.
Having access to the _raw_, _untransformed_ data points allows us to store this data in a
memory-efficient way and ensures that data augmentation operations do not cumulate over time.

It is on the user to decide which transforms to apply inside the dataset and which to pass to
`renate` explicitly. As a general rule, `dataset[i]` should return `torch.Tensor`s of fixed size and data
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

These are optional as well but, if ommitted, we will use `train_transform` and
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


