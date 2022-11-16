# How to Write `renate_config.py`

User input required by `renate` goes into a config file.
It contains the definition of the model you want to train, code to load the data, and (optionally)
a set of transforms to be applied to that data.

## Model: `model_fn`

Signature: `model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule`

This function takes a path to a model state and returns a model in the form of a `RenateModule`.
For more information on `RenateModule`, see [here](TODO).
If no path is given (i.e., when we first train a model) the model should be created from scratch.

Example

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
        x = torch.flatten(x)
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

Signature:
`data_module_fn(data_path: Union[Path, str], seed: int = defaults.SEED) -> RenateDataModule`

This function takes a path to a data folder and returns data in the form of a `RenateDataModule`.
For more information on the `RenateDataModule`, see [here](TODO).
The function also accepts a `seed`, which should be used for any randomized operations used when
providing the data, such as data subsampling or splitting.

Example

```python
class MyMNISTDataModule(RenateDataModule):

    pass


def data_module_fn(data_path: Union[Path, str], seed) -> RenateDataModule:
    pass
```


## Transforms

Transforms for data preprocessing or augmentation are often done "inside" of dataset objects.
That is, `x, y = dataset[i]` returns a fully-preprocessed and potentially augmented data point, ready to
be passed to a torch model.

In `renate`, transforms should, to some extent, be handled _outside_ of the dataset object.
We do this because many continual learning methods maintain a memory of previously-encountered data points.
Having access to the _raw_, _untransformed_ data points allows us to store this data in a
memory-efficient way and ensures that data augmentation operations do not cumulate over time.

It is on the user to decide which transforms to apply inside the dataset and which to pass to
`renate` explicitly. As a general rule, `dataset[i]` should return tensors of fixed size and data
type. Randomized data augmentation operations should be passed explicitly.

Example
```python
TODO
```


