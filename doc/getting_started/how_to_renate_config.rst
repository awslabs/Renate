How to Write a Config File
**************************

User input is passed to Renate via a config file.
It contains the definition of the model you want to train, code to load and preprocess the data,
and (optionally) a set of transforms to be applied to that data.
These components are provided by implementing functions with a fixed name.
When accessing your config file, Renate will inspect it for these functions.

Model Definition
================

This function takes a path to a model state and returns a model in the form of a
:py:class:`~renate.models.renate_module.RenateModule`.
Its signature is

.. code-block:: python

    def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:

A :py:class:`~renate.models.renate_module.RenateModule` is a :code:`torch.nn.Module` with some
additional functionality relevant to continual learning.
If no path is given (i.e., when we first train a model) your :code:`model_fn` should create
the model from scratch.
Otherwise it should be reloaded from the stored state, for which
:py:class:`~renate.models.renate_module.RenateModule` provides a
:py:meth:`~renate.models.renate_module.RenateModule.from_state_dict`
method, which automatically handles model hyperparameters.

.. literalinclude:: ../../examples/getting_started/renate_config.py
    :caption: Example
    :lines: 12-37

If you are using a torch model with **no or fixed hyperparameters**, you can use
:py:class:`~renate.models.renate_module.RenateWrapper`.
In this case, do not use the
:py:meth:`~renate.models.renate_module.RenateModule.from_state_dict`
method, but simply reinstantiate your model and call :code:`load_state_dict`.

.. code-block:: python
    :caption: Example

    def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
        my_torch_model = torch.nn.Linear(28*28, 10)  # Instantiate your torch model.
        model = RenateWrapper(my_torch_model)
        if model_state_url is not None:
            state_dict = torch.load(str(model_state_url))
            model.load_state_dict(state_dict)
        return model


Data Preparation
================

This function takes a path to a data folder and returns data in the form of a
:py:class:`~renate.data.data_module.RenateDataModule`.
Its signature is

.. code-block:: python

    def data_module_fn(data_path: Union[Path, str], seed: int = defaults.SEED) -> RenateDataModule:

:py:class:`~renate.data.data_module.RenateDataModule` provides a structured interface to
download, set up, and access train/val/test datasets.
The function also accepts a :code:`seed`, which should be used for any randomized operations,
such as data subsampling or splitting.

.. literalinclude:: ../../examples/getting_started/renate_config.py
    :caption: Example
    :lines: 40-70

Transforms
==========

Transforms for data preprocessing or augmentation are often applied as part of torch datasets.
That is, :code:`x, y = dataset[i]` returns a fully-preprocessed and potentially augmented
data point, ready to be passed to a torch model.

In Renate, transforms should, to some extent, be handled _outside_ of the dataset object.
This is because many continual learning methods maintain a memory of previously-encountered data
points.
Having access to the _raw_, _untransformed_ data points allows us to store this data in a
memory-efficient way and ensures that data augmentation operations do not cumulate over time.
Explicit access to the preprocessing transforms is also useful when deploying a trained model.

It is on the user to decide which transforms to apply inside the dataset and which to
pass to Renate explicitly. As a general rule, :code:`dataset[i]` should return a
:code:`torch.Tensor` of fixed size and data type. Randomized data augmentation
operations should be passed explicitly.

Transforms are specified in the config file via four functions

* :code:`def train_transform() -> Callable`
* :code:`def train_target_transform() -> Callable`
* :code:`def test_transform() -> Callable`
* :code:`def test_target_transform() -> Callable`

which return a transform in the form of a (single) callable.
These are applied to train and test data, as well as inputs (:code:`X`) and targets (:code:`y`), respectively.
The transform functions are optional and each of them can be omitted if no respective transform
should be applied.

Some methods perform a separate set of transforms to data kept in a memory buffer, e.g., for
enhanced augmentation.
These can be set via two addition transform functions

* :code:`def buffer_transform() -> Callable`
* :code:`def buffer_target_transform() -> Callable`

These are optional as well but, if omitted, Renate will use :code:`train_transform` and
:code:`train_target_transform`, respectively.

.. literalinclude:: ../../examples/getting_started/renate_config.py
    :caption: Example
    :lines: 73-

