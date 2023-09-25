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

    def model_fn(model_state_url: Optional[str] = None) -> RenateModule:

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
    :lines: 18-42

If you are using a torch model with **no or fixed hyperparameters**, you can use
:py:class:`~renate.models.renate_module.RenateWrapper`.
In this case, do not use the
:py:meth:`~renate.models.renate_module.RenateModule.from_state_dict`
method, but simply reinstantiate your model and call :code:`load_state_dict`.

.. code-block:: python
    :caption: Example

    def model_fn(model_state_url: Optional[str] = None) -> RenateModule:
        my_torch_model = torch.nn.Linear(28 * 28, 10)  # Instantiate your torch model.
        model = RenateWrapper(my_torch_model)
        if model_state_url is not None:
            state_dict = torch.load(str(model_state_url))
            model.load_state_dict(state_dict)
        return model


Loss Definition
================

This function returns a :code:`torch.nn.Module` object that computes the loss with the 
signature 

.. code-block:: python 
    
    def loss_fn() -> torch.nn.Module:

An example of this for the task of MNIST classification above as

.. literalinclude:: ../../examples/getting_started/renate_config.py
    :caption: Loss function example
    :lines: 99-100

Please note, loss functions should not be reduced.

Data Preparation
================

This function takes a path to a data folder and returns data in the form of a
:py:class:`~renate.data.data_module.RenateDataModule`.
Its signature is

.. code-block:: python

    def data_module_fn(data_path: str, seed: int = defaults.SEED) -> RenateDataModule:

:py:class:`~renate.data.data_module.RenateDataModule` provides a structured interface to
download, set up, and access train/val/test datasets.
The function also accepts a :code:`seed`, which should be used for any randomized operations,
such as data subsampling or splitting.

.. literalinclude:: ../../examples/getting_started/renate_config.py
    :caption: Example
    :lines: 45-72

Optimizer
=========

Optimizers such as ``SGD`` or ``Adam`` can be selected by passing the corresponding arguments.
If you want to use other optimizers, you can do so by returning a partial optimizer object as
outlined in the example below.

.. literalinclude:: ../../examples/getting_started/renate_config.py
    :caption: Example
    :lines: 103-104

Learning Rate Schedulers
========================

A learning rate scheduler can be provided by creating a function as demonstrated below.
This function will need to return a partial object of a learning rate scheduler as well as a string
that indicates whether the scheduler is updated after each ``epoch`` or after each ``step``.

.. literalinclude:: ../../examples/getting_started/renate_config.py
    :caption: Example
    :lines: 107-108

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
    :lines: 75-82

Custom Metrics
==============

It is possible to specify a set of custom metrics to be measured during the training process.
The metrics can be either imported from :code:`torchmetrics`, which offers a vast collection,
or created ad-hoc by implementing the same interface
(see this `tutorial <https://torchmetrics.readthedocs.io/en/stable/pages/implement.html>`_).

.. literalinclude:: ../../examples/getting_started/renate_config.py
    :caption: Example
    :lines: 95-96

To enable the usage of additional metrics in Renate it is sufficient to implement the
:code:`metrics_fn` function, returning a dictionary where the key is a string containing the
metric's name and the value is an instantiation of the metric class.
In the example above we add a metric called :code:`my_accuracy` by instantiating the accuracy
metric from :code:`torchmetrics`.

Custom Function Arguments
=========================
In many cases, the standard arguments passed to all functions described above are not sufficient.
More arguments can be added by simply adding them to the interface (with some limitations).
We will demonstrate this at the example of :code:`data_module_fn` but the same rules apply to all other functions
introduced in this chapter.

Let us assume we already have a config file in which we implemented a simple linear model:

.. code-block:: python

    def model_fn(model_state_url: Optional[str] = None) -> RenateModule:
        my_torch_model = torch.nn.Linear(28 * 28, 10)
        model = RenateWrapper(my_torch_model)
        if model_state_url is not None:
            state_dict = torch.load(model_state_url)
            model.load_state_dict(state_dict)
        return model

However, we have different datasets and each of them has different input and output dimensions.
The natural change would be to change it to something like

.. code-block:: python

    def model_fn(num_inputs: int, num_outputs: int, model_state_url: Optional[str] = None) -> RenateModule:
        my_torch_model = torch.nn.Linear(num_inputs, num_outputs)
        model = RenateWrapper(my_torch_model)
        if model_state_url is not None:
            state_dict = torch.load(model_state_url)
            model.load_state_dict(state_dict)
        return model

And in fact, this is exactly how it works. However, there are few limitations:

* Typing is required.
* Only types allowed: ``bool``, ``float``, ``int``, ``str``, ``list``, and ``tuple``.
  (typing with ``List``, ``Tuple`` or ``Optional`` is okay)

How to set the actual values, will be discussed in
:ref:`the next chapter <getting_started/how_to_run_training:custom function arguments>`.

.. note::
    You can use an argument with the same name in the different functions as long as they have the same typing.
    The same value will provided to them.
