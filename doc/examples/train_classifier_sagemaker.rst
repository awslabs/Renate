Training and Tuning on SageMaker
********************************

This example is designed to demonstrate how to use Renate on Amazon SageMaker for
both training the model and tuning the hyperparameters required for that.
To this purpose we will train a ResNet model on CIFAR10 and tune some hyperparameters
using ASHA, an advanced optimizer able to quickly terminate suboptimal hyperparameter
configurations.

Configuration
=============
The model and dataset definitions are in the file :code:`renate_config.py`.
The :code:`model_fn` function instantiates a ResNet neural network (a common choice
for many image classifiers) and the :code:`data_module_fn` function loads the
CIFAR10 dataset.

.. note::
    The CIFAR10 dataset is split in two chunks (first five classes and last five classes respectively)
    using the ClassIncrementalScenario.
    The splitting operation is not necessary in real-world applications, but
    it can be useful to run experiments when testing the library and
    it was useful for us to create a simple example using a single dataset :)

In the :code:`renate_config.py` file we also create some simple transformations to
normalize and augment the dataset. More transformations can be added if needed,
details on how to write a configuration file are available in :doc:`../getting_started/how_to_renate_config`.

.. literalinclude:: ../../examples/simple_classifier_cifar10/renate_config.py
    :lines: 3-

The configuration file uses a scenario in the definition of the data module function.
The scenario is just splitting the dataset in several chunks and allows us to train the model on
different parts of the dataset without adding complex code to the example. For most practical purposes
the definition of the scenario can just be removed from the function.

Training
========

The example also contains :code:`start_with_hpo.py`,
which launches a training job with integrated hyperparameters optimization.
To this purpose, in the file we define a dictionary containing the
configuration of the learning algorithm. In some cases instead of a single
value we define a range (e.g., :code:`uniform(0.0, 1.0)`) in which the
optimizer will try to identify the best value of the hyperparameter.
We also specify which algorithm to use for the optimization using the argument :code:`scheduler="asha"`.
In this case we will use the ASHA algorithm with 4 workers evaluating up to
100 hyperparameters combinations.
The model and the output of the HPO process will be saved in the S3 bucket provided
in :code:`next_state_url`, to simplify the example in this case we provide two variables
that can be used to set AWS Account ID and AWS region used, but any accessible S3 bucket can be
used for storing the output.
The description of the other arguments and a high level overview of how to run a
training jobs are available in :doc:`../getting_started/how_to_run_training`.

.. literalinclude:: ../../examples/simple_classifier_cifar10/start_with_hpo.py
    :lines: 3-

Once the training job terminates, the output will be available in the S3 bucket indicated
in :code:`next_state_url`. For more information about how to interpret the output, see
:doc:`../getting_started/output`.

To simulate an application where data are made available incrementally over time,
after the first training job has been executed, it is possible to re-train the model
on the second chunk of the dataset that we left intentionally untouched during
the first training process.

To do this, it is sufficient to modifying the arguments passed to the :py:func:`~renate.training.training.run_training_job` function.
In particular:

1. select the second part of the datasets by setting :code:`chunk-id = 1`.

2. load the model trained in the first training job by adding the :code:`state_url`
argument pointing to the same S3 location. In this case it will be useful to change the
url for the :code:`next_state_url` to avoid overwriting the old artefacts.

Note that in our example we specified :code:`requirements_file="requirements.txt"`
even if it is not necessary since the only dependency in the file is Renate itself.
The only purpose of this is to show how additional dependencies can be added when needed.

