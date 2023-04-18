Local Training with MNIST
*************************

This example is designed to demonstrate Renate's capabilities
using a small dataset and a small model in order to run the training locally
in a few minutes.

Configuration
=============

To this purpose, the :code:`model_fn` function defined in the :code:`renate_config.py`
contains the definition of a Multi-Layer Perceptron. In the same file,
we also created :code:`data_module_fn`, a function loading the MNIST dataset from a public
repository.

.. note::
    The MNIST dataset is split in two chunks (first five classes and last five classes respectively)
    using the ClassIncrementalScenario.
    The splitting operation is not necessary in real-world applications, but
    it can be useful to run experiments when testing the library and
    it was useful for us to create a simple example using a single dataset.

In the :code:`renate_config.py` file we also create a simple transformation flattening
the input images (matrices 28x28) into vectors. Transformations do not provide
only the ability to reshape the inputs, but they can be used for normalization,
augmentation, and several other purposes. More details on how to write a
configuration file are available in :doc:`../getting_started/how_to_renate_config`.

.. literalinclude:: ../../examples/train_mlp_locally/renate_config.py
    :lines: 3-
    
The configuration file uses a scenario in the definition of the data module function.
The scenario is just splitting the dataset in several chunks and allows us to train the model on
different parts of the dataset without adding complex code to the example. For most practical purposes
the definition of the scenario can be removed from the function.

Training
========

The example also contains :code:`start_training_without_hpo.py`,
which is the one launching the training jobs. In the file we defined a
configuration using the :code:`config_space` dictionary and pass it to
a function launching the training job.
The configuration controls a number of aspects of the learning process,
for example the learning rate and the optimizer. The list depends on the
learning algorithm used for the training.
There are also parameters that we pass directly like
the folder in which the learner state will be saved (via :code:`next_state_url`).
In order to update an existing model, it will also be necessary to provide the path
to the previously saved state using :code:`state_url`, as done in our example.
More details about running training jobs are available in :doc:`../getting_started/how_to_run_training`.

.. literalinclude:: ../../examples/train_mlp_locally/start_training_without_hpo.py
    :lines: 3-

Results
=======

The results obtained by running this simple example are usually quite good
with an almost-perfect accuracy on the first chunk of data and an accuracy
still above 90% after processing the second one.
After the execution is completed, it will be possible to inspect the two
different folders containing the learner states.

Another example using Repeated Distillation and HPO is available in the file called
:code:`start_training_with_hpo.py`.



