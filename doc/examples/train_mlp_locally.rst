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
    it was useful for us to create a simple example using a single dataset :)

In the :code:`renate_config.py` file we also create a simple transformation flattening
the input images (matrices 28x28) into vectors. Transformations do not provide
only the ability to reshape the inputs, but they can be used for normalization,
augmentation, and several other purposes. More details on how to write a
configuration file are available in :doc:`../getting_started/how_to_renate_config`.

.. literalinclude:: ../../examples/train_mlp_locally/renate_config.py

Training
========

The example also contains :code:`start_training_without_hpo.py`,
which is the one launching the training jobs. In the file we defined a
fixed configuration (:code:`config_space` dictionary) and pass it to a function launching the training job.
Since the search space contains a single configuration there will not be
any optimization of the hyperparameters. To define in which folder the learner state will be saved,
we provide a local path with :code:`next_state_url`.
In order to update an existing model, it will be necessary to provide the path
to the previously saved state using :code:`state_url`, as done in our example.
More details about running training jobs are available in :doc:`../getting_started/hot_to_run_training`.

.. literalinclude:: ../../examples/train_mlp_locally/start_training_without_hpo.py

Results
=======

The results obtained by running this simple example are usually quite good
with an almost-perfect accuracy on the first chunk of data and an accuracy
still above 90% after processing the second one.
After the execution is completed, it will be possible to inspect the two
different folders containing the learner states.



