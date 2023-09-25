How to Run a Training Job
*************************

Renate offers possibility to run training jobs using both CLIs and functions that can be called
programmatically in python. The best choice may be different depending on the requirements (e.g.,
a CLI can be convenient to run remote jobs). In the following we illustrate the solution that we
find to be the simplest and more convenient. The complete documentation is available in
:py:func:`~renate.training.training.run_training_job`.


Setup
=====

The first step that needs to be completed before running a training job is to define which model needs
to be trained and on which data. This is explained in :doc:`how_to_renate_config`.

Once completed the first step, a simple way to run a training job is to use the
:py:func:`~renate.training.training.run_training_job`,
this can work for most training needs: it can launch trainings with and without HPO,
either locally or on Amazon SageMaker.


Run a local training job
========================

Running a local training is very easy: it requires providing a configuration and call
a function. The configuration is stored in a dictionary and may contain different arguments depending 
on the method you want to use to update your model.

In this example we will use a simple Experience Replay method and provide a configuration similar
to the following one. The arguments to be specified in the configuration are passed to the
:py:class:`~renate.updaters.model_updater.ModelUpdater`
instantiating the method you selected. See :doc:`supported_algorithms` for more information about the methods.

.. code-block:: python

    configuration = {
        "optimizer": "SGD",
        "momentum": 0.0,
        "weight_decay": 1e-2,
        "learning_rate": 0.05,
        "batch_size": 32,
        "max_epochs": 50,
        "memory_batch_size": 32,
        "memory_size": 500,
    }



..  note::
    If you have defined the ``optimizer_fn`` function in your Renate config, do not pass values for the keys
    ``optimizer``, ``momentum``, ``weight_decay``, or ``learning_rate``, unless you have specified them as
    :ref:`custom arguments <getting_started/how_to_renate_config:custom function arguments>`.

Once the configuration of the learning algorithm is specified, we need to set another couple of arguments
in the :py:func:`~renate.training.training.run_training_job` function to make sure we obtain the desired behavior:

* :code:`mode`: it can be either :code:`min` or :code:`max` and define if the aim is to minimize or maximize the metric
* :code:`metric`: it is the target metric. Metrics measured on the validation set are prefixed with :code:`val_`,
  while the ones measured on the training set are prefixed with :code:`train_`. Mode an metric will be used to checkpoint
  the best model if a validation set is provided, otherwise do not pass these arguments.
* :code:`updater`: the name of the algorithm to be used for updating the model. See :doc:`supported_algorithms` for more info.
* :code:`max_epochs`: the maximum number of training epochs.
* :code:`input_state_url`: this is the location at which the state of learner and the model to be updated are made available.
  If this argument is not passed, the model will be trained from scratch.
* :code:`output_state_url`: this is the location at which the output of the training job (e.g., model, state) will be stored.
* :code:`backend`: when set to :code:`local` will run the training job on the local machine

In both cases the urls can point to local folders or S3 locations.

Putting everything together will result in a script like the following.

.. literalinclude:: ../../examples/train_mlp_locally/start_training_with_er_without_hpo.py
    :lines: 3-

Once the training has been executed you will see some metrics printed on the screen (e.g., validation accuracy)
and you will find the output of the training process in the folder specified. For more information about the
output see :doc:`output`.

Run a training job on SageMaker
===============================

Running a job on SageMaker is very similar to run the training job locally, but it will require a few changes
to the arguments passed to :py:func:`~renate.training.training.run_training_job`:

* :code:`backend`: the backend will have to be set to :code:`sagemaker`.
* :code:`role`: an `execution role <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`_ will need to be passed.
* :code:`instance_type`: the type of machine to be used for the training. AWS provides a `list of training instances available <https://aws.amazon.com/sagemaker/pricing/>`_.
* :code:`job_name`: (optional) a prefix used to name the training job to make it recognizable in the SageMaker jobs list.

When using the SageMaker training you should use a S3 location as :code:`next_state_url` to make sure
you have access to the result after the job has finished.
We provide an example in :doc:`../examples/train_classifier_sagemaker`.


Run a training job with HPO
===========================

Running the training job with hyperparameter optimization (HPO) will require a few minor
additions to the components already discussed.

The first step to run an HPO job is to define the search space.
To this purpose, it will be sufficient to extend our configuration to include some ranges
instead of exact values. If a hyperparameter does not need to be tuned, an exact value can be provided.

.. code-block:: python

    config_space = {
        "optimizer": "SGD",
        "momentum": 0.0,
        "weight_decay": 1e-2,
        "learning_rate": uniform(0.001, 0.1),
        "batch_size": choice([32,64,128]),
        "max_epochs": 50,
        "memory_batch_size": uniform(1, 32),
        "memory_size": 500,
    }



For more suggestions and details about how to design a search space,
see the `Syne Tune documentation <https://github.com/awslabs/syne-tune/blob/main/docs/search_space.md>`_.
If you do not know which search space to use, you can adopt a default one by calling
:py:func:`~renate.utils.config_spaces.config_space` and passing the name of your algorithm to it.

.. code-block:: python

    from renate.utils.config_spaces import config_space

    config_space("ER")

After configuring the search space, it will be sufficient to add a few more arguments to
the :py:func:`~renate.training.training.run_training_job` function.
To start, please make sure that :code:`mode` and :code:`metric` (already introduced above) reflect your aim.
Also, please make sure that in :code:`data_module_fn` a reasonable
fraction of the data is assigned to the validation set, otherwise it will not be possible to measure validation performance reliably
(:code:`val_size` is controlling that).

It also possible to define more aspects of the HPO process:

* :code:`n_workers`: the number of workers evaluating configurations in parallel (useful for multi-cpu or multi-gpu machines).
* :code:`scheduler`: to decide which optimizer to use for the hyperparameters (e.g., "bo", "asha").
* specify one of the stopping criteria available, for example :code:`max_time` stops the tuning job after a certain amount of time.

After defining these arguments it will be sufficient to run the script and wait :)
The output will be available in the location specified in :code:`output_state_url`.

We provide an example of training on SageMaker with HPO at :doc:`../examples/train_classifier_sagemaker`.

Custom Function Arguments
=========================
Now that we know how to run basic training jobs, we can discuss how to use custom defined function arguments.
We are building upon the linear model example introduced in the
:ref:`previous chapter <getting_started/how_to_renate_config:custom function arguments>` where we added
``num_inputs`` and ``num_outputs`` to :code:`data_module_fn`.
The values for these inputs are passed via the configuration alongside the other arguments.

.. code-block:: python

    config_space = {
        # Define all remaining standard arguments as well
        "num_inputs": 28 * 28,
        "num_outputs": 10,
    }

While it does not make any sense for this example, we can also define ranges for our custom function arguments and
automatically optimize them during hyperparameter optimization.

.. note::
    If you have functions defined with the same argument name, the value defined in the configuration will be passed
    to both.
