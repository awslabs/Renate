How to Run a Training Job
*************************

Renate offers possibility to run training jobs using both CLIs and functions that can be called
programmatically in python. The best choice may be different depending on the requirements (e.g.,
a CLI can be convenient to run remote jobs). In the following we illustrate the solution that we
find to be the simplest and more convenient. The complete documentation is available in :py:func:`renate.tuning.tuning.execute_tuning_job`.


Setup
-----

The first step that needs to be completed before running a training job is to define which model needs
to be trained and on which data. This is explained in :doc:`how_to_renate_config`.

Once completed the first step, a simple way to run a training job is to use the :py:func:`renate.tuning.tuning.execute_tuning_job`,
this can work for most training needs: it can launch trainings with and without HPO,
either locally or on Amazon SageMaker.


Run a local training job
---------------------------

Running a local training is very easy: it requires providing a configuration and call
a function. The configuration is stored in a dictionary and may contain different arguments depending 
on the method you want to use to update your model.

In this example we will use a simple Experience Replay method and provide a configuration similar
to the following one. The arguments to be specified in the configuration are passed to the :py:class:`renate.updaters.model_updater.ModelUpdater`
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



Once the configuration of the learning algorithm is specified, we need to set another couple of arguments
in the :code:`execute_tuning_job` function to make sure we obtain the desired behavior:

* :code:`mode`: it can be either :code:`min` or :code:`max` and define if the aim is to minimize or maximize the metric
* :code:`metric`: it is the target metric. Metrics measured on the validation set are prefixed with :code:`val_`,
  while the ones measured on the training set are prefixed with :code:`train_`. Mode an metric will be used to checkpoint
  the best model if a validation set is provided, otherwise do not pass these arguments.
* :code:`updater`: the name of the algorithm to be used for updating the model. See :doc:`supported_algorithms` for more info.
* :code:`max_epochs`: the maximum number of training epochs.
* :code:`state_url`: this is the location at which the state of learner and the model to be updated are made available.
  If this argument is not passed, the model will be trained from scratch.
* :code:`next_state_url`: this is the location at which the output of the training job (e.g., model, state) will be stored.
* :code:`backend`: when set to :code:`local` will run the training job on the local machine

In both cases the urls can point to local folders or S3 locations.

Putting everything together will result in a script like the following.

.. literalinclude:: ../../examples/train_mlp_locally/start_training_with_er_without_hpo.py

Once the training has been executed you will see some metrics printed on the screen (e.g., validation accuracy)
and you will find the output of the training process in the folder specified. For more information about the
output see :doc:`output`.

Run a training job on SageMaker
-------------------------------

Running a job on SageMaker is very similar to run the training job locally, but it will require a few changes
to the arguments passed to :code:`execute_tuning_job`:

* :code:`backend`: the backend will have to be set to :code:`sagemaker`.
* :code:`role`: an execution role will need to be passed. If uncertain about what to pass just import :code:`get_execution_role()` from Syne Tune as in the example below.
* :code:`instance_type`: the type of machine to be used for the training. The list training instances is available `here <https://aws.amazon.com/sagemaker/pricing/>`_.
* :code:`job_name`: (optional) a prefix used to name the training job to make it recognizable in the SageMaker jobs list.

When using the SageMaker training it is often convenient to set an S3 location as :code:`next_state_url`.

.. note::
    A working example is available in: :doc:`../examples/train_classifier_sagemaker`.


Run a training job with HPO
---------------------------

Running the training job with HPO will require in a few minor additions to the components already discussed.

The first step to run an HPO job is to define the search space within which the optimizer will try to locate
the optimal configuration. To this purpose, it will be sufficient to extend our configuration to include some ranges
instead of exact values. If a hyperparamter does not need to be tuned, an exact value can be provided.
For an example see the code below.

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
If you don't know which search space to use, you can adopt a default one by calling :py:func:`renate.tuning.config_spaces.config_space` and passing the
name of your algorithm to it.

After configuring the search space, it will be sufficient to add a few more arguments to the :code:`execute_tuning_job` function.
To start, please make sure that :code:`mode` and :code:`metric` (already introduced above) reflect your aim.
Also, please make sure that in :code:`data_module_fn` a reasonable
fraction of the data is assigned to the validation set, otherwise it will not be possible to measure validation performance reliably
(:code:`val_size` is controlling that).

It also possible to define more aspects of the HPO process:

* :code:`n_workers`: the number of workers evaluating configurations in parallel (useful for multi-cpu or multi-gpu machines).
* :code:`scheduler`: to decide which optimizer to use for the hyperparameters (e.g., "bo", "asha").
* specify one of the stopping criteria available, for example :code:`max_time` stops the tuning job after a certain amount of time.

After defining these arguments it will be sufficient to run the script and wait :)
The output will be available in the location specified in :code:`next_state_url`.

.. note::
    A working example of training on SageMaker with HPO is available in :doc:`../examples/train_classifier_sagemaker`.


Running experiments
-------------------

If you want to run a large number of training jobs, for example for experimentation, it will
be useful to take a look at the :doc:`../benchmarking/index`.