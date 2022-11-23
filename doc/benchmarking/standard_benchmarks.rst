.. _benchmarking-standard-benchmarks:

Standard Benchmarks
*******************

Renate features a variety of :ref:`models <benchmarking-standard-benchmarks-models>`,
:ref:`datasets <benchmarking-standard-benchmarks-datasets>` and :ref:`scenarios <benchmarking-standard-benchmarks-scenarios>`.
This allows for evaluating the different Renate updaters on many standard benchmarks.
In the following, we describe how to combine the different components to your very own benchmark.
Independent of the benchmark, they all are started via :py:func:`renate.benchmark.experimentation.execute_experiment_job`.
The function call below demonstrates the simplest setup where the benchmark is run locally.
The benchmark will be configured by :code:`config_space`.

.. code-block:: python

    from renate.benchmark import execute_experiment_job, experimentation_config

    execute_experiment_job(
        backend="local",
        config_file=experimentation_config(),
        config_space=config_space,
        experiment_outputs_url="results/",
        mode="max",
        metric="val_accuracy",
        num_updates=5,
    )

.. _benchmarking-standard-benchmarks-models:

Models
======

You can select the model by assigning the corresponding name to :code:`config_space["model_fn_model_name"]`.
For example, to use a ResNet-18 model, you use

.. code-block:: python

    config_space["model_fn_model_name"] = "ResNet18"

.. _benchmarking-standard-benchmarks-datasets:

Datasets
========

.. _benchmarking-standard-benchmarks-scenarios:

Scenarios
=========

