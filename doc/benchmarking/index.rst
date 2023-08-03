************
Benchmarking
************

Introduction
************

Renate provides a feature to simulate continual learning offline.
By providing a dataset, you will be able to split it into smaller parts and simulate the behavior of updating your model
on a regular basis.
Among other things, Renate will support you in evaluating different optimizers, hyperparameters and the expected
performance on your historic data or public benchmark data.
At the core of this feature is the function :py:func:`~renate.benchmark.experimentation.execute_experiment_job`.
For the reader familiar with the function :py:func:`~renate.training.training.run_training_job`, the use will be very
intuitive.

Renate's benchmarking functionality may require additional dependencies.
Please install them via

.. code-block:: bash

    pip install Renate[benchmark]

In the following chapters, we will discuss how this interface can be used to experiment on
:doc:`Renate benchmarks <renate_benchmarks>` as well as
:doc:`custom benchmarks <custom_benchmarks>`.

.. toctree::
    :maxdepth: 2

    renate_benchmarks
    custom_benchmarks
