Introduction
************

Renate provides a feature to simulate continual learning offline.
By providing a dataset, you will be able to split it into smaller parts and simulate the behavior of updating your model
on a regular basis.
Among other things, Renate will support you in evaluating different optimizers, hyperparameters and the expected
performance on your historic data or public benchmark data.
At the core of this feature is the function :py:func:`~renate.benchmark.experimentation.execute_experiment_job`.
For the reader familiar with the function :py:func:`~renate.tuning.tuning.execute_tuning_job`, the use will be very
intuitive.

In the following chapters, we will discuss how this interface can be used to experiment on
:ref:`standard benchmarks <benchmarking-standard-benchmarks>` as well as
:ref:`custom benchmarks <benchmarking-custom-benchmarks>`.
