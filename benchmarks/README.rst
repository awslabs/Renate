Running Standard Benchmarks
******************************************************************************

This folder contains a couple of files to run experiments with Renate.
Add new experimentation configuration files to add new experiments.
Edit the ``requirements.txt`` to add additional requirements for an experiment (SageMaker only).

Instructions
============
1. Clone the repository
2. ``cd Renate`` and install Renate
3. Run a benchmark via

    .. code-block:: bash

        python benchmarks/run_benchmark.py --benchmark-file fine-tuning-clear10.json \
        --backend sagemaker  --budget-factor 1 --job-name clear10-finetuning-1 --num-repetitions 1

    This is an example command to run an experiment with ClEAR10 on SageMaker using a Fine-Tuning
    updater.

    Quick explanation of the arguments:

    - ``benchmark-file``: Any filename of files in ``experiment_configs``. This file specifies all
      properties of the experiment, i.e., dataset, scenario, updater, and hyperparameters settings.
      Modify or add more to run own experiments.
    - ``backend``: Run the experiment on SageMaker (``sagemaker``) or locally (``local``).
    - ``budget-factor``: Each update run will make ``budget_factor * max_epochs`` many passes over
      the new data during training. ``max_epochs`` is typically defines as part of the scenario
      .json file. Default: ``1``.
    - ``job-name``: Defines the name of the output folder and the name of the SageMaker training
      job.
    - ``num-repetitions``: The number of times the experiment will be repeated. Only the seed
      differs between repetitions.
    - ``max-time``: The wall clock time spent per update. Default: 5 days.