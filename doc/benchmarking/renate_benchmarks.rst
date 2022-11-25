Renate Benchmarks
*****************

Renate features a variety of :ref:`models <benchmarking-renate-benchmarks-models>`,
:ref:`datasets <benchmarking-renate-benchmarks-datasets>` and :ref:`scenarios <benchmarking-renate-benchmarks-scenarios>`.
This allows for evaluating the different Renate updaters on many standard benchmarks.
In the following, we describe how to combine the different components to your very own benchmark.
Independent of the benchmark, they all are started via :py:func:`~renate.benchmark.experimentation.execute_experiment_job`.
The function call below demonstrates the simplest setup where the benchmark is run locally.
The benchmark will be configured by :code:`config_space`.

.. code-block:: python

    from renate.benchmark.experimentation import execute_experiment_job, experiment_config_file

    execute_experiment_job(
        backend="local",
        config_file=experiment_config_file(),
        config_space=config_space,
        experiment_outputs_url="results/",
        mode="max",
        metric="val_accuracy",
        num_updates=5,
    )

.. _benchmarking-renate-benchmarks-models:

Models
======

You can select the model by assigning the corresponding name to :code:`config_space["model_fn_model_name"]`.
For example, to use a ResNet-18 model, you use

.. code-block:: python

    config_space["model_fn_model_name"] = "ResNet18"

When using the model :py:class:`~renate.benchmark.models.mlp.MultiLayerPerceptron`,
additional information needs to be provided.
You need to define input and output size as well as depth and width of the network.

.. code-block:: python

    config_space["model_fn_num_inputs"] = 32*32*3     # Number of inputs
    config_space["model_fn_num_outputs"] = 10         # Number of outputs
    config_space["model_fn_num_hidden_layers"] = 2    # Number of hidden layers
    config_space["model_fn_hidden_size"] = "[16,16]"  # Hidden layer 1 and 2 have size 16

.. warning::
    All additional arguments passed to :py:class:`~renate.benchmark.models.mlp.MultiLayerPerceptron` will be
    converted to string and eventually converted back.
    These strings must not contain any whitespaces. Since a Python lists or tuples automatically contain whitespaces,
    they have to manually passed as strings without whitespaces.

The full list of models and model names including a short description is provided in the following table.

.. list-table:: Renate Model Overview
    :header-rows: 1

    * - Model Name
      - Description
    * - MultiLayerPerceptron
      - Neural network consisting of a sequence of dense layers.
    * - ResNet18
      - 18 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture
    * - ResNet34
      - 34 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture
    * - ResNet50
      - 50 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture
    * - ResNet18CIFAR
      - 18 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture for small image sizes (approx 32x32)
    * - ResNet34CIFAR
      - 34 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture for small image sizes (approx 32x32)
    * - ResNet50CIFAR
      - 50 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture for small image sizes (approx 32x32)
    * - VisionTransformerCIFAR
      - Base `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 32x32 with patch size 4.
    * - VisionTransformerB16
      - Base `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 224x224 with patch size 16.
    * - VisionTransformerB32
      - Base `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 224x224 with patch size 32.
    * - VisionTransformerL16
      - Large `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 224x224 with patch size 16.
    * - VisionTransformerL32
      - Large `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 224x224 with patch size 32.
    * - VisionTransformerH14
      - Huge `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 224x224 with patch size 14.


.. _benchmarking-renate-benchmarks-datasets:

Datasets
========

Similarly, you select the dataset by assigning the corresponding name to :code:`config_space["data_module_fn_dataset_name"]`.
To use the matching preprocessing, the same value needs to be applied to :code:`config_space["transform_dataset_name"]`.
For example, to use the CIFAR-10 dataset with 10% of the data used for validation, you use

.. code-block:: python

    dataset_name = "CIFAR10"
    config_space["data_module_fn_dataset_name"] = dataset_name
    config_space["transform_dataset_name"] = dataset_name
    config_space["data_module_fn_val_size"] = 0.1

The following table contains the list of supported datasets.

.. list-table:: Renate Dataset Overview
    :header-rows: 1

    * - Dataset Name
      - Task
      - Data Summary
      - Reference
    * - CIFAR10
      - Image Classification
      - 50k train, 10k test, 10 classes, image shape 32x32x3
      - Alex Krizhevsky: Learning Multiple Layers of Features from Tiny Images. 2009.
    * - CIFAR100
      - Image Classification
      - 50k train, 10k test, 100 classes, image shape 32x32x3
      - Alex Krizhevsky: Learning Multiple Layers of Features from Tiny Images. 2009.
    * - FashionMNIST
      - Image Classification
      - 60k train, 10k test, 10 classes, image shape 28x28x1
      - Han Xiao et al.: Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. 2017.
    * - MNIST
      - Image Classification
      - 60k train, 10k test, 10 classes, image shape 28x28x1
      - Li Deng: The MNIST Database of Handwritten Digit Images for Machine Learning Research. IEEE Signal Processing Magazine. 2012.

.. _benchmarking-renate-benchmarks-scenarios:

Scenarios
=========

A scenario defines how the dataset is split into several data partitions.
While running the benchmark, the model is trained sequentially on each data partition.
The scenario can be selected by setting :code:`config_space["data_module_fn_scenario_name"]` accordingly.
Each scenario might have specific settings which require additional changes of :code:`config_space`.
We will describe those settings in the table below.

The following is an example that uses the class-incremental scenario which splits the entire dataset into 2 parts.
The first part contains all instances with classes 1 and 2, the second with classes 3 and 4.

.. code-block:: python

    config_space["data_module_fn_scenario_name"] = "ClassIncrementalScenario"
    config_space["data_module_fn_class_groupings"] = "[[1,2],[3,4]]"

.. warning::
    As already explained in more detail above, lists and tuples must be passed as strings without whitespaces.

.. list-table:: Renate Scenario Overview
    :widths: 15 35 35
    :header-rows: 1

    * - Scenario Name
      - Description
      - Settings
    * - :py:class:`~renate.benchmark.scenarios.BenchmarkScenario`
      - Used in combination only with CLEAR-10 or CLEAR-100.
      - * :code:`data_module_fn_num_tasks`: Number of data partitions.
    * - :py:class:`~renate.benchmark.scenarios.ClassIncrementalScenario`
      - Creates data partitions by splitting the data according to class labels.
      - * :code:`data_module_fn_class_groupings`: List of list containing the class labels.
    * - :py:class:`~renate.benchmark.scenarios.PermutationScenario`
      - Creates data partitions by randomly permuting the input features.
      - * :code:`data_module_fn_num_tasks`: Number of data partitions.
        * :code:`data_module_fn_input_dim`: Data dimensionality (tuple or int as string).
    * - :py:class:`~renate.benchmark.scenarios.ImageRotationScenario`
      - Creates data partitions by rotating the images by different angles.
      - * :code:`data_module_fn_degrees`: Tuple of degrees as string.

Example: Class-incremental Learning on CIFAR-10
===============================================

The following example reproduces the results shown in
`Dark Experience for General Continual Learning: a Strong, Simple Baseline <https://arxiv.org/abs/2004.07211>`_
Table 2, Buffer 500, S-CIFAR-10, Class-IL, DER++.
These settings use a class-incremental scenario in which the CIFAR-10 dataset is partitioned into 5 parts, each
containing two unique classes.
Dark Experience Replay++ is used as the updating method with a memory buffer size of 500
and the experiment is repeated 10 times.

.. literalinclude:: ../../examples/benchmarking/class_incremental_learning_cifar10_der.py
