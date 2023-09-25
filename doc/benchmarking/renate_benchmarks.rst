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

You can select the model by assigning the corresponding name to :code:`config_space["model_name"]`.
For example, to use a ResNet-18 model, you use

.. code-block:: python

    config_space["model_name"] = "ResNet18"

Each model may have independent arguments.
The ResNet-18 model requires to define the number of outputs.

.. code-block:: python

    config_space["num_outputs"] = 10


The full list of models and model names including a short description is provided in the following table.

.. list-table:: Renate Model Overview
    :header-rows: 1

    * - Model Name
      - Description
      - Additional Inputs
    * - `~renate.benchmark.models.mlp.MultiLayerPerceptron`
      - Neural network consisting of a sequence of dense layers.
      - * ``num_inputs``: Input dimensionality of data.
        * ``num_outputs``: Output dimensionality, for classification the number of classes.
        * ``num_hidden_layers``: Number of hidden layers.
        * ``hidden_size``: Size of hidden layers, can be ``int`` or ``Tuple[int]``.
    * - `~renate.benchmark.models.resnet.ResNet18`
      - 18 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.resnet.ResNet34`
      - 34 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.resnet.ResNet50`
      - 50 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.resnet.ResNet18CIFAR`
      - 18 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture for small image sizes (approx 32x32)
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.resnet.ResNet34CIFAR`
      - 34 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture for small image sizes (approx 32x32)
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.resnet.ResNet50CIFAR`
      - 50 layer `ResNet <https://arxiv.org/abs/1512.03385>`_ CNN architecture for small image sizes (approx 32x32)
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.vision_transformer.VisionTransformerCIFAR`
      - Base `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 32x32 with patch size 4.
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.vision_transformer.VisionTransformerB16`
      - Base `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 224x224 with patch size 16.
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.vision_transformer.VisionTransformerB32`
      - Base `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 224x224 with patch size 32.
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.vision_transformer.VisionTransformerL16`
      - Large `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 224x224 with patch size 16.
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.vision_transformer.VisionTransformerL32`
      - Large `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 224x224 with patch size 32.
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.vision_transformer.VisionTransformerH14`
      - Huge `Vision Transformer <https://arxiv.org/abs/2010.11929>`_ architecture for images of size 224x224 with patch size 14.
      - * ``num_outputs``: Output dimensionality, for classification the number of classes.
    * - `~renate.benchmark.models.transformer.HuggingFaceSequenceClassificationTransformer`
      - Wrapper around Hugging Face transformers.
      - * ``pretrained_model_name_or_path``: Hugging Face `transformer ID <https://huggingface.co/models>`__.
        * ``num_outputs``: The number of classes.


.. _benchmarking-renate-benchmarks-datasets:

Datasets
========

Similarly, you select the dataset by assigning the corresponding name to :code:`config_space["dataset_name"]`.
For example, to use the CIFAR-10 dataset with 10% of the data used for validation, you use

.. code-block:: python

    config_space["dataset_name"] = "CIFAR10"
    config_space["val_size"] = 0.1

The following table contains the list of supported datasets.

.. list-table:: Renate Dataset Overview
    :header-rows: 1

    * - Dataset Name
      - Task
      - Data Summary
      - Reference
    * - arxiv
      - Text Classification: category recognition of arXiv papers.
      - ~1.9M train, ~206k test, 172 classes, years 2007-2023
      - Huaxiu Yao et al.: Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time. Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2022.
    * - CIFAR10
      - Image Classification
      - 50k train, 10k test, 10 classes, image shape 32x32x3
      - Alex Krizhevsky: Learning Multiple Layers of Features from Tiny Images. 2009.
    * - CIFAR100
      - Image Classification
      - 50k train, 10k test, 100 classes, image shape 32x32x3
      - Alex Krizhevsky: Learning Multiple Layers of Features from Tiny Images. 2009.
    * - CLEAR10
      - Image Classification
      - 10 different datasets, one for each year. Each with 3,300 train, 550 test, 11 classes
      - Zhiqiu Lin et al.: The CLEAR Benchmark: Continual LEArning on Real-World Imagery. NeurIPS Datasets and Benchmarks 2021.
    * - CLEAR100
      - Image Classification
      - 11 different datasets, one for each year. Each with roughly 10k train, 5k test, 100 classes
      - Zhiqiu Lin et al.: The CLEAR Benchmark: Continual LEArning on Real-World Imagery. NeurIPS Datasets and Benchmarks 2021.
    * - DomainNet
      - Image Classification
      - 6 datasets from different domains. 345 classes, number of train and test image varies
      - Xingchao Peng et al.: Moment Matching for Multi-Source Domain Adaptation. ICCV 2019.
    * - FashionMNIST
      - Image Classification
      - 60k train, 10k test, 10 classes, image shape 28x28x1
      - Han Xiao et al.: Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. 2017.
    * - fmow
      - Image Classification: land use recognition from satellite images.
      - 62 classes, image shape 32x32x3
      - Huaxiu Yao et al.: Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time. Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2022.
    * - huffpost
      - Text Classification: category recognition of news paper articles.
      - ~58k train, ~6k test, 11 classes, years 2012-2019
      - Huaxiu Yao et al.: Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time. Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2022.
    * - MNIST
      - Image Classification
      - 60k train, 10k test, 10 classes, image shape 28x28x1
      - Li Deng: The MNIST Database of Handwritten Digit Images for Machine Learning Research. IEEE Signal Processing Magazine. 2012.
    * - MultiText
      - Text Classification
      - 115k train, 7.6k test, access to one of four datasets: ag_news, yelp_review_full, dbpedia_14, yahoo_answers_topics
      - Please refer to `the official documentation <https://huggingface.co/datasets>`__.
    * - yearbook
      - Image Classification: gender identification in yearbook photos.
      - ~33k train, ~4k test, 2 classes, years 1930-2013, image shape 32x32x1
      - Huaxiu Yao et al.: Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time. Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2022.
    * - hfd-{dataset_name}
      - multiple
      - Any `Hugging Face dataset <https://huggingface.co/datasets>`__ can be used. Just prepend the prefix ``hfd-``, e.g., ``hfd-rotten_tomatoes``. Select input and target columns via ``config_space``, e.g., add ``"input_column": "text", "target_column": "label"`` for the `rotten_tomatoes <https://huggingface.co/datasets/rotten_tomatoes>`__ example.
      - Please refer to `the official documentation <https://huggingface.co/datasets>`__.

.. _benchmarking-renate-benchmarks-scenarios:

Scenarios
=========

A scenario defines how the dataset is split into several data partitions.
While running the benchmark, the model is trained sequentially on each data partition.
The scenario can be selected by setting :code:`config_space["scenario_name"]` accordingly.
Each scenario might have specific settings which require additional changes of :code:`config_space`.
We will describe those settings in the table below.

The following is an example that uses the class-incremental scenario which splits the entire dataset into 2 parts.
The first part contains all instances with classes 1 and 2, the second with classes 3 and 4.

.. code-block:: python

    config_space["scenario_name"] = "ClassIncrementalScenario"
    config_space["groupings"] = ((1, 2), (3, 4))

.. list-table:: Renate Scenario Overview
    :widths: 15 35 35
    :header-rows: 1

    * - Scenario Name
      - Description
      - Settings
    * - :py:class:`~renate.benchmark.scenarios.DataIncrementalScenario`
      - Used in combination only with :py:class:`~renate.benchmark.datasets.base.DataIncrementalDataModule`,
        e.g., Wild-Time datasets, CLEAR, MultiText, or DomainNet.
        Data is presented data by data, where the data could represent a domain or a time slice.
      - * :code:`num_tasks`: You can provide this argument if the different datasets are identified by
          ids 0 to `num_tasks`. This is the case for time-incremental datasets such as CLEAR or Wild-Time.
        * :code:`data_ids`: Tuple of data identifiers. Used for DomainNet to select order or subset of domains,
          e.g., ``("clipart", "infograph", "painting")``.
        * :code:`groupings`: An alternative to data identifiers that in addition to defining the sequence
          allows to combine different domains to one chunk, e.g., ``(("clipart", ), ("infograph", "painting"))``.
    * - :py:class:`~renate.benchmark.scenarios.ClassIncrementalScenario`
      - Creates data partitions by splitting the data according to class labels.
      - * :code:`groupings`: Tuple of tuples containing the class labels, e.g., ``((1, ), (2, 3, 4))``.
    * - :py:class:`~renate.benchmark.scenarios.FeatureSortingScenario`
      - Splits data into different tasks after sorting the data according to a specific feature.
        Can be used for image data as well. In that case channels are selected and we select according to
        average channel value.
        Random permutations may be applied to have a less strict sorting.
      - * :code:`num_tasks`: Number of data partitions.
        * :code:`feature_idx`: The feature index used for sorting.
        * :code:`randomness`: After sorting, ``0.5 * N * randomness`` random pairs in the sequence are swapped where
          ``N`` is the number of data points. This must be a value between 0 and 1. This allows for creating less strict
          sorted scenarios.
    * - :py:class:`~renate.benchmark.scenarios.HueShiftScenario`
      - A specific scenario only for image data. Very similar to
        :py:class:`~renate.benchmark.scenarios.FeatureSortingScenario` but this scenario sorts according to the hue
        value of an image. Sorting can be less strict by applying random permutations.
      - * :code:`num_tasks`: Number of data partitions.
        * :code:`randomness`: After sorting, ``0.5 * N * randomness`` random pairs in the sequence are swapped where
          ``N`` is the number of data points. This must be a value between 0 and 1. This allows for creating less strict
          sorted scenarios.
    * - :py:class:`~renate.benchmark.scenarios.IIDScenario`
      - Divides the dataset uniformly at random into equally-sized partitions.
      - * :code:`num_tasks`: Number of data partitions.
    * - :py:class:`~renate.benchmark.scenarios.ImageRotationScenario`
      - Creates data partitions by rotating the images by different angles.
      - * :code:`degrees`: Tuple of degrees, e.g., ``(45, 90, 180)``.
    * - :py:class:`~renate.benchmark.scenarios.PermutationScenario`
      - Creates data partitions by randomly permuting the input features.
      - * :code:`num_tasks`: Number of data partitions.
        * :code:`input_dim`: Data dimensionality (tuple or int as string).

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
    :lines: 3-
