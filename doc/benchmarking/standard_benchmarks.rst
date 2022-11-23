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

    from renate.benchmark import execute_experiment_job, experimentation_config_file

    execute_experiment_job(
        backend="local",
        config_file=experimentation_config_file(),
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

The full list of models and model names including a short description is provided in the following table.

.. list-table:: Renate Model Overview
    :header-rows: 1

    * - Model Name
      - Description
      - API Reference
    * - MultiLayerPerceptron
      - Neural network consisting of a sequence of dense layers.
      - :py:class:`renate.benchmark.models.mlp.MultiLayerPerceptron`
    * - ResNet18
      - 18 layer `ResNet <https://arxiv.org/pdf/1512.03385.pdf>`_ CNN architecture
      - :py:class:`renate.benchmark.models.resnet.ResNet18`
    * - ResNet34
      - 34 layer `ResNet <https://arxiv.org/pdf/1512.03385.pdf>`_ CNN architecture
      - :py:class:`renate.benchmark.models.resnet.ResNet34`
    * - ResNet50
      - 50 layer `ResNet <https://arxiv.org/pdf/1512.03385.pdf>`_ CNN architecture
      - :py:class:`renate.benchmark.models.resnet.ResNet50`
    * - ResNet18CIFAR
      - 18 layer `ResNet <https://arxiv.org/pdf/1512.03385.pdf>`_ CNN architecture for small image sizes (approx 32x32)
      - :py:class:`renate.benchmark.models.resnet.ResNet18CIFAR`
    * - ResNet34CIFAR
      - 34 layer `ResNet <https://arxiv.org/pdf/1512.03385.pdf>`_ CNN architecture for small image sizes (approx 32x32)
      - :py:class:`renate.benchmark.models.resnet.ResNet34CIFAR`
    * - ResNet50CIFAR
      - 50 layer `ResNet <https://arxiv.org/pdf/1512.03385.pdf>`_ CNN architecture for small image sizes (approx 32x32)
      - :py:class:`renate.benchmark.models.resnet.ResNet50CIFAR`
    * - VisionTransformerCIFAR
      - Base `Vision Transformer <https://arxiv.org/pdf/2010.11929.pdf>`_ architecture for images of size 32x32 with patch size 4.
      - :py:class:`renate.benchmark.models.vision_transformer.VisionTransformerCIFAR`
    * - VisionTransformerB16
      - Base `Vision Transformer <https://arxiv.org/pdf/2010.11929.pdf>`_ architecture for images of size 224x224 with patch size 16.
      - :py:class:`renate.benchmark.models.vision_transformer.VisionTransformerB16`
    * - VisionTransformerB32
      - Base `Vision Transformer <https://arxiv.org/pdf/2010.11929.pdf>`_ architecture for images of size 224x224 with patch size 32.
      - :py:class:`renate.benchmark.models.vision_transformer.VisionTransformerB32`
    * - VisionTransformerL16
      - Large `Vision Transformer <https://arxiv.org/pdf/2010.11929.pdf>`_ architecture for images of size 224x224 with patch size 16.
      - :py:class:`renate.benchmark.models.vision_transformer.VisionTransformerL16`
    * - VisionTransformerL32
      - Large `Vision Transformer <https://arxiv.org/pdf/2010.11929.pdf>`_ architecture for images of size 224x224 with patch size 32.
      - :py:class:`renate.benchmark.models.vision_transformer.VisionTransformerL32`
    * - VisionTransformerH14
      - Huge `Vision Transformer <https://arxiv.org/pdf/2010.11929.pdf>`_ architecture for images of size 224x224 with patch size 14.
      - :py:class:`renate.benchmark.models.vision_transformer.VisionTransformerH14`


.. _benchmarking-standard-benchmarks-datasets:

Datasets
========

.. _benchmarking-standard-benchmarks-scenarios:

Scenarios
=========

