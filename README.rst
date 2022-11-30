.. image:: https://img.shields.io/pypi/status/Renate
    :target: #
    :alt: PyPI - Status
.. image:: https://img.shields.io/github/v/release/awslabs/Renate
    :target: https://github.com/awslabs/Renate/releases/tag/v0.1.0
    :alt: Latest Release
.. image:: https://img.shields.io/pypi/dm/Renate
    :target: https://pypistats.org/packages/renate
    :alt: PyPI - Downloads
.. image:: https://img.shields.io/github/license/awslabs/Renate
    :target: https://github.com/awslabs/Renate/blob/main/LICENSE
    :alt: License
.. image:: https://readthedocs.org/projects/renate/badge/?version=latest
    :target: https://renate.readthedocs.io
    :alt: Documentation Status

Renate: Automatic Neural Networks Retraining and Continual Learning in Python
******************************************************************************

Renate is a Python package for automatic retraining of neural networks models.
It uses advanced Continual Learning and Lifelong Learning algorithms to achieve this purpose. 
The implementation is based on `PyTorch <https://pytorch.org>`_
and `Lightning <https://www.pytorchlightning.ai>`_ for deep learning, and
`Syne Tune <https://github.com/awslabs/syne-tune>`_ for hyperparameter optimization.

Quick links
===========
* Install renate with `pip install renate` or look at `these instructions <https://renate.readthedocs.io/en/latest/getting_started/install.html>`_
* Examples for `local training <https://renate.readthedocs.io/en/latest/examples/train_mlp_locally.html>`_ and `training on Amazon SageMaker <https://renate.readthedocs.io/en/latest/examples/train_classifier_sagemaker.html>`_.
* `Documentation <https://renate.readthedocs.io>`_
* `Supported Algorithms <https://renate.readthedocs.io/en/latest/getting_started/supported_algorithms.html>`_


Who needs Renate?
=================

In many applications data is made available over time and retraining from scratch for
every new batch of data is prohibitively expensive. In these cases, we would like to use
the new batch of data provided to update our previous model with limited costs.
Unfortunately, since data in different chunks is not sampled according to the same distribution,
just fine-tuning the old model creates problems like *catastrophic forgetting*.
The algorithms in Renate help mitigating the negative impact of forgetting and increase the 
model performance overall. 

.. figure:: https://raw.githubusercontent.com/awslabs/Renate/main/doc/_images/improvement_renate.svg
    :scale: 80%
    :align: center
    :alt: Renate vs Model Fine-Tuning.

    Renate's update mechanisms improve over naive fine-tuning approaches. [#]_

Renate also offers hyperparameter optimization (HPO), a functionality that can heavily impact
the performance of the model when continuously updated. To do so, Renate employs
`Syne Tune <https://github.com/awslabs/syne-tune>`_ under the hood, and can offer
advanced HPO methods such multi-fidelity algorithms (ASHA) and transfer learning algorithms
(useful for speeding up the retuning).

.. figure:: https://raw.githubusercontent.com/awslabs/Renate/main/doc/_images/improvement_tuning.svg
    :scale: 80%
    :align: center
    :alt: Impact of HPO on Renate's Updating Algorithms.

    Renate will benefit from hyperparameter tuning compared to Renate with default settings. [#]_


Key features
============

* Easy to scale and run in the cloud
* Designed for real-world retraining pipelines
* Advanced HPO functionalities available out-of-the-box
* Open for experimentation 


What are you looking for?
=========================

* `Installation Instructions <https://renate.readthedocs.io/en/latest/getting_started/install.html>`_
    .. code-block:: bash

      pip install renate

* Examples:
    * `Train an MLP locally on MNIST <https://renate.readthedocs.io/en/latest/examples/train_mlp_locally.html>`_
    * `Train a ResNet on SageMaker <https://renate.readthedocs.io/en/latest/examples/train_classifier_sagemaker.html>`_
* `Documentation website with API doc and examples <https://renate.readthedocs.io>`_
* `List of the supported algorithms <https://renate.readthedocs.io/en/latest/getting_started/supported_algorithms.html>`_
* `How to run continual learning experiments using Renate <https://renate.readthedocs.io/en/latest/benchmarking/index.html>`_
* `Guidelines for Contributors <https://github.com/awslabs/renate/tree/master/CONTRIBUTING.md>`_

If you did not find what you were looking for, open an `issue <https://github.com/awslabs/Renate/issues/new>`_ and
we will do our best to improve the documentation.


.. [#] To create this plot, we simulated class-incremental learning with CIFAR-10.
    The training data was divided into 5 partitions, and we trained sequentially on them.
    Fine-tuning refers to the strategy to learn on the first partition from scratch, and
    train on each of the subsequent partitions for few epochs only.
    We compare to Experience Replay with a memory size of 500.
    For both methods we use the same number of epochs and choose the best checkpoint
    using a validation set.
    Results reported are on the test set.

.. [#] The setup is the same as in the last experiment. However, this time we compare
    Experience Replay against a version in which its hyperparameters were tuned.
