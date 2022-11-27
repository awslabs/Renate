Renate - Automatic Neural Networks Retraining and Continual Learning in Python
******************************************************************************

Renate is a Python package for automatic retraining of neural networks models.
It uses advanced Continual Learning and Lifelong Learning algorithms to achieve this purpose. 
The implementation is based on `PyTorch <https://pytorch.org>`_
and `Lightning <https://www.pytorchlightning.ai>`_ for deep learning, and
`Syne Tune <https://github.com/awslabs/syne-tune>`_ for hyperparameter optimization.


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

* **Installation instructions**\
    `pip install renate` :) or look at the [installation instructions]().
* **Examples how to use Renate**\
    We provide examples to train your model :doc:`locally <../examples/train_mlp_locally>`
    or :doc:`using Amazon SageMaker <examples/train_classifier_sagemaker>`.
* **Supported algorithms**\
    A list of the supported algorithms is available [here]().
* **Experimenting with Renate**\
    A tutorial on how to run experiments with Renate is available [here]().
* **Documentation**\
    All the documentation you may need is available on [here]().
* **Guidelines for contributors**\
    If you wish to contribute to the project, please refer to our
    [contribution guidelines](https://github.com/awslabs/renate/tree/master/CONTRIBUTING.md).
* **You did not find what you were looking for?**\
    Open an [issue](https://github.com/awslabs/Renate/issues/new) and we will do our best to improve the documentation.


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
