# Overview

## Training Algorithms
> **TODO** add list of ModelUpdaters/Learners

## HPO/NAS Optimizers
The HPO and NAS algorithms are available via Syne Tune, the complete description is 
available [here](https://github.com/awslabs/syne-tune/blob/main/docs/schedulers.md).

For most use-cases it will be sufficient to select one option from this list:
* `bo`
* `asha` for Asynchronous Successive Halving. An algorithm using partial evaluations
hyperparameters configurations to speed up the tuning.
* `rush`

More information, including [how to create a search space for HPO/NAS](https://github.com/awslabs/syne-tune/blob/main/docs/search_space.md),
are available in the [Syne Tune FAQs](https://github.com/awslabs/syne-tune/blob/main/docs/faq.md).

## Backends

There are two main backend on which the training jobs can be executed: `local` and `sagemaker`.
* `local` uses the local machine to run the training jobs. It is not important if that is a
personal laptop or a virtual machine in the cloud, but performance will vary.
* `sagemaker` uses Amazon SageMaker to run the training jobs. In this case it will be necessary
to specify the role to be assumed and the type of instance selected. For more information about
getting started with SageMaker, please refer to the [official documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/gs.html).

## Other Components

While algorithms for re-training is the main focus of the library, we do provide other components
that can be used for testing different scenarios or setups. These are not intended to be used 
in production environments but to be used for testing purposes.

### Models
The models are usually available in `/src/renate/benchmark/models/` and we have:
* MultiLayerPerceptron with configurable hidden layers quantity, hidden layers size, and configurable activations
* ResNet in different variants and sizes
* ViT, a vision transformer, in different variants and sizes

### Datasets
> **TODO**: add list of datasets for experimentation

