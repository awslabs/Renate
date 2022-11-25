# Renate - Automatic Neural Networks retraining and Continual Learning in Python

Renate is a Python package for automatic retraining of neural networks models.
It uses advanced Continual Learning and Lifelong Learning algorithms to achieve this purpose. 
The implementation is based on [PyTorch](https://pytorch.org),
and [PyTorch Lightning](https://www.pytorchlightning.ai/).
It also leverages [SyneTune](https://github.com/awslabs/syne-tune) for hyperparameters optimization (HPO) and neural architecture search (NAS).


## Who needs Renate?
In many applications data are made available over time and retraining from scratch for
every new batch of data is prohibitively expensive. In these cases, we would like to use
the new batch of data provided to update our previous model with limited costs.
Unfortunately, since data in different chunks are not sampled according to the same distribution,
just fine-tuning the old model creates problems like "catastrophic forgetting".
The algorithms in Renate help mitigating the negative impact of forgetting and increase the 
model performance overall. 

<div style="text-align: center;">
<img src="https://raw.githubusercontent.com/awslabs/Renate/c62f55046bf1e2d72ef17b3b9fd9df6508ef1f07/doc/_images/improvement_renate.svg" alt="Renate vs Model Fine-Tuning" style="width:80%;" />
</div>
Renate's update mechanisms improve over naive fine-tuning approaches.[^1]

[^1]: test


Renate also offers hyperparameters optimization (HPO), a functionality that can heavily impact
the performance of the model over several retrainings. To do so Renate employs
[Syne Tune](https://github.com/awslabs/syne-tune) under the hood, and can offer
advanced HPO methods such multi-fidelity algorithms (ASHA) and transfer learning algorithms
(useful for speeding up the retuning).

<div style="text-align: center;">
<img src="https://raw.githubusercontent.com/awslabs/Renate/c62f55046bf1e2d72ef17b3b9fd9df6508ef1f07/doc/_images/improvement_tuning.svg" alt="Impact of HPO on Renate's Updating Algorithms" style="width:80%;" />
<p>Renate will benefit from hyperparameter tuning compared to Renate with default settings.</p>
</div>


## Key features
* Easy to scale and run in the cloud
* Advanced HPO and NAS functionalities available out-of-the-box
* Designed for real-world retraining pipelines
* Open for experimentation 


## What are you looking for?
* **Installation instructions**\
`pip install renate` :) or look at the [installation instructions]().
* **Examples showing how to use Renate**\
You can start looking at [this example]() if you want to train your model
locally or to [this one]() if you prefer to use Amazon SageMaker. 
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
