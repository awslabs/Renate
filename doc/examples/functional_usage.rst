Using Renate Functionally
*************************

Usually, we use Renate by writing a :code:`renate_config.py` and launching training jobs via the
:py:func:`~renate.training.training.run_training_job` function. In this example, we demonstrate how
to write your own training script and use renate in a functional way. This can be useful, e.g., for
debugging new components.

Here, we use Renate to fine-tune a pretrained Transformer model on a sequence classification
dataset. First, we create the model and a loss function. Since this is a static model, we simply
wrap it using the :py:class:`~renate.models.renate_module.RenateWrapper` class.

.. literalinclude:: ../../examples/functional_usage/train.py
    :lines: 12-16

Next, we prepare the dataset on which we want to fine-tune the model. Here, we use the
:py:class:`~renate.benchmark.datasets.nlp_datasets.HuggingFaceTextDataModule` to load the
:code:`"imdb"` dataset from the Hugging Face hub. This will also take care of tokenization for us,
if we pass it the corresponding tokenizer.

.. literalinclude:: ../../examples/functional_usage/train.py
    :lines: 19-24

Now we can instantiate a :py:class:`~renate.updaters.model_updater.ModelUpdater` to perform the
training. Since we just want to fine-tune the model on a single dataset here, we use the
:py:class:`~renate.updaters.experimental.fine_tuning.FineTuningModelUpdater`. We pass our
:code:`model` as well as training details, such as the optimizer to use as well as its
hyperparameters. Once the model updater is created, we initiate the training by calling its
:py:meth:`~renate.updaters.model_updater.ModelUpdater.update` method and passing training and
(optionally) validation datasets.

.. literalinclude:: ../../examples/functional_usage/train.py
    :lines: 27-37

Once the training is terminated, your model is ready to deploy. Here, we just save its weights for
later use.

.. literalinclude:: ../../examples/functional_usage/train.py
    :lines: 41-
