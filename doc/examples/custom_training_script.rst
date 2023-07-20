Using Renate in a Custom Training Script
****************************************

Usually, we use Renate by writing a :code:`renate_config.py` and launching training jobs via the
:py:func:`~renate.training.training.run_training_job` function. In this example, we demonstrate how
to write your own training script and use renate in a functional way. This can be useful, e.g., for
debugging new components.

Here, we use Renate to fine-tune a pretrained Transformer model on a sequence classification
dataset. First, we create the model and a loss function. Since this is a static model, we simply
wrap it using the :py:class:`~renate.models.renate_module.RenateWrapper` class. Recall that loss
functions should produce one loss value per input example (:code:`reduction="none"` for PyTorch's
built-in losses), as explained in :ref:`getting_started/how_to_renate_config:Loss Definition`.

.. literalinclude:: ../../examples/custom_training_script/train.py
    :lines: 14-18

Next, we prepare the dataset on which we want to fine-tune the model. Here, we use the
:py:class:`~renate.benchmark.datasets.nlp_datasets.HuggingFaceTextDataModule` to load the
:code:`"imdb"` dataset from the Hugging Face hub. This will also take care of tokenization for us,
if we pass it the corresponding tokenizer.

.. literalinclude:: ../../examples/custom_training_script/train.py
    :lines: 21-26

Now we can instantiate a :py:class:`~renate.updaters.model_updater.ModelUpdater` to perform the
training. Since we just want to fine-tune the model on a single dataset here, we use the
:py:class:`~renate.updaters.experimental.fine_tuning.FineTuningModelUpdater`. We pass our
:code:`model` as well as training details, such as the optimizer to use and its hyperparameters.
The model updater also receives all options related to distributed training, as explained in
:ref:`examples/nlp_finetuning:Support for training large models`.
Once the model updater is created, we initiate the training by calling its
:py:meth:`~renate.updaters.model_updater.ModelUpdater.update` method and passing training and
(optionally) validation datasets.

.. literalinclude:: ../../examples/custom_training_script/train.py
    :lines: 29-39

Once the training is terminated, your model is ready to deploy. Here, we just save its weights for
later use using standard
`PyTorch functionality <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_.

.. literalinclude:: ../../examples/custom_training_script/train.py
    :lines: 43
