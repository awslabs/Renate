Working with NLP
****************

This example demonstrates how to use Renate to train NLP models. We will train a sequence classifier
to distinguish between positive and negative movie reviews. Using Renate, we will sequentially
train this model on two movie review datasets, called :code:`"imdb"` and :code:`"rotten_tomatoes"`.

Configuration
=============

Let us take a look at the :code:`renate_config.py` for this example. In the :code:`model_fn`
function, we use the Hugging Face :code:`transformers` library to instantiate a sequence
classification model. Since this model is static, we can easily turn it into a :code:`RenateModule`
by wrapping it in :py:class:`~renate.models.renate_module.RenateWrapper`.

In the :code:`data_module_fn`, we load the matching tokenizer from the :code:`transformers` library.
We then use Renate's :py:class:`~renate.benchmark.datasets.nlp_datasets.HuggingfaceTextDataModule`
to access datasets from the `Hugging Face datasets hub <https://huggingface.co/datasets>`_. This
data module expects the name of a dataset as well as a tokenizer. Here, we load the :code:`"imdb"`
dataset in the first training stage (:code:`chunk_id = 0`) and the :code:`"rotten_tomatoes"` dataset
for the subsequent model update (:code:`chunk_id = 1`).

The data module will return pre-tokenized data and no further transforms are needed in this case.

.. literalinclude:: ../../examples/nlp_finetuning/renate_config.py
    :lines: 3-

Training
========

As in previous examples, we also include a launch script called :code:`start.py`. For more details
on this see previous examples or :doc:`../getting_started/how_to_run_training`.

.. literalinclude:: ../../examples/nlp_finetuning/start.py
    :lines: 3-



