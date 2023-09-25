Working with NLP and Large Language Models
******************************************

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

The function :code:`loss_fn` defines the appropriate loss criterion. As this is a classification 
problem we use :code:`torch.nn.CrossEntropyLoss`.

The data module will return pre-tokenized data and no further transforms are needed in this case.

.. literalinclude:: ../../examples/nlp_finetuning/renate_config.py
    :lines: 3-

Training
========

As in previous examples, we also include a launch script called :code:`start.py`. For more details
on this see previous examples or :doc:`../getting_started/how_to_run_training`.

.. literalinclude:: ../../examples/nlp_finetuning/start.py
    :lines: 3-


Support for training large models
---------------------------------

To support training methods for larger models, we expose two arguments in the 
:code:`run_experiment_job` to enable training on multiple GPUs. For this we exploit the 
strategy functionality provided by `Lightning` 
`large model tutorial <https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html>`_ and 
`documentation <https://lightning.ai/docs/pytorch/stable/extensions/strategy.html>`_. Currently, we 
support 
the strategies:

* `"ddp_find_unused_parameters_false"`
* `"ddp"`
* `"deepspeed"`
* `"deepspeed_stage_1"`
* `"deepspeed_stage_2"`
* `"deepspeed_stage_2_offload"`
* `"deepspeed_stage_3"`
* `"deepspeed_stage_3_offload"`
* `"deepspeed_stage_3_offload_nvme"`

These can be enabled by passing one of the above options to :code:`strategy`. The number of devices 
to be used for parallel training can be specified using :code:`devices` argument which defaults to 
`1`. We also support lower precision training by passing the :code:`precision` argument which 
accepts the options `"16"`, `"32"`, `"64"`, `"bf16"`. Note that it has to be a string and not the
integer `32`. `bf16` is restricted to newer hardware and thus need slightly more attention before 
using it.

See last four lines in the previous code example.

.. literalinclude:: ../../examples/nlp_finetuning/start.py
    :lines: 47-49

