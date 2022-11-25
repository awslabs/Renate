Create Custom Benchmarks
************************

There are several reasons to create a custom benchmark.
For example, you want to use your own dataset, your own model, your own scenario or
your own data splits.
Creating a benchmark config file is very similar to creating the
:doc:`Renate config file <../getting_started/how_to_renate_config>`.
It is not required to build it from scratch but can be build on top of
:py:mod:`~renate.benchmark.experiment_config` in case you want to reuse parts of the
Renate benchmarks.
The main difference is that the object returned by the :code:`model_fn` needs to
follow a slightly different interface.


Your Own Model
==============

If you want to use your own model, simply provide the :code:`model_fn` function as explained in the
:ref:`Renate config chapter <getting_started/how_to_renate_config:model definition>`.

Your Own Data
=============

As the first step, you need to create a :py:class:`~renate.data.data_module.RenateDataModule`
as described in the :ref:`Renate config chapter <getting_started/how_to_renate_config:data preparation>`.
In order to use it, you additionally need to split the train and test data into partitions using
a :py:class:`~renate.benchmark.scenarios.Scenario`.
For this purpose, you can use one of the
:ref:`Renate scenarios <benchmarking/renate_benchmarks:scenarios>` which we described
in the last chapter.
For example, you can combine the :py:class:`~renate.benchmark.scenarios.ClassIncrementalScenario`
with your data module as follows.

.. code-block:: python

    def data_module_fn(
        data_path: Union[Path, str], chunk_id: int, seed: int
    ):
        data_module = CustomDataModule(data_path=data_path, seed=seed)
        return ClassIncrementalScenario(
            data_module=data_module,
            class_groupings=[[0, 1], [2, 3]],
            chunk_id=chunk_id,
        )

Please note, that :code:`data_module_fn` takes :code:`chunk_id` as a special argument which indicates
the partition id. The :code:`chunk_id` will be in range :math:`0\ldots\text{num_updates}-1`
and indicates the partition that should be loaded as training and validation data.


Your Own Scenario
=================

If you want to use a custom scenario, create your own class extending
:py:class:`~renate.benchmark.scenarios.Scenario`.
You will need to implement :py:meth:`~renate.benchmark.scenarios.Scenario.prepare_data()`
such that it assigns the right subset of your data to :code:`self._train_data` and
:code:`self._val_data` based on :code:`self._chunk_id`.
By default, your model will be evaluated on the entire test data after each update step.
If you want to change that, please override :py:meth:`~renate.benchmark.scenarios.Scenario.test_data()`
as well.
Please check the implementation of :py:class:`~renate.benchmark.scenarios.TransformScenario`
for an example.
