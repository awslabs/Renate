Avalanche - Usage and Limitations
*********************************

`Avalanche <https://github.com/ContinualAI/avalanche>`__ is a popular continual learning library for rapid prototyping
and benchmarking algorithms. For that reason, we make some Avalanche updaters available in Renate. Avalanche updaters
can be used in the same way you can use other Renate updaters but they have some limitations you should be aware of.

Usage
=====
Using Avalanche updaters for training works the same as with Renate updaters which we explained earlier in
:doc:`how_to_run_training`. You can select an Avalanche updater by passing the respective string to the ``updater``
argument of :py:func:`~renate.training.training.run_training_job`. The available Avalanche options are
``"Avalanche-ER"``, ``"Avalanche-EWC"``, ``"Avalanche-LwF"``, and ``"Avalanche-iCaRL"``.
More details about these algorithms are given in :doc:`supported_algorithms`.
Further Avalanche updaters can be created if needed.

Limitations
===========
Not all Renate features work with every Avalanche updater. In the following, we list the limitations you face when
using an Avalanche updater. These limitations are subject to change.

PyTorch 1.13.1
--------------
The checkpointing functionality in Avalanche does not work with the latest version of PyTorch 1. Therefore, you will
be required to use PyTorch 1.12.1 instead.

.. warning::
    There is a `known vulnerability <https://nvd.nist.gov/vuln/detail/CVE-2022-45907>`__ with PyTorch <= 1.13.0.

No Scalable Buffers
-------------------
Renate stores the memory buffer on disk. In contrast, Avalanche requires it to be in memory. Therefore, Avalanche
updaters may not work if you intend to use very large buffer sizes.

No Multi-Fidelity HPO
---------------------
Currently, we do not support multi-fidelity hyperparameter optimization with Avalanche updaters. For that reason,
please do not use ``asha`` as a scheduler but use ``random`` or ``bo`` instead.
For more details about HPO in Renate, please refer to the
:ref:`Renate HPO section <getting_started/how_to_run_training:run a training job with hpo>`.

No Early-Stopping
-----------------
Currently, Avalanche updaters will not work with early stopping. Please keep ``early_stopping=True`` (default setting).

iCaRL Limitations
-----------------
The implementation of iCaRL makes strong assumptions about the continual learning scenario. Only classification is
supported and it assumes that data points of a particular class only occur in one update. If this is not the case,
iCaRL will crash as soon as you attempt an update with a class seen before. Furthermore, it requires a specific
model interface to account for its strategy. For that purpose, please create a model class which extends
:py:class:`RenateBenchmarkingModule <renate.benchmark.models.base.RenateBenchmarkingModule>` or copy the relevant
parts over to your ``RenateModule``.
