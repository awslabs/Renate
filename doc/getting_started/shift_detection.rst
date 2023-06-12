Distribution Shift Detection
****************************

Retraining or updating of a machine learning model is usually necessitated by *shifts* in the
distribution of data that is being served to the model.
Renate provides methods for distribution shift detection that can help you decide when to update
your model.
This functionality resides in the :py:mod:`renate.shift` subpackage.

Shift Types
===========

In supervised machine learning tasks, one can distinguish different types of shifts in the joint
distribution :math:`p(x, y)`.
A common assumption is that of *covariate shift*, where we assume that :math:`p(x)` changes while
:math:`p(y|x)` stays constant.
In that case, one only needs to inspect :math:`x` data to detect a shift.
Currently, Renate only supports covariate shift detection.

Shift Detector Interface
========================

The shift detectors in :py:mod:`renate.shift` derive from a common class
:py:class:`~renate.shift.detector.ShiftDetector`, which defines the main interface. Once a
:code:`detector` object has been initialized, one calls :code:`detector.fit(dataset_ref)` on a
reference dataset (a PyTorch dataset object). This reference dataset characterizes the expected
data distribution. It may, e.g., be the validation set used during the previous fitting of the
model. Subsequently, we can score one or multiple query datasets using the
:code:`detector.score(dataset_query)` method. This method returns a scalar distribution shift score.
We use the convention that high scores indicate a likely distribution shift. For all currently
available models, this score lies between 0 and 1.

Available Methods
=================

At the moment, Renate provides two method for covariate shift detection

* :py:class:`~renate.shift.mmd_detectors.MMDCovariateShiftDetector` uses a multivariate kernel MMD
  test.
* :py:class:`~renate.shift.ks_detector.KolmogorovSmirnovCovariateShiftDetector` uses a univariate
  Kolmogorov-Smirnov test on each feature, aggregated with a Bonferroni correction.

Both tests operate on features extracted from the raw data, which is passed using the
:code:`feature_extractor` argument at initialization. The feature extractor is expected to map the
raw input data to informative vectorial representations of moderate dimension. It may be based on
a pretrained model, e.g., by using its penultimate-layer embeddings (see also the example below).


Example
=======

The following example illustrates how to apply the MMD covariate shift detector.
We will work with the CIFAR-10 dataset, which we can conveniently load using Renate's
:py:class:`~renate.benchmark.datasets.vision_datasets.TorchVisionDataModule`.
In practice, you would ingest your own data here, see the documentation for
:py:class:`~renate.data.data_module.RenateDataModule`.

.. literalinclude:: ../../examples/shift_detection/image_shift_detection.py
    :lines: 12-15

For the purpose of this demonstration, we now generate a reference dataset as well as two query
datasets: one from the same distribution, and one where we simulate a distribution shift by
blurring images.
In practice, the reference dataset should represent your expected data distribution.
It could, e.g., be the validation set you used during the previous training of your model.
The query dataset would be the data you want to check for distribution shift, e.g., data collected
during the deployment of your model.

.. literalinclude:: ../../examples/shift_detection/image_shift_detection.py
    :lines: 21-25

Shift detection methods rely on informative (and relatively low-dimensional) features.
Here, we use a pretrained ResNet model and chop of its output layer.
This leads to 512-dimensional vectorial features.

.. literalinclude:: ../../examples/shift_detection/image_shift_detection.py
    :lines: 30-32

You can use any :py:class:`torch.nn.Module`, which may be a pretrained model or use a custom model
that has been trained on the data at hand.
Generally, we have observed very good result when using generic pre-trained models such as ResNets
for image data or BERT models for text.

Now we can instantiate an MMD-based shift detector. We first fit it to our reference datasets and
then score both the in-distribution query dataset as well as the out-of-distribution query dataset.

.. literalinclude:: ../../examples/shift_detection/image_shift_detection.py
    :lines: 38-46

In this toy example, the shift is quite obvious and we will see a very high score for the
out-of-distribution data::

    Fitting detector...
    Scoring in-distribution data...
    score = 0.5410000085830688
    Scoring out-of-distribution data...
    score = 1.0
