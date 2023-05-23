Distribution Shift Detection
****************************

Retraining or updating of a machine learning model is usually necessitated by _shifts_ in the
distribution of data that is being served to the model.
Renate provides methods for distribution shift detection that can help you decide when to update
your model.
This functionality resides in the :py:mod:`~renate.shift` subpackage.

Shift Detector Interface
========================

The shift detectors in :py:mod:`~renate.shift` derive from a common class
:py:class:`~renate.shift.detector.ShiftDetector`, which defines the main interface. Once a
:code:`detector` object has been initialized, one calls :code:`detector.fit(dataset_ref)` on a
reference dataset (a PyTorch dataset object). This reference dataset characterizes the expected
data distribution. It may, e.g., be the validation set used during the (previous) fitting of the
model. Subsequently, we can score one or multiple query datasets using the
:code:`detector.score(dataset_query)` method. This method returns a scalar distribution shift score.
We use the convention that high scores indicate a likely distribution shift.

Available Methods
=================

At the moment, Renate provides method for covariate shift detection

* :py:class:`~renate.shift.mmd_detectors.MMDCovariateShiftDetector` uses a kernel MMD test
* :py:class:`~renate.shift.ks_detector.KolmogorovSmirnovCovariateShiftDetector` uses a univariate
    Kolmogorov-Smirnov test on each feature.

Both tests operate on features extracted from the raw data, which is passed using the
:code:`feature_extractor` argument at initialization. The feature extractor is expected to map the
raw input data to informative vectorial representations of moderate dimension. It may be based on
a pretrained model, e.g., using the penultimate layer activations of a ResNet for image data.


Example
=======

The following example applies the MMD covariate shift detector to an example where we simulate a
shift in image data by adding Gaussian blur. We use a pretrained ResNet model as the feature
extractor.

.. literalinclude:: ../../examples/shift_detection/image_shift_detection.py
    :caption: Example
