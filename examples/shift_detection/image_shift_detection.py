# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import GaussianBlur

from renate.benchmark.datasets.vision_datasets import TorchVisionDataModule
from renate.data.datasets import _TransformedDataset
from renate.shift.mmd_detectors import MMDCovariateShiftDetector

# Load CIFAR-10 training dataset.
data_module = TorchVisionDataModule(data_path="data", dataset_name="CIFAR10", val_size=0.2)
data_module.prepare_data()
data_module.setup()
dataset = data_module.train_data()

# We now generate a reference dataset as well as two query datasets: one from the same distribution,
# and one where we simulate a distribution shift by blurring images. In practice, the reference
# dataset should represent your expected data distribution. It could, e.g., be the validation set
# you used during the previous training of your model.
dataset_ref = torch.utils.data.Subset(dataset, list(range(1000)))
dataset_query_in = torch.utils.data.Subset(dataset, list(range(1000, 2000)))
dataset_query_out = torch.utils.data.Subset(dataset, list(range(2000, 3000)))
transform = GaussianBlur(kernel_size=5, sigma=1.0)
dataset_query_out = _TransformedDataset(dataset_query_in, transform)

# Shift detection methods rely on informative (and relatively low-dimensional) features. Here, we
# use a pretrained ResNet model and chop of its output layer. This leads to 512-dimensional
# vectorial features.
feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
feature_extractor.fc = torch.nn.Identity()
feature_extractor.eval()  # Eval mode to use frozen batchnorm stats.

# Now we can instantiate an MMD-based shift detector. We first fit it to our reference datasets,
# and then score both the in-distribution query dataset and the out-of-distribution query dataset.
# In this toy example, the shift is quite obvious and we will see a very high score for the
# out-of-distribution data.
detector = MMDCovariateShiftDetector(feature_extractor=feature_extractor)
print("Fitting detector...")
detector.fit(dataset_ref)
print("Scoring in-distribution data...")
score_in = detector.score(dataset_query_in)
print(f"score = {score_in}")
print("Scoring out-of-distribution data...")
score_out = detector.score(dataset_query_out)
print(f"score = {score_out}")
