# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import GaussianBlur

from renate.benchmark.datasets.vision_datasets import TorchVisionDataModule
from renate.data.datasets import _TransformedDataset
from renate.shift.mmd_detectors import MMDCovariateShiftDetector


# Load CIFAR-10 training dataset.
data_module = TorchVisionDataModule(data_path="data", dataset_name="CIFAR10", val_size=0.2)
data_module.prepare_data()
data_module.setup()
dataset = data_module.train_data()

# Use the first 1000 points as reference data.
dataset_ref = torch.utils.data.Subset(dataset, list(range(1000)))

# Use some data points as in-distrubtion query data.
dataset_query_in = torch.utils.data.Subset(dataset, list(range(1000, 2000)))

# We simulate a shift by blurring the images.
transform = GaussianBlur(kernel_size=5, sigma=1.0)
dataset_query_out = _TransformedDataset(dataset_query_in, transform)

# As a feature extractor, we use a pre-trained ResNet and chop off the output layer.
feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
feature_extractor.linear = torch.nn.Identity()  # Replace output layer with identity.
feature_extractor.eval()  # Eval mode to use frozen batchnorm stats.

# Now we can instantiate a shift detector and run the test.
detector = MMDCovariateShiftDetector(feature_extractor=feature_extractor)
print("Fitting detector...")
detector.fit(dataset_ref)
print("Scoring in-distribution data...")
score_in = detector.score(dataset_query_in)
print(f"score = {score_in}")
print("Scoring out-of-distribution data...")
score_out = detector.score(dataset_query_out)
print(f"score = {score_out}")
