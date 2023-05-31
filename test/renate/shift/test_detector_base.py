# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.shift.detector import ShiftDetectorWithFeatureExtractor


@pytest.mark.parametrize(
    "dataset",
    [
        torch.utils.data.TensorDataset(torch.randn(20, 2)),
        torch.utils.data.TensorDataset(torch.randn(20, 2), torch.randint(0, 10, size=(20,))),
    ],
)
@pytest.mark.parametrize("feature_extractor", [torch.nn.Linear(2, 4)])
def test_extract_features(dataset, feature_extractor):
    detector = ShiftDetectorWithFeatureExtractor(feature_extractor=feature_extractor)
    features = detector.extract_features(dataset)
    assert features.size() == (20, 4)
