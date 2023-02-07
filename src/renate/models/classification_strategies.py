# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class ClassificationStrategy(ABC):
    @abstractmethod
    def __call__(self, inputs: Tensor, training: bool, **kwargs: Any):
        return inputs, False


class ICaRLClassificationStrategy(ClassificationStrategy):
    def __call__(self, inputs: Tensor, training: bool, class_means: Tensor):
        if training:
            return super().forward(inputs, training)
        normalized_inputs = (inputs.T / torch.norm(inputs.T, dim=0)).T
        return (-torch.cdist(class_means.to(normalized_inputs.device)[:, :].T, normalized_inputs)).T
