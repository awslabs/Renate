# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class PredictionStrategy(ABC):
    @abstractmethod
    def __call__(self, inputs: Tensor, training: bool, **kwargs: Any) -> Tensor:
        return inputs


class ICaRLClassificationStrategy(PredictionStrategy):
    def __call__(self, inputs: Tensor, training: bool, class_means: Tensor) -> Tensor:
        if training:
            return super().__call__(inputs, training)
        normalized_inputs = (inputs.T / torch.norm(inputs.T, dim=0)).T
        return (-torch.cdist(class_means.to(normalized_inputs.device)[:, :].T, normalized_inputs)).T
