# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional, Union

import torch

from datasets import DummyTorchVisionDataModule
from renate.benchmark.models.mlp import MultiLayerPerceptron
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule


def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    if model_state_url is None:
        return MultiLayerPerceptron(5 * 5, 10, 0, 64)
    state_dict = torch.load(str(model_state_url))
    return MultiLayerPerceptron.from_state_dict(state_dict)


def data_module_fn(
    data_path: Union[Path, str], chunk_id: Optional[int] = None, seed: int = 0
) -> RenateDataModule:
    return DummyTorchVisionDataModule(transform=None, val_size=0.9, seed=seed)
