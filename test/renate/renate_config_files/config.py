# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional, Union

import torch

from dummy_datasets import DummyTorchVisionDataModule
from renate.benchmark.models.mlp import MultiLayerPerceptron
from renate.benchmark.scenarios import ClassIncrementalScenario
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule


def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    if model_state_url is None:
        return MultiLayerPerceptron(5 * 5, 10, 0, 64)
    state_dict = torch.load(str(model_state_url))
    return MultiLayerPerceptron.from_state_dict(state_dict)


def data_module_fn(
    data_path: Union[Path, str],
    chunk_id: Optional[int] = None,
    data_module_fn_val_size: str = "0.0",
    seed: int = 0,
    data_module_fn_use_scenario: str = "False",
) -> RenateDataModule:
    data_module = DummyTorchVisionDataModule(
        transform=None, val_size=float(data_module_fn_val_size), seed=seed
    )
    if data_module_fn_use_scenario == "True":
        return ClassIncrementalScenario(
            data_module=data_module,
            chunk_id=chunk_id,
            class_groupings=[[0, 1], [2, 3, 4]],
        )
    return data_module
