# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from collections import Counter

import numpy as np
import pytest
import torch
from torchvision.transforms.functional import rotate

from datasets import DummyTorchVisionDataModule, DummyTorchVisionDataModuleWithChunks
from renate.benchmark.datasets.vision_datasets import TorchVisionDataModule
from renate.benchmark.scenarios.data_module_modification import (
    BenchmarkScenario,
    ClassIncrementalScenario,
    ImageRotationScenario,
    PermutationScenario,
)
from renate.utils.pytorch import randomly_split_data


@pytest.mark.parametrize(
    "scenario_cls,kwargs",
    [
        [
            ImageRotationScenario,  # Wrong chunk id
            {"num_tasks": 3, "chunk_id": 4, "degrees": [0, 60, 180]},
        ],
        [
            ClassIncrementalScenario,
            {
                "num_tasks": 3,
                "chunk_id": 4,  # Wrong chunk id
                "class_groupings": [[0, 1, 2], [3, 5, 9], [4, 6, 7, 8]],
            },
        ],
        [PermutationScenario, {"num_tasks": 3, "chunk_id": 2}],  # Missing input dim
    ],
)
def test_failing_to_init(tmpdir, scenario_cls, kwargs):
    dataset_name = "FashionMNIST"
    data_module = TorchVisionDataModule(
        tmpdir, src_bucket=None, src_object_name=None, dataset_name=dataset_name, download=True
    )
    with pytest.raises(Exception):
        scenario_cls(data_module=data_module, **kwargs)


def test_class_incremental_scenario():
    data_module = DummyTorchVisionDataModule(val_size=0.3, seed=42)
    class_groupings = [[0, 1, 3], [2], [3, 4]]
    train_data_class_counts = Counter({3: 16, 4: 15, 0: 15, 2: 13, 1: 11})
    val_data_class_counts = Counter({1: 9, 2: 7, 4: 5, 0: 5, 3: 4})
    test_data_class_counts = Counter({0: 20, 1: 20, 2: 20, 3: 20, 4: 20})
    for i in range(len(class_groupings)):
        scenario = ClassIncrementalScenario(
            data_module=data_module, num_tasks=3, class_groupings=class_groupings, chunk_id=i
        )
        scenario.prepare_data()
        for stage, class_counts in zip(
            ["train", "val"], [train_data_class_counts, val_data_class_counts]
        ):
            scenario.setup(stage)
            data = getattr(scenario, f"{stage}_data")()
            assert len(data) == sum([class_counts[c] for c in class_groupings[i]])

        scenario.setup("test")
        for j, test_data in enumerate(scenario.test_data()):
            assert len(test_data) == sum([test_data_class_counts[c] for c in class_groupings[j]])


def test_image_rotation_scenario():
    data_module = DummyTorchVisionDataModule(val_size=0.3)
    degrees = [15, 75]
    data_module.prepare_data()
    data_module.setup()
    orig_data_module = copy.deepcopy(data_module)

    for i in range(len(degrees)):
        scenario = ImageRotationScenario(
            data_module=data_module,
            num_tasks=2,
            degrees=degrees,
            chunk_id=i,
            seed=data_module._seed,
        )
        for stage in ["train", "val"]:
            scenario.setup(stage)
            scenario_data = getattr(scenario, f"{stage}_data")()
            orig_data_module_data = getattr(orig_data_module, f"{stage}_data")()
            split_orig_data_module_data = randomly_split_data(
                orig_data_module_data, [0.5, 0.5], seed=orig_data_module._seed
            )[i]
            assert len(scenario_data) == len(split_orig_data_module_data)
            for j in range(len(scenario_data)):
                assert torch.equal(
                    rotate(split_orig_data_module_data[j][0], degrees[i]), scenario_data[j][0]
                )

        scenario.setup("test")
        for j, test_data in enumerate(scenario.test_data()):
            assert len(test_data) == len(orig_data_module.test_data())
            for k in range(len(test_data)):
                assert torch.equal(
                    rotate(orig_data_module.test_data()[k][0], degrees[j]), test_data[k][0]
                )


def test_permutation_scenario():
    data_module = DummyTorchVisionDataModule(val_size=0.3)
    data_module.prepare_data()
    data_module.setup()
    orig_data_module = copy.deepcopy(data_module)
    permutation_indices = torch.tensor(
        [
            list(range(np.prod(data_module.input_shape)))[::-1],
            list(range(np.prod(data_module.input_shape))),
        ]
    )

    for i in range(len(permutation_indices)):
        scenario = PermutationScenario(
            data_module=data_module,
            num_tasks=3,
            chunk_id=i,
            input_dim=np.prod(data_module.input_shape),
            seed=data_module._seed,
        )
        # Chunk id 0 is a special case
        current_permutation_index = (
            permutation_indices[i - 1]
            if i > 0
            else torch.tensor(list(range(np.prod(data_module.input_shape))))
        )
        scenario._indices = permutation_indices
        for stage in ["train", "val"]:
            scenario.setup(stage)
            scenario_data = getattr(scenario, f"{stage}_data")()
            orig_data_module_data = getattr(orig_data_module, f"{stage}_data")()
            split_orig_data_module_data = randomly_split_data(
                orig_data_module_data, [1 / 3 for _ in range(3)], seed=orig_data_module._seed
            )[i]
            assert len(scenario_data) == len(split_orig_data_module_data)
            for j in range(len(scenario_data)):
                orig_shape = split_orig_data_module_data[j][0].shape
                assert torch.equal(
                    split_orig_data_module_data[j][0]
                    .view(-1)[current_permutation_index]
                    .reshape(orig_shape),
                    scenario_data[j][0],
                )

        scenario.setup("test")
        for j, test_data in enumerate(scenario.test_data()):
            assert len(test_data) == len(orig_data_module.test_data())
            # Chunk id 0 is a special case
            current_permutation_index = (
                permutation_indices[j - 1]
                if j > 0
                else torch.tensor(list(range(np.prod(data_module.input_shape))))
            )
            for k in range(len(test_data)):
                orig_shape = orig_data_module.test_data()[k][0].shape
                assert torch.equal(
                    orig_data_module.test_data()[k][0]
                    .view(-1)[current_permutation_index]
                    .reshape(orig_shape),
                    test_data[k][0],
                )


def test_benchmark_scenario():
    data_module = DummyTorchVisionDataModuleWithChunks(num_chunks=3, val_size=0.2)
    scenario = BenchmarkScenario(data_module=data_module, num_tasks=3, chunk_id=0)

    scenario.prepare_data()

    for chunk_id in range(3):
        scenario.setup(chunk_id=chunk_id)

        assert scenario.train_data() is not None
        assert scenario.val_data() is not None
        assert len(scenario.test_data()) == 3
