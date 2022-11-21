# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import Dataset, TensorDataset

from renate.benchmark.datasets.nlp_datasets import TorchTextDataModule
from renate.benchmark.datasets.vision_datasets import (
    CLEARDataModule,
    TinyImageNetDataModule,
    TorchVisionDataModule,
)
from renate.data.data_module import CSVDataModule
from renate.utils.pytorch import _proportions_into_sizes


def test_csv_data_module(tmpdir):
    target_name = "y"

    # Create toy data
    features = np.random.randint(10, size=(10, 4))
    train_data = pd.DataFrame(features, columns=list("ABCD"), dtype="float")
    train_data[target_name] = np.random.randint(10, size=(10, 1))

    features = np.random.randint(10, size=(3, 4))
    test_data = pd.DataFrame(features, columns=list("ABCD"), dtype="float")
    test_data[target_name] = np.random.randint(10, size=(3, 1))

    for stage in ["train", "test"]:
        data = train_data if stage == "train" else test_data
        data_file_tmp = os.path.join(tmpdir, f"{stage}.csv")
        data.to_csv(data_file_tmp, index=False)

    val_size = 0.2
    csv_module = CSVDataModule(
        data_path=tmpdir,
        filenames={"train": "train.csv", "test": "test.csv"},
        target_name=target_name,
        val_size=val_size,
    )
    csv_module.prepare_data()
    csv_module.setup()
    train_data = csv_module.train_data()
    val_data = csv_module.val_data()
    test_data = csv_module.test_data()

    assert len(train_data) == round(10 * (1 - val_size))
    assert isinstance(train_data, Dataset)

    assert len(val_data) == round(10 * val_size)
    assert isinstance(val_data, Dataset)

    assert len(test_data) == 3
    assert isinstance(test_data, TensorDataset)


@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset_name,num_tr,num_te,x_shape",
    [
        ("MNIST", 60000, 10000, (1, 28, 28)),
        ("FashionMNIST", 60000, 10000, (1, 28, 28)),
        ("CIFAR10", 50000, 10000, (3, 32, 32)),
        ("CIFAR100", 50000, 10000, (3, 32, 32)),
    ]
)
def test_torchvision_data_module(tmpdir, dataset_name, num_tr, num_te, x_shape):
    """Test downloading of Torchvision data."""
    val_size = 0.35
    data_module = TorchVisionDataModule(
        tmpdir,
        dataset_name=dataset_name,
        download=True,
        val_size=val_size,
    )
    data_module.prepare_data()
    data_module.setup()
    train_data = data_module.train_data()
    val_data = data_module.val_data()
    test_data = data_module.test_data()
    assert len(train_data) == round(num_tr * (1 - val_size))
    assert isinstance(train_data, Dataset)
    assert len(val_data) == round(num_tr * val_size)
    assert isinstance(val_data, Dataset)
    assert len(test_data) == num_te
    assert isinstance(test_data, Dataset)
    assert train_data[0][0].size() == test_data[0][0].size() == x_shape


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_module_cls,data_module_kwargs,train_len,test_len,src_bucket,src_object_name",
    [
        [
            CLEARDataModule,
            {"dataset_name": "CLEAR10", "val_size": 0.2, "chunk_id": 1},
            2986,
            500,
            None,
            None,
        ],
        [
            TinyImageNetDataModule,
            {"val_size": 0.2},
            100000,
            10000,
            None,
            None,
        ],
        [
            CLEARDataModule,
            {"dataset_name": "CLEAR100", "val_size": 0.4, "chunk_id": 1},
            9945,
            4984,
            None,
            None,
        ],
        [
            TorchTextDataModule,
            {"dataset_name": "AmazonReviewFull", "val_size": 0.21},
            3000000,
            650000,
            None,
            None,
        ],
        [
            TorchTextDataModule,
            {"dataset_name": "DBpedia", "val_size": 0.21},
            560000,
            70000,
            None,
            None,
        ],
        [
            TorchTextDataModule,
            {"dataset_name": "AG_NEWS", "val_size": 0.21},
            120000,
            7600,
            None,
            None,
        ],
    ],
)
def test_custom_data_module(
    tmpdir, data_module_cls, data_module_kwargs, train_len, test_len, src_bucket, src_object_name
):
    val_size = data_module_kwargs["val_size"]
    data_module = data_module_cls(
        tmpdir,
        src_bucket,
        src_object_name,
        **data_module_kwargs,
    )
    data_module.prepare_data()
    data_module.setup()
    train_data = data_module.train_data()
    val_data = data_module.val_data()
    test_data = data_module.test_data()

    assert isinstance(train_data, Dataset)
    assert isinstance(val_data, Dataset)
    assert isinstance(test_data, Dataset)
    _train_len, _val_len = _proportions_into_sizes([1 - val_size, val_size], train_len)
    assert len(train_data) == _train_len
    assert len(val_data) == _val_len
    assert len(test_data) == test_len
