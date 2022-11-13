# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import FashionMNIST

from renate.benchmark.datasets.nlp_datasets import TorchTextDataModule
from renate.benchmark.datasets.vision_datasets import (
    CLEARDataModule,
    CORE50DataModule,
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
    "src_bucket, src_object_name, download",
    [
        ("mnemosyne-team-bucket", "FashionMNIST", True),
        (None, None, True),
        ("mnemosyne-team-bucket", "FashionMNIST", False),
        (None, None, False),
    ],
)
def test_torchvision_data_module_loading(tmpdir, src_bucket, src_object_name, download):
    """Uses TorchVisionDataModule to load data from s3 or original source.

    There are four different cases. Cases with `download=True` check whether download from different sources work.
    These tests are successful if the downloaded data has the expected size.
    Cases with `download=False` test whether the download flag is respected. In this case the test is successful if the
    execution of the data preparation raises a RuntimeError.
    """
    dataset_name = "FashionMNIST"
    val_size = 0.35
    torchvision_data_module = TorchVisionDataModule(
        tmpdir,
        src_bucket=src_bucket,
        src_object_name=src_object_name,
        dataset_name=dataset_name,
        download=download,
        val_size=val_size,
    )
    torchvision_data_module.prepare_data()
    if not download:
        with pytest.raises(RuntimeError):
            torchvision_data_module.setup()
        return
    torchvision_data_module.setup()
    train_data = torchvision_data_module.train_data()
    val_data = torchvision_data_module.val_data()
    test_data = torchvision_data_module.test_data()

    assert len(train_data) == round(60000 * (1 - val_size))
    assert isinstance(train_data, Dataset)

    assert len(val_data) == round(60000 * val_size)
    assert isinstance(val_data, Dataset)

    assert len(test_data) == 10000
    assert isinstance(test_data, FashionMNIST)


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_module_cls,data_module_kwargs,train_len,test_len,src_bucket,src_object_name",
    [
        [
            CLEARDataModule,
            {"dataset_name": "CLEAR10", "val_size": 0.4, "chunk_id": 1},
            2986,
            500,
            "mnemosyne-team-bucket",
            os.path.join("dataset/", "clear10"),
        ],
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
            "mnemosyne-team-bucket",
            os.path.join("dataset/", "tiny_imagenet"),
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
            "mnemosyne-team-bucket",
            os.path.join("dataset/", "clear100"),
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
            CORE50DataModule,
            {"scenario": "ni", "val_size": 0.4, "chunk_id": 1},
            119894,
            44972,
            "mnemosyne-team-bucket",
            os.path.join("dataset/", "core50"),
        ],
        [
            CORE50DataModule,
            {"scenario": "nc", "val_size": 0.4, "chunk_id": 5},
            119894,
            44972,
            "mnemosyne-team-bucket",
            os.path.join("dataset/", "core50"),
        ],
        [
            CORE50DataModule,
            {"scenario": "ni", "val_size": 0.4, "chunk_id": 1},
            119894,
            44972,
            None,
            None,
        ],
        [
            CORE50DataModule,
            {"scenario": "nc", "val_size": 0.4, "chunk_id": 5},
            119894,
            44972,
            None,
            None,
        ],
        [
            TorchVisionDataModule,
            {"dataset_name": "FashionMNIST", "val_size": 0.33, "download": True},
            60000,
            10000,
            None,
            None,
        ],
        [
            TorchVisionDataModule,
            {"dataset_name": "CIFAR10", "val_size": 0.33, "download": True},
            50000,
            10000,
            None,
            None,
        ],
        [
            TorchVisionDataModule,
            {"dataset_name": "CIFAR100", "val_size": 0.33, "download": True},
            50000,
            10000,
            None,
            None,
        ],
        [
            TorchVisionDataModule,
            {"dataset_name": "MNIST", "val_size": 0.33, "download": True},
            60000,
            10000,
            None,
            None,
        ],
        [
            TorchTextDataModule,
            {"dataset_name": "AmazonReviewFull", "val_size": 0.21},
            3000000,
            650000,
            "mnemosyne-team-bucket",
            os.path.join("datasets/", "amazon_review_full_csv"),
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
            "mnemosyne-team-bucket",
            os.path.join("datasets/", "dbpedia_csv"),
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
            "mnemosyne-team-bucket",
            os.path.join("datasets/", "ag_news_csv"),
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
