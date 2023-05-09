# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pandas as pd
import pytest
import torch
import transformers
from torch.utils.data import Dataset, TensorDataset

from renate.benchmark.datasets.nlp_datasets import HuggingFaceTextDataModule
from renate.benchmark.datasets.vision_datasets import (
    CLEARDataModule,
    TinyImageNetDataModule,
    TorchVisionDataModule,
)
from renate.data import CSVDataModule


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
        train_filename="train.csv",
        test_filename="test.csv",
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
    ],
)
def test_torchvision_data_module(tmpdir, dataset_name, num_tr, num_te, x_shape):
    """Test loading of torchvision data."""
    val_size = 0.35
    data_module = TorchVisionDataModule(tmpdir, dataset_name=dataset_name, val_size=val_size)
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
    "dataset_name,chunk_id,num_tr,num_te",
    [
        ("CLEAR10", 0, 2986, 500),
        ("CLEAR100", 0, 9945, 4984),
    ],
)
def test_clear_data_module(tmpdir, dataset_name, chunk_id, num_tr, num_te):
    """Test loading of CLEAR data."""
    val_size = 0.2
    data_module = CLEARDataModule(
        tmpdir, dataset_name=dataset_name, chunk_id=chunk_id, val_size=val_size
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


@pytest.mark.slow
def test_tiny_imagenet_data_module(tmpdir):
    num_tr = 100000
    num_te = 10000
    val_size = 0.2
    data_module = TinyImageNetDataModule(tmpdir, val_size=val_size)
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
    assert train_data[0][0].size() == test_data[0][0].size() == (3, 64, 64)


@pytest.mark.parametrize(
    "dataset_name,input_column,target_column",
    [
        ("rotten_tomatoes", "text", "label"),
    ],
)
@pytest.mark.parametrize(
    "tokenizer", [transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")]
)
@pytest.mark.parametrize(
    "tokenizer_kwargs",
    [
        None,
        dict(padding="longest", max_length=512, truncation=True),
    ],
)
def test_hugging_face_data_module(
    tmpdir, dataset_name, input_column, target_column, tokenizer, tokenizer_kwargs
):
    data_module = HuggingFaceTextDataModule(
        data_path=tmpdir,
        dataset_name=dataset_name,
        input_column=input_column,
        target_column=target_column,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        val_size=0.2,
    )
    data_module.prepare_data()
    data_module.setup()
    train_data = data_module.train_data()
    val_data = data_module.val_data()
    test_data = data_module.test_data()
    assert isinstance(train_data, Dataset)
    assert isinstance(val_data, Dataset)
    assert isinstance(test_data, Dataset)
    for dataset in [train_data, val_data, test_data]:
        inputs, targets = dataset[0]
        assert isinstance(targets, torch.Tensor)
        assert isinstance(inputs, dict)
        assert set(inputs.keys()) == set(["input_ids", "attention_mask"])
        assert isinstance(inputs["input_ids"], torch.Tensor)
        assert isinstance(inputs["attention_mask"], torch.Tensor)


@pytest.mark.parametrize("column", ("input", "target"), ids=("input", "target"))
def test_hugging_face_exception_raised_with_wrong_column(tmpdir, column):
    input_column = "text"
    target_column = "label"
    if column == "input":
        input_column = "WRONG_COLUMN"
    elif column == "target":
        target_column = "WRONG_COLUMN"
    data_module = HuggingFaceTextDataModule(
        data_path=tmpdir,
        dataset_name="rotten_tomatoes",
        input_column=input_column,
        target_column=target_column,
        tokenizer=None,
        val_size=0.2,
    )
    if column == "input":
        with pytest.raises(
            ValueError,
            match="Input column 'WRONG_COLUMN' does not exist in rotten_tomatoes. "
            "Available columns: \\['text', 'label'\\].",
        ):
            data_module.prepare_data()
    elif column == "target":
        with pytest.raises(
            ValueError,
            match="Target column 'WRONG_COLUMN' does not exist in rotten_tomatoes. "
            "Available columns: \\['text', 'label'\\].",
        ):
            data_module.prepare_data()
