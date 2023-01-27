# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from renate.benchmark.datasets.nlp_datasets import TorchTextDataModule


def test_nlp_dataset():
    dm = TorchTextDataModule("./data/aginews/", dataset_name="AG_NEWS", val_size=0.2)

    dm.prepare_data()
    dm.setup()

    assert len(dm.train_data()) > 0

    print(dm.train_data()[0])
