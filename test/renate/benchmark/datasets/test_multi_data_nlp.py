# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import transformers as transformers

from renate.benchmark.datasets.nlp_datasets import MultiTextDataModule


@pytest.mark.skip(reason="This test requires downloading five datasets.")
def test_multi_data_nlp():
    TRAIN_SIZE = 100

    data = MultiTextDataModule(
        "./remove_folder/",
        train_size=TRAIN_SIZE,
        tokenizer=transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
    )

    data.prepare_data()
    data.setup()

    assert len(data.train_data()) == 5
    assert len(data.test_data()) == 5

    for i in range(len(data.train_data())):
        assert len(data.train_data()[i]._dataset) == TRAIN_SIZE
