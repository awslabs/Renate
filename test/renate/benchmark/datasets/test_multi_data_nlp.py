# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import transformers as transformers

from renate.benchmark.datasets.nlp_datasets import MultiTextDataModule


@pytest.mark.skip(reason="This test create problems with the syne-tune redirect test")
def test_multi_data_nlp_small(tmpdir):
    TRAIN_SIZE = 100
    TEST_SIZE = 100

    data = MultiTextDataModule(
        tmpdir,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        data_id="ag_news",
        tokenizer=transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
        seed=42,
    )

    data.prepare_data()
    data.setup()

    assert len(data.train_data()) == TRAIN_SIZE
    assert len(data.test_data()) == TEST_SIZE

    first_input_agnews = data.train_data()[0][0]["input_ids"]

    data.data_id = "dbpedia_14"
    data.setup()

    tr_data_dbpedia = data.train_data()
    te_data_dbpedia = data.test_data()
    assert len(tr_data_dbpedia) == TRAIN_SIZE
    assert len(te_data_dbpedia) == TEST_SIZE

    first_input_dbpedia = data.train_data()[0][0]["input_ids"]

    assert not torch.all(torch.eq(first_input_dbpedia, first_input_agnews))


@pytest.mark.skip(reason="This test requires downloading and processing four datasets.")
def test_multi_data_nlp_full(tmpdir):
    TRAIN_SIZE = 115000
    TEST_SIZE = 7600

    for d in MultiTextDataModule.domains:
        data = MultiTextDataModule(
            tmpdir,
            train_size=TRAIN_SIZE,
            test_size=TEST_SIZE,
            data_id=d,
            tokenizer=transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
        )

        data.prepare_data()
        data.setup()

        tr_data = data.train_data()
        te_data = data.test_data()
        assert len(tr_data) == TRAIN_SIZE
        assert len(te_data) == TEST_SIZE
