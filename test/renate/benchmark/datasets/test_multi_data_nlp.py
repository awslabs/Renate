# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import transformers as transformers

from renate.benchmark.datasets.nlp_datasets import MultiTextDataModule


def test_multi_data_nlp_small():
    TRAIN_SIZE = 100
    TEST_SIZE = 100

    data = MultiTextDataModule(
        "./remove_folder/",
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        domain="ag_news",
        tokenizer=transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
    )

    data.prepare_data()
    data.setup()

    assert len(data.train_data()) == TRAIN_SIZE
    assert len(data.test_data()) == TEST_SIZE


@pytest.mark.skip(reason="This test requires downloading and processing five datasets.")
def test_multi_data_nlp_full():
    TRAIN_SIZE = 115000
    TEST_SIZE = 7600

    for d in [
        "ag_news",
        "yelp_review_full",
        "amazon_reviews_multi",
        "dbpedia_14",
        "yahoo_answers_topics",
    ]:
        data = MultiTextDataModule(
            "./remove_folder/",
            train_size=TRAIN_SIZE,
            test_size=TEST_SIZE,
            domain=d,
            tokenizer=transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
        )

        data.prepare_data()
        data.setup()

        assert len(data.train_data()) == TRAIN_SIZE
        assert len(data.test_data()) == TEST_SIZE
