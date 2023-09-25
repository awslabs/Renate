# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.benchmark.models.transformer import HuggingFaceSequenceClassificationTransformer


@pytest.mark.parametrize("model_name", ["distilbert-base-uncased", "bert-base-uncased"])
def test_init(model_name):
    HuggingFaceSequenceClassificationTransformer(
        pretrained_model_name_or_path=model_name, num_outputs=10
    )


@pytest.mark.parametrize(
    "model_name,input_dim",
    [
        ["distilbert-base-uncased", (128,)],
        ["bert-base-uncased", (256,)],
    ],
)
def test_text_transformer_fwd(model_name, input_dim):
    transformer = HuggingFaceSequenceClassificationTransformer(
        pretrained_model_name_or_path=model_name
    )

    x = {"input_ids": torch.randint(0, 30000, (5, *input_dim))}
    y_hat = transformer(x)

    assert y_hat.shape[0] == 5
    assert y_hat.shape[1] == 10
