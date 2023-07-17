# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoTokenizer

from renate.utils.hf_utils import BatchEncoding, DataCollatorWithPaddingForWildTime


@pytest.mark.parametrize(
    "features, shape, error",
    [
        (
            [
                ({"input_ids": [0, 1, 2]}, 0),
                ({"input_ids": [0, 1, 2, 3, 4, 5]}, 1),
            ],
            torch.Size([2, 6]),
            False,
        ),
        (
            [
                (({"input_ids": [1, 2, 4]}, 0), {}),
                (({"input_ids": [4, 5, 6, 7, 8, 9]}, 1), {}),
            ],
            torch.Size([2, 6]),
            False,
        ),
        (
            [
                (({"input_ids": [1, 2, 4]}, 0), {"logits": [0.1, 0.2, 0.3]}),
                (({"input_ids": [4, 5, 6, 7, 8, 9]}, 1), {"logits": [0.3, 0.4, 0.5]}),
            ],
            torch.Size([2, 6]),
            False,
        ),
        (
            [[1, ({"input_ids": [0, 1, 2]}, 0)], [2, ({"input_ids": [0, 1, 2, 3, 4, 5]}, 1)]],
            torch.Size([2]),
            False,
        ),
        (
            [
                ({"input_ids": [1, 2, 4]}, 0, {}),
            ],
            torch.Size([1, 3]),
            True,
        ),
    ],
)
@pytest.mark.parametrize("tokenizer", [AutoTokenizer.from_pretrained("bert-base-cased")])
def test_serialize_wildtime_collator(features, shape, error, tokenizer):
    if not error:
        collator = DataCollatorWithPaddingForWildTime(tokenizer)
        out = collator(features)[0]
        print(collator(features))
        if isinstance(out, (dict, BatchEncoding)):
            assert out["input_ids"].shape == shape
        elif isinstance(out, torch.Tensor):
            assert out.shape == shape
        else:
            assert out[0]["input_ids"].shape == shape
    else:
        collator = DataCollatorWithPaddingForWildTime(tokenizer)
        with pytest.raises(Exception):
            out = collator(features)[0]
