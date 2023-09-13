# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from renate.benchmark.models.spromptmodel import PromptPool, SharedMultipleLinear


@pytest.mark.parametrize("share_parameters", [True, False])
def test_shared_multiple_classifier(share_parameters):
    model = SharedMultipleLinear(5, 3, share_parameters=share_parameters, num_updates=10)
    num_elems = sum(x.numel() for x in model.parameters())
    assert num_elems == [10 * 5 * 3 + 10 * 3, 5 * 3 + 3][share_parameters]

    model.increment_task()
    num_elems = sum(x.numel() for x in model.parameters())
    assert num_elems == [(10 + 1) * 5 * 3 + (10 + 1) * 3, 5 * 3 + 3][share_parameters]


def test_prompt_pool():
    prompt_size = 7
    embedding_size = 12
    curr_update_id = 3
    pool = PromptPool(
        prompt_size=prompt_size, embedding_size=embedding_size, current_update_id=curr_update_id
    )

    for i in range(curr_update_id):
        assert pool(i).shape == (prompt_size, embedding_size)
        assert pool.get_params(i)[0].shape == (prompt_size, embedding_size)

    pool.increment_task()
    assert len(pool._pool) == curr_update_id + 1
