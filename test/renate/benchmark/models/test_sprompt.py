# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from renate.benchmark.models.spromptmodel import PromptPool


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
