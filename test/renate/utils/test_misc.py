# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from renate.utils.misc import int_or_str


@pytest.mark.parametrize(
    "data_type,target",
    [
        ["16", 16],
        ["32", 32],
        ["bfloat", "bfloat"],
        ["bfloat16", "bfloat16"],
        ["notdata", "notdata"],
    ],
)
def test_int_or_str(data_type, target):
    assert int_or_str(data_type) == target
