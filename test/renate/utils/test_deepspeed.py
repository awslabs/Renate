# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from renate.utils.deepspeed import convert_to_tensor, recover_object_from_tensor


@pytest.mark.parametrize(
    "obj",
    [
        {
            "constructor_arguments": {
                "a": 1,
                "b": "2",
                "nested_dict": {
                    "k1": "v1",
                    "k2": "v2",
                },
            },
            "tasks_params_ids": "task_param_id_1",
            "misc_args": tuple(range(10)),
        }
    ],
)
def test_serialize_random_objects(obj):
    assert recover_object_from_tensor(convert_to_tensor(obj)) == obj
