# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from unittest import mock

import pytest

from renate.utils.syne_tune import redirect_to_tmp


@pytest.mark.parametrize(
    "uri, expected_uri, raises",
    [
        ("/opt/ml/checkpoints/0/", "/tmp/checkpoints/0/", False),
        ("/home/checkpoints/0/", "/home/checkpoints/0/", True),
    ],
    ids=["redirect-path-to-tmp-dir", "wrong-checkpoint-path-raise-exception"],
)
@mock.patch.dict(os.environ, {"SM_MODEL_DIR": ""})
def test_redirect_to_tmp_on_sagemaker(uri, expected_uri, raises):
    if raises:
        with pytest.raises(AssertionError):
            redirect_to_tmp(uri)
    else:
        assert redirect_to_tmp(uri) == expected_uri


@pytest.mark.parametrize(
    "uri, expected_uri",
    [
        ("/opt/ml/checkpoints/0/", "/opt/ml/checkpoints/0/"),
        ("/home/checkpoints/0/", "/home/checkpoints/0/"),
    ],
)
def test_redirect_to_tmp(uri, expected_uri):
    """This function is not supposed to change anything if not running on SageMaker."""
    assert redirect_to_tmp(uri) == expected_uri
