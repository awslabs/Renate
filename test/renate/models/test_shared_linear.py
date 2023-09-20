# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from renate.models.layers.shared_linear import SharedMultipleLinear


@pytest.mark.parametrize("share_parameters", [True, False])
def test_shared_multiple_classifier(share_parameters):
    model = SharedMultipleLinear(5, 3, share_parameters=share_parameters, num_updates=10)
    num_elems = sum(x.numel() for x in model.parameters())
    assert num_elems == [10 * 5 * 3 + 10 * 3, 5 * 3 + 3][share_parameters]

    model.increment_task()
    num_elems = sum(x.numel() for x in model.parameters())
    assert num_elems == [(10 + 1) * 5 * 3 + (10 + 1) * 3, 5 * 3 + 3][share_parameters]
