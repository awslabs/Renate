# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from renate.utils.distributed_strategies import (
    create_strategy,
    _SUPPORTED_STRATEGIES,
    _UNSUPPORTED_STRATEGIES,
)


@pytest.mark.parametrize("devices", [1, 2, 3, 10])
@pytest.mark.parametrize("strategy_name", _SUPPORTED_STRATEGIES)
def test_valid_strategy_creation(devices, strategy_name):
    assert isinstance(create_strategy(devices, strategy_name), (str, object, None))


@pytest.mark.parametrize("devices", [1, 2, 3, 10])
@pytest.mark.parametrize("strategy_name", _UNSUPPORTED_STRATEGIES)
def test_invalid_strategy_creation(devices, strategy_name):
    with pytest.raises(ValueError) as _:
        create_strategy(devices, strategy_name)
