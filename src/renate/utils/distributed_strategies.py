# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import warnings
from typing import Optional

from pytorch_lightning.strategies import Strategy, StrategyRegistry

_SUPPORTED_STRATEGIES = [
    "ddp_find_unused_parameters_false",
    "ddp",
    "deepspeed",
    "deepspeed_stage_1",
    "deepspeed_stage_2",
    "deepspeed_stage_2_offload",
    "deepspeed_stage_3",
    "deepspeed_stage_3_offload",
    "deepspeed_stage_3_offload_nvme",
]
_UNSUPPORTED_STRATEGIES = [
    x for x in StrategyRegistry.available_strategies() if x not in _SUPPORTED_STRATEGIES
]


def create_strategy(devices: int = 1, strategy_name: Optional["str"] = None) -> Strategy:
    """Function returns a strategy object based on the number of devices queried
    and name of strategy"""

    devices = devices or 1
    if strategy_name in _UNSUPPORTED_STRATEGIES:
        raise ValueError(
            f"Current strategy: {strategy_name} is unsupported. Choose deepspeed variants or ddp."
        )
    if devices < 0:
        raise ValueError("Number of devices has to be at least 0.")

    elif devices == 1:
        # If one GPU, use standard training. Enabled by passing strategy=None
        # to pl.Trainer
        if strategy_name is not None:
            warnings.warn(f"With devices=1, strategy is ignored. But got {strategy_name}.")

        return None
    elif strategy_name in ["none", "None", None]:
        # Nothing is specified and devices > 1. Fall back to DDP
        return StrategyRegistry.get("ddp")

    elif "deepspeed" in strategy_name:
        strategy = StrategyRegistry.get(strategy_name)

        # TODO: This should be changed to instantiating Deepspeed and settting it in
        # the constructor. This works for nowbecause forcing PyTorch optimizer flag isn't used
        # anywhere by Deepspeed.
        strategy.config["zero_force_ds_cpu_optimizer"] = False
        return strategy

    else:
        # Something else happened. Fall back to whatever is happening.
        return StrategyRegistry.get(strategy_name)
