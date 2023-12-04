# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn


class SharedMultipleLinear(nn.ModuleDict):
    """This implements a linear classification layer for multiple tasks (updates).
    This linear layer can be shared across all tasks or can have a separate layer per task.
    This follows the `_task_params` in the `RenateBenchmarkingModule` that is a `nn.ModuleDict`
    that holds a classifier per task (as in TIL).

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        share_parameters: Flag whether to share parameters or use individual linears per task.
            The interface remains identical, and the underlying linear layer is shared (or not).
        num_updates: Number of updates that have happened/is happening.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        share_parameters: bool = True,
        num_updates: int = 0,
    ) -> None:
        self._share_parameters = share_parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        super().__init__()
        for _ in range(num_updates):
            self.increment_task()

    def increment_task(self) -> None:
        currlen = len(self)
        if self._share_parameters:
            self[f"{currlen}"] = (
                self[list(self.keys())[0]]
                if currlen > 0
                else nn.Linear(in_features=self.in_features, out_features=self.out_features)
            )
        else:
            self[f"{currlen}"] = nn.Linear(
                in_features=self.in_features, out_features=self.out_features
            )
