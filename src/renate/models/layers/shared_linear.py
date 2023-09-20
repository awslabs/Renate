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

        if share_parameters:
            # we only have a single linear.
            layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
            all_layers = {f"{id}": layer for id in range(num_updates)}
        else:
            all_layers = {
                f"{id}": nn.Linear(in_features=in_features, out_features=out_features)
                for id in range(num_updates)
            }
        super().__init__(all_layers)

    def increment_task(self) -> None:
        currlen = len(self)
        if self._share_parameters:
            self[f"{currlen}"] = self[list(self.keys())[0]]
        else:
            self[f"{currlen}"] = nn.Linear(
                in_features=self.in_features, out_features=self.out_features
            )
