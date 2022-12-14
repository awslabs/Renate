# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class ContinualNorm(_BatchNorm):
    """Continual Normalization as a replacement for Batch Normalization.

    Pham, Quang, Chenghao Liu, and Steven Hoi.
    "Continual normalization: Rethinking batch normalization for online continual learning."
    International Conference on Learning Representations (2022).

    It combines Group Normalization with respect to a user-defined `num_groups` parameter, the
    number of groups in Group Normalization, followed by Batch Normalization.

    Args:
        num_features: The number of input features in the channel dimension.
        eps:  A value added to the denominator for numerical stability.
        momentum: the value used for the running_mean and running_var computation.
                  Can be set to ``None`` for cumulative moving average.
        affine: Whether learnable affine parameters are going to be used in Batch Normalization.
        track_running_stats:  Whether running stats are tracked in Batch Normalization.
        device: What device to store the parameters.
        dtype: The data type of the learnable parameters.
        num_groups: The number of groups in the Group Normalization.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        num_groups: int = 32,
    ):
        super(ContinualNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
        self._num_groups = num_groups

    def forward(self, input: torch.Tensor):
        return F.batch_norm(
            F.group_norm(input, self._num_groups, None, None, self.eps),
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )

    def extra_repr(self):
        return super().extra_repr() + f", num_groups={self._num_groups}"
