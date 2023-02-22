# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Tuple, Union

import torch

Inputs = Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]
