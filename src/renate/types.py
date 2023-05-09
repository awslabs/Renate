# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Tuple, Union

import torch

NestedTensors = Union[torch.Tensor, Tuple["NestedTensors"], Dict[str, "NestedTensors"]]
