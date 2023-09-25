# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Set, Union

import torch

from renate.utils.pytorch import complementary_indices


def int_or_str(x: str) -> Union[str, int]:
    """Function to cast to int or str.

    This is used to tackle precision which can be int (16, 32) or str (bf16)
    """
    try:
        return int(x)
    except ValueError:
        return x


def maybe_populate_mask_and_ignore_logits(
    use_masking: bool,
    class_mask: Optional[torch.Tensor],
    classes_in_current_task: Optional[Set[int]],
    logits: torch.Tensor,
):
    """Snippet to compute which logits to ignore after computing the class mask if required."""
    if use_masking:
        if class_mask is None:
            # Now is the time to repopulate the class_mask
            class_mask = torch.tensor(
                complementary_indices(logits.size(1), classes_in_current_task),
                device=logits.device,
                dtype=torch.long,
            )
            # fill the logits with -inf
        logits.index_fill_(1, class_mask.to(logits.device), -float("inf"))

    return logits, class_mask
