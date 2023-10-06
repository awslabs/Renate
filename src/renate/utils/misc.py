# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Dict, Optional, Set, Tuple, Union

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


class AdditionalTrainingMetrics:
    def __init__(self) -> None:
        """We gather memory stats, Time stats, number of trainable params, total params"""
        self._train_start_time = time.time()

    def __call__(self, model: torch.nn.Module) -> Dict[str, Union[float, int]]:
        curr_running_time = time.time() - self._train_start_time
        # maximum amount of memory used in training. This might
        # not be the best
        peak_memory_usage = (
            torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            if torch.cuda.is_available()
            else 0
        )
        trainable_params, total_params = self.parameters_count(model)

        return dict(
            curr_running_time=curr_running_time,
            peak_memory_usage=peak_memory_usage,
            trainable_params=trainable_params,
            total_params=total_params,
        )

    def parameters_count(self, model: torch.nn.Module) -> Tuple[int, int]:
        trainable_params, total_params = 0, 0
        for param in model.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, total_params
