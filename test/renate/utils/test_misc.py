# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.utils.misc import int_or_str, maybe_populate_mask_and_ignore_logits


@pytest.mark.parametrize(
    "data_type,target",
    [
        ["16", 16],
        ["32", 32],
        ["bfloat", "bfloat"],
        ["bfloat16", "bfloat16"],
        ["notdata", "notdata"],
    ],
)
def test_int_or_str(data_type, target):
    assert int_or_str(data_type) == target


@pytest.mark.parametrize(
    "use_masking, class_mask, classes_in_task, logits, correct_output",
    [
        [False, None, None, 0.5 * torch.ones((1, 5)), (0.5 * torch.ones((1, 5)), None)],
        [
            True,
            None,
            {0, 1, 2},
            0.5 * torch.ones((1, 5)),
            (
                torch.tensor([[0.5000, 0.5000, 0.5000, -float("inf"), -float("inf")]]),
                torch.tensor([3, 4]),
            ),
        ],
        [
            True,
            torch.tensor([3, 4]),
            {0, 1, 2},
            0.5 * torch.ones((1, 5)),
            (
                torch.tensor([[0.5000, 0.5000, 0.5000, -float("inf"), -float("inf")]]),
                torch.tensor([3, 4]),
            ),
        ],
    ],
)
def test_possibly_populate_mask_and_ignore_logits(
    use_masking, class_mask, classes_in_task, logits, correct_output
):
    out_logits, out_cm = maybe_populate_mask_and_ignore_logits(
        use_masking=use_masking,
        class_mask=class_mask,
        classes_in_current_task=classes_in_task,
        logits=logits,
    )

    # There are a few cases to test. Besides the obvious ones, we also check if the function is a
    # no-op for some parameters. For eg: when a class_mask is provided, it should be returned as is
    # All operations on logits should be inplace. Both are accomplished through comparing
    # data_ptr().
    assert torch.equal(out_logits, correct_output[0])
    if out_cm is None:
        assert out_cm == correct_output[1]
    else:
        assert torch.equal(out_cm, correct_output[1])

    assert out_logits.data_ptr() == logits.data_ptr()
    if class_mask is not None:
        assert class_mask.data_ptr() == out_cm.data_ptr()
