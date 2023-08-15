# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.defaults import TASK_ID


def test_renate_vision_transformer_init():
    pytest.helpers.get_renate_module_vision_transformer(sub_class="visiontransformercifar")
    pytest.helpers.get_renate_module_vision_transformer(sub_class="visiontransformerb16")
    pytest.helpers.get_renate_module_vision_transformer(sub_class="visiontransformerb32")
    pytest.helpers.get_renate_module_vision_transformer(sub_class="visiontransformerl16")
    pytest.helpers.get_renate_module_vision_transformer(sub_class="visiontransformerl32")
    pytest.helpers.get_renate_module_vision_transformer(sub_class="visiontransformerh14")


@pytest.mark.slow
@pytest.mark.parametrize(
    "sub_class,input_dim",
    [
        ["visiontransformercifar", (3, 32, 32)],
        ["visiontransformerb16", (3, 224, 224)],
        ["visiontransformerb32", (3, 224, 224)],
        ["visiontransformerl16", (3, 224, 224)],
        ["visiontransformerl32", (3, 224, 224)],
        ["visiontransformerh14", (3, 224, 224)],
    ],
)
def test_renate_vision_transformer_fwd(sub_class, input_dim):
    vit = pytest.helpers.get_renate_module_vision_transformer(sub_class=sub_class)

    x = torch.rand(5, *input_dim)
    y_hat = vit(x)

    assert y_hat.shape[0] == 5
    assert y_hat.shape[1] == 10


# The following numbers have been computed by
# for m in [VisionTransformerB16, VisionTransformerB32, VisionTransformerCIFAR,
#           VisionTransformerH14, VisionTransformerL16, VisionTransformerL32]:
#     print(len(list(m()._backbone.parameters())))
@pytest.mark.parametrize(
    "sub_class, expected_num_params",
    [
        ["visiontransformercifar", 54],
        ["visiontransformerb16", 198],
        ["visiontransformerb32", 198],
        ["visiontransformerl16", 390],
        ["visiontransformerl32", 390],
        ["visiontransformerh14", 518],
    ],
)
def test_renate_vision_transformer_get_params(sub_class, expected_num_params):
    vit = pytest.helpers.get_renate_module_vision_transformer(sub_class=sub_class)
    vit.add_task_params(TASK_ID)

    first_task_params = vit.get_params(TASK_ID)

    vit.add_task_params("second_task")

    second_task_params = vit.get_params("second_task")
    assert len(first_task_params) == len(second_task_params)
    assert len(first_task_params) == expected_num_params + 2
    assert len(second_task_params) == expected_num_params + 2
    assert len(list(vit.parameters())) == expected_num_params + 4
    for i in range(len(first_task_params) - 2):
        # -2 because the last two parameters are weight and bias of a task specific linear layer
        assert torch.equal(first_task_params[i], second_task_params[i])

    for i in range(len(first_task_params) - 2, len(first_task_params)):
        # -2 because the last two parameters are weight and bias of a task specific linear layer
        assert not torch.equal(first_task_params[i], second_task_params[i])
