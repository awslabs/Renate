# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.benchmark.models import ResNet18
from renate.defaults import TASK_ID


def test_renate_resnet_init():
    pytest.helpers.get_renate_module_resnet(sub_class="resnet18cifar")
    pytest.helpers.get_renate_module_resnet(sub_class="resnet34cifar")
    pytest.helpers.get_renate_module_resnet(sub_class="resnet50cifar")
    pytest.helpers.get_renate_module_resnet(sub_class="resnet18")
    pytest.helpers.get_renate_module_resnet(sub_class="resnet34")
    pytest.helpers.get_renate_module_resnet(sub_class="resnet50")


@pytest.mark.parametrize(
    "sub_class,input_dim",
    [
        ["resnet18cifar", (3, 32, 32)],
        ["resnet34cifar", (3, 32, 32)],
        ["resnet50cifar", (3, 32, 32)],
        ["resnet18", (3, 224, 224)],
        ["resnet34", (3, 224, 224)],
        ["resnet50", (3, 224, 224)],
    ],
)
def test_renate_resnet_fwd(sub_class, input_dim):
    resnet = pytest.helpers.get_renate_module_resnet(sub_class=sub_class)

    x = torch.rand(5, *input_dim)
    y_hat = resnet(x)

    assert y_hat.shape[0] == 5
    assert y_hat.shape[1] == 10


@pytest.mark.parametrize(
    "sub_class, expected_num_params",
    [
        ["resnet18cifar", 60],
        ["resnet34cifar", 108],
        ["resnet50cifar", 159],
        ["resnet18", 60],
        ["resnet34", 108],
        ["resnet50", 159],
    ],
)
def test_renate_resnet_get_params(sub_class, expected_num_params):
    resnet = pytest.helpers.get_renate_module_resnet(sub_class=sub_class)
    resnet.add_task_params(TASK_ID)

    first_task_params = resnet.get_params(TASK_ID)

    resnet.add_task_params("second_task")

    second_task_params = resnet.get_params("second_task")
    assert len(first_task_params) == len(second_task_params)
    assert len(first_task_params) == expected_num_params + 2
    assert len(second_task_params) == expected_num_params + 2
    assert len(list(resnet.parameters())) == expected_num_params + 4
    for i in range(len(first_task_params) - 2):
        # -2 because the last two parameters are weight and bias of a task specific linear layer
        assert torch.equal(first_task_params[i], second_task_params[i])

    for i in range(len(first_task_params) - 2, len(first_task_params)):
        # -2 because the last two parameters are weight and bias of a task specific linear layer
        assert not torch.equal(first_task_params[i], second_task_params[i])


@pytest.mark.parametrize("gray_scale", (True, False), ids=("gray scale", "rgb"))
def test_renate_resnet_gray_scale_parameter(gray_scale):
    """Tests if gray_scale parameter correctly controls number of input channels."""
    expected_in_channels = 1 if gray_scale else 3
    assert ResNet18(gray_scale=gray_scale).get_backbone().conv1.in_channels == expected_in_channels
