# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as _ResNet

from renate.defaults import TASK_ID
from renate.models.renate_module import RenateModule


class ResNet(RenateModule):
    """ResNet model base class.

    TODO: Fix citation
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun:
    Deep Residual Learning for Image Recognition. CVPR 2016: 770-778

    Args:
        block: The type of the block to use as the core building block.
        layers: The number of blocks in the respective parts of ResNet.
        num_outputs: The number of output units.
        zero_init_residual: Whether to set the initial weights of the residual blocks
                            to zero through initializing the Batch Normalization parameters at the end of the block to zero.
        groups: The number of groups to be used for the group convolution.
        width_per_group: The width of the group convolution.
        replace_stride_with_dilation: Whether to replace the stride with a dilation to save memory.
        norm_layer: What kind of normalization layer to use, following convolutions.
        cifar_stem: Whether to use a stem for CIFAR-sized images.
        loss: Loss function to be used for training.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_outputs: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        cifar_stem: bool = True,
        loss: nn.Module = nn.CrossEntropyLoss(),
    ) -> None:
        RenateModule.__init__(
            self,
            constructor_arguments={
                "block": block,
                "layers": layers,
                "num_outputs": num_outputs,
                "zero_init_residual": zero_init_residual,
                "groups": groups,
                "width_per_group": width_per_group,
                "replace_stride_with_dilation": replace_stride_with_dilation,
                "norm_layer": norm_layer,
                "cifar_stem": cifar_stem,
            },
            loss_fn=loss,
        )
        self._model = _ResNet(
            block=block,
            layers=layers,
            num_classes=num_outputs,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )
        if cifar_stem:
            self._model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self._model.maxpool = nn.Identity()

        self._last_hidden_size = self._model.fc.in_features
        self._num_outputs = num_outputs
        self._model.fc = nn.Identity()
        self._tasks_params: nn.ModuleDict = nn.ModuleDict()
        self.add_task_params(TASK_ID)

        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, x: torch.Tensor, task_id: str = TASK_ID) -> torch.Tensor:
        """Performs a forward pass on the inputs and returns the predictions."""
        x = self._model(x)
        return self._tasks_params[task_id](x)

    def _add_task_params(self, task_id: str = TASK_ID) -> None:
        """Adds new parameters associated to a specific task to the model."""
        self._tasks_params[task_id] = nn.Linear(self._last_hidden_size, self._num_outputs)

    def get_params(self, task_id: str = TASK_ID) -> List[nn.Parameter]:
        """Returns the list of parameters for the core model and a specific `task_id`."""
        return list(self._model.parameters()) + list(self._tasks_params[task_id].parameters())


class ResNet18CIFAR(ResNet):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], cifar_stem=True, **kwargs) -> None:
        super().__init__(block=block, layers=layers, cifar_stem=cifar_stem, **kwargs)


class ResNet34CIFAR(ResNet):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3], cifar_stem=True, **kwargs) -> None:
        super().__init__(block=block, layers=layers, cifar_stem=cifar_stem, **kwargs)


class ResNet50CIFAR(ResNet):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], cifar_stem=True, **kwargs) -> None:
        super().__init__(block=block, layers=layers, cifar_stem=cifar_stem, **kwargs)


class ResNet18(ResNet):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], cifar_stem=False, **kwargs) -> None:
        super().__init__(block=block, layers=layers, cifar_stem=cifar_stem, **kwargs)


class ResNet34(ResNet):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3], cifar_stem=False, **kwargs) -> None:
        super().__init__(block=block, layers=layers, cifar_stem=cifar_stem, **kwargs)


class ResNet50(ResNet):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], cifar_stem=False, **kwargs) -> None:
        super().__init__(block=block, layers=layers, cifar_stem=cifar_stem, **kwargs)
