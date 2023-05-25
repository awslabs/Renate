# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Type, Union

import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as _ResNet

from renate.benchmark.models.base import RenateBenchmarkingModule
from renate.models.prediction_strategies import PredictionStrategy


class ResNet(RenateBenchmarkingModule):
    """ResNet model base class.

    TODO: Fix citation
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun:
    Deep Residual Learning for Image Recognition. CVPR 2016: 770-778

    Args:
        block: The type of the block to use as the core building block.
        layers: The number of blocks in the respective parts of ResNet.
        num_outputs: The number of output units.
        zero_init_residual: Whether to set the initial weights of the residual blocks to zero
            through initializing the Batch Normalization parameters at the end of the block to zero.
        groups: The number of groups to be used for the group convolution.
        width_per_group: The width of the group convolution.
        replace_stride_with_dilation: Whether to replace the stride with a dilation to save memory.
        norm_layer: What kind of normalization layer to use, following convolutions.
        cifar_stem: Whether to use a stem for CIFAR-sized images.
        gray_scale: Whether input images are gray-scale images, i.e. only 1 color channel.
        prediction_strategy: Continual learning strategies may alter the prediction at train or test
            time.
        add_icarl_class_means: If ``True``, additional parameters used only by the
            ``ICaRLModelUpdater`` are added. Only required when using that updater.
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
        gray_scale: bool = False,
        prediction_strategy: Optional[PredictionStrategy] = None,
        add_icarl_class_means: bool = True,
    ) -> None:
        model = _ResNet(
            block=block,
            layers=layers,
            num_classes=num_outputs,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )
        super().__init__(
            embedding_size=model.fc.in_features,
            num_outputs=num_outputs,
            constructor_arguments={
                "block": block,
                "layers": layers,
                "zero_init_residual": zero_init_residual,
                "groups": groups,
                "width_per_group": width_per_group,
                "replace_stride_with_dilation": replace_stride_with_dilation,
                "norm_layer": norm_layer,
                "cifar_stem": cifar_stem,
                "gray_scale": gray_scale,
            },
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )
        self._backbone = model
        if cifar_stem:
            self._backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self._backbone.maxpool = nn.Identity()
        if gray_scale:
            self._backbone.conv1 = nn.Conv2d(
                1,
                self._backbone.conv1.out_channels,
                kernel_size=self._backbone.conv1.kernel_size,
                stride=self._backbone.conv1.stride,
                padding=self._backbone.conv1.padding,
                bias=self._backbone.conv1.bias is not None,
            )
        self._backbone.fc = nn.Identity()

        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


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
