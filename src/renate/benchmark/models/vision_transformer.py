# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import ConvStemConfig, WeightsEnum
from torchvision.models.vision_transformer import VisionTransformer as _VisionTransformer

from renate.defaults import TASK_ID
from renate.models import RenateModule


class VisionTransformer(RenateModule):
    """Vision Transformer base model.

    TODO: Fix citation
    Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale."
    arXiv preprint arXiv:2010.11929 (2020).

    Args:
        image_size: Size of the input image.
        patch_size: Size of the patches.
        num_layers: Number of Encoder layers.
        num_heads: Number of Attention heads.
        hidden_dim: Size of the Encoder's hidden state.
        mlp_dim: Size of the intermediate Multi-layer Perceptron in the Encoder.
        dropout: Dropout probability.
        attention_dropout: Dropout probability for the attention in the Multi-head Attention layer.
        num_outputs: Size of the output.
        representation_size: If specified, the model will return a linear projection of the last hidden state.
        norm_layer: Normalization layer.
        conv_stem_configs: List of ConvStemConfig. Each ConvStemConfig corresponds to a convolutional stem.
        loss: Loss function.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        num_outputs: int = 10,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        weights: Optional[WeightsEnum] = None,
        loss: nn.Module = nn.CrossEntropyLoss(),
    ) -> None:
        RenateModule.__init__(
            self,
            constructor_arguments={
                "image_size": image_size,
                "patch_size": patch_size,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "hidden_dim": hidden_dim,
                "mlp_dim": mlp_dim,
                "dropout": dropout,
                "attention_dropout": attention_dropout,
                "num_outputs": num_outputs,
                "representation_size": representation_size,
                "norm_layer": norm_layer,
                "conv_stem_configs": conv_stem_configs,
            },
            loss_fn=loss,
        )
        self._model = _VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_outputs,
            representation_size=representation_size,
            norm_layer=norm_layer,
            conv_stem_configs=conv_stem_configs,
        )
        if weights:
            self._model.load_state_dict(weights.get_state_dict())

        self._last_hidden_size = self._model.heads.head.in_features
        self._num_outputs = num_outputs
        self._model.heads.head = nn.Identity()
        self._tasks_params: nn.ModuleDict = nn.ModuleDict()
        self.add_task_params(TASK_ID)

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


class VisionTransformerCIFAR(VisionTransformer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            image_size=32,
            patch_size=4,
            num_layers=3,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,
            **kwargs,
        )


class VisionTransformerB16(VisionTransformer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            **kwargs,
        )


class VisionTransformerB32(VisionTransformer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            image_size=224,
            patch_size=32,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            **kwargs,
        )


class VisionTransformerL16(VisionTransformer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            image_size=224,
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            **kwargs,
        )


class VisionTransformerL32(VisionTransformer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            image_size=224,
            patch_size=32,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            **kwargs,
        )


class VisionTransformerH14(VisionTransformer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            image_size=224,
            patch_size=14,
            num_layers=32,
            num_heads=16,
            hidden_dim=1280,
            mlp_dim=5120,
            **kwargs,
        )
