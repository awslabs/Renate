# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional, Tuple, Union

import torch
from transformers import ViTConfig, ViTModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from renate import defaults
from renate.benchmark.models.base import RenateBenchmarkingModule
from renate.models.prediction_strategies import PredictionStrategy


class FeatureExtractorViTModel(ViTModel):
    """This class directly outputs [CLS] features if cls_feat is True else returns per patch
    embeddings
    """

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cls_feat: bool = True,
        task_id: str = defaults.TASK_ID,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """Output has patch embeddings and the pooled output. We extract pooled CLS out by
        taking the second element.
        """
        out_to_filter = super().forward(
            pixel_values,
            head_mask,
            output_attentions,
            output_hidden_states,
            interpolate_pos_encoding,
            return_dict,
        )
        if isinstance(out_to_filter, BaseModelOutputWithPooling):
            return out_to_filter.pooler_output if cls_feat else out_to_filter.last_hidden_state
        return out_to_filter[0][:, 0] if cls_feat else out_to_filter[0]


class VisionTransformer(RenateBenchmarkingModule):
    """Vision Transformer base model.

    TODO: Fix citation
    Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition
    at scale."
    arXiv preprint arXiv:2010.11929 (2020).

    Args:
        pretrained_model_name_or_path: A string that denotes which pretrained model from the HF hub
            to use. If provided, it overrides other arguments about architecture.
        image_size: Size of the input image.
        patch_size: Size of the patches.
        num_layers: Number of Encoder layers.
        num_heads: Number of Attention heads.
        hidden_dim: Size of the Encoder's hidden state.
        mlp_dim: Size of the intermediate Multi-layer Perceptron in the Encoder.
        dropout: Dropout probability.
        attention_dropout: Dropout probability for the attention in the Multi-head Attention layer.
        num_outputs: Size of the output.
        prediction_strategy: Continual learning strategies may alter the prediction at train or test
            time.
        add_icarl_class_means: If ``True``, additional parameters used only by the
            ``ICaRLModelUpdater`` are added. Only required when using that updater.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        image_size: int = 32,
        patch_size: int = 4,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        num_outputs: int = 10,
        prediction_strategy: Optional[PredictionStrategy] = None,
        add_icarl_class_means: bool = True,
    ) -> None:
        if pretrained_model_name_or_path:
            model = FeatureExtractorViTModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                return_dict=False,
                add_pooling_layer=False,
            )

            constructor_args = dict()
        else:
            model_config = ViTConfig(
                hidden_size=hidden_dim,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                intermediate_size=mlp_dim,
                hidden_act="gelu",
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=attention_dropout,
                layer_norm_eps=1e-6,
                image_size=image_size,
                patch_size=patch_size,
                num_channels=3,
                qkv_bias=True,
                return_dict=False,
            )

            model = FeatureExtractorViTModel(config=model_config, add_pooling_layer=False)
            constructor_args = {
                "image_size": image_size,
                "patch_size": patch_size,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "hidden_dim": hidden_dim,
                "mlp_dim": mlp_dim,
                "dropout": dropout,
                "attention_dropout": attention_dropout,
            }

        super().__init__(
            embedding_size=model.config.hidden_size,
            num_outputs=num_outputs,
            constructor_arguments=constructor_args,
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )
        self._backbone = model

    def get_features(self, *args, **kwargs):
        # This is need as a shortcut to not call the base class's forward and directly call the
        # backbone's forward. Used only in L2P.
        return self._backbone(*args, **kwargs)


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
            pretrained_model_name_or_path="google/vit-base-patch16-224",
            **kwargs,
        )


class VisionTransformerB32(VisionTransformer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            pretrained_model_name_or_path="google/vit-base-patch32-224-in21k",
            **kwargs,
        )


class VisionTransformerL16(VisionTransformer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            pretrained_model_name_or_path="google/vit-large-patch16-224-in21k",
            **kwargs,
        )


class VisionTransformerL32(VisionTransformer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            pretrained_model_name_or_path="google/vit-large-patch32-224-in21k",
            **kwargs,
        )


class VisionTransformerH14(VisionTransformer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            pretrained_model_name_or_path="google/vit-huge-patch14-224-in21k",
            **kwargs,
        )
