# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.benchmark.models.l2p import LearningToPromptTransformer, PromptPool, PromptedTransformer


def test_prompt_pool():
    D_emb = 2
    Lp = 3
    feat_dim = 6
    B = 4
    N = 2

    pool = PromptPool(
        embedding_dim=feat_dim, prompt_key_dim=D_emb, prompt_size=Lp, pool_selection_size=N
    )
    input = torch.rand(B, D_emb)
    out = pool(input)[0]

    assert out.shape == (B, Lp * N, feat_dim)


def test_prompted_vision_transformer():
    combined = LearningToPromptTransformer()
    inp = torch.rand(1, 3, 224, 224)
    assert combined(inp).shape == torch.Size((1, 10))


def test_prompted_text_transformer():
    model = LearningToPromptTransformer(pretrained_model_name_or_path="bert-base-uncased")
    inp = {"input_ids": torch.randint(0, 3000, (10, 128))}
    assert model(inp).shape == torch.Size((10, 10))


@pytest.mark.parametrize(
    "cls,arg,argval,error",
    [
        [PromptPool, "similarity_fn", "not_cosine", ValueError],
        [LearningToPromptTransformer, "prompt_embedding_features", "not_cls", AssertionError],
        [LearningToPromptTransformer, "patch_pooler", "not_cls", AssertionError],
    ],
)
def test_pool_vision_transformer_raises_errors(cls, arg, argval, error):
    with pytest.raises(error):
        cls(**{arg: argval})


@pytest.mark.parametrize(
    "backbone,num_trainable_params",
    [
        ["google/vit-base-patch16-224", 4],
        ["google/vit-base-patch32-224-in21k", 4],
        ["google/vit-large-patch32-224-in21k", 4],
        ["bert-base-uncased", 4],
        ["distilbert-base-uncased", 4],
    ],
)
def test_prompt_vision_transformer_trainable_parameters(backbone, num_trainable_params):
    # The result is always 4: prompt pool, prompt pool key, classifier wt, classifier bias.
    model = LearningToPromptTransformer(pretrained_model_name_or_path=backbone)
    n = sum(1 for x in model.parameters() if x.requires_grad)
    assert n == num_trainable_params


@pytest.mark.parametrize("backbone", ["google/vit-base-patch16-224", "bert-base-uncased"])
@pytest.mark.parametrize("prompt", [None, torch.rand(3, 10, 768)])
@pytest.mark.parametrize("cls_feat", [True, False])
def test_prompted_transformer(backbone, prompt, cls_feat):
    model = PromptedTransformer(
        pretrained_model_name_or_path=backbone,
        num_outputs=10,
        prediction_strategy=None,
        add_icarl_class_means=False,
    )

    B, P_len, _ = prompt.shape if prompt is not None else (5, 0, 0)
    if "vit" in backbone:
        # we are doing ViT.
        inputs = torch.rand(B, 3, 224, 224)
        expected_output_size = (B, 197 + P_len, 768) if not cls_feat else (B, 768)
    else:
        inputs = {"input_ids": torch.randint(0, 10000, (B, 128))}
        expected_output_size = (B, 768)

    out = model(inputs, prompt, cls_feat=cls_feat)
    assert out.shape == expected_output_size
