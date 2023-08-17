# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.benchmark.models.l2p import LearningToPromptTransformer, PromptPool


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
