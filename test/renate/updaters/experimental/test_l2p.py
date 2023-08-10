# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from renate.benchmark.models.l2p import PromptedVisionTransformer, PromptPool
from renate.benchmark.models.vision_transformer import VisionTransformerB16
from renate.updaters.experimental.l2p import LearningToPromptLearner, LearningToPromptReplayLearner


def test_prompt_pool():
    D_emb = 2
    Lp = 3
    feat_dim = 6
    B = 4

    pool = PromptPool(embedding_dim=feat_dim, prompt_key_dim=D_emb, prompt_size=Lp)
    input = torch.rand(B, D_emb)
    out = pool(input)[0]

    assert out.shape == (B, Lp * pool._N, feat_dim)


def test_prompted_vision_transformer():
    combined = PromptedVisionTransformer()

    inp = torch.rand(1, 3, 224, 224)

    print(combined(inp).shape)
    assert combined(inp).shape == torch.Size((1, 10))
