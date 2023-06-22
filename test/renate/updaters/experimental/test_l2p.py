# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# import pytest
import torch

from renate.updaters.experimental.l2p import (
    LearningToPromptLearner,
    PromptedVisionTransformer,
    PromptPool,
)
from renate.benchmark.models import VisionTransformerB16


def test_prompt_pool():
    D_emb = 2
    Lp = 3
    feat_dim = 6
    B = 4

    pool = PromptPool(embedding_dim=feat_dim, prompt_key_dim=D_emb, prompt_size=Lp)
    input = torch.rand(B, D_emb)
    out = pool(input)

    assert out.shape == (B, Lp * pool.N, feat_dim)


def test_prompted_vision_transformer():
    model = VisionTransformerB16()
    prompt = PromptPool(embedding_dim=model._embedding_size, prompt_key_dim=model._embedding_size)

    combined = PromptedVisionTransformer(model, prompt, "cls")

    inp = torch.rand(1, 3, 224, 224)

    print(combined(inp).shape)
    # print(combined)


if __name__ == "__main__":
    # model = VisionTransformerB16()
    # inp = torch.rand(10, 3, 224, 224)
    # # print(model.get_logits(inp).shape)
    # print(model._backbone(inp).shape)
    test_prompted_vision_transformer()
