# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from avalanche.models import TrainEvalModel
from avalanche.training import ICaRLLossPlugin
from avalanche.training.plugins import EWCPlugin, LwFPlugin
from torch.optim import Adam, SGD

from conftest import (
    AVALANCHE_LEARNERS,
    AVALANCHE_LEARNER_HYPERPARAMETER_UPDATES,
    AVALANCHE_LEARNER_KWARGS,
)
from renate.updaters.avalanche.learner import (
    AvalancheEWCLearner,
    AvalancheICaRLLearner,
    AvalancheLwFLearner,
)
from renate.utils.avalanche import plugin_by_class


def check_learner_settings(
    learner,
    learner_kwargs,
    avalanche_learner,
    expected_model,
    expected_optimizer,
    expected_max_epochs,
    expected_device,
    expected_eval_every,
    expected_batch_size,
    expected_loss_fn=None,
):
    if isinstance(learner, AvalancheICaRLLearner):
        assert isinstance(avalanche_learner._criterion, ICaRLLossPlugin)
        assert avalanche_learner.eval_every == -1
        assert isinstance(avalanche_learner.model, TrainEvalModel)
        assert avalanche_learner.model.feature_extractor == expected_model.get_backbone()
        assert avalanche_learner.model.classifier == expected_model.get_predictor()
    else:
        assert avalanche_learner.eval_every == expected_eval_every
        assert avalanche_learner.model == expected_model
        if expected_loss_fn is None:
            assert avalanche_learner._criterion == learner_kwargs["loss_fn"]
        else:
            assert avalanche_learner._criterion == expected_loss_fn
    assert avalanche_learner.optimizer == expected_optimizer
    assert avalanche_learner.train_epochs == expected_max_epochs
    assert avalanche_learner.train_mb_size == expected_batch_size
    assert avalanche_learner.eval_mb_size == learner_kwargs["batch_size"]

    assert avalanche_learner.device == expected_device
    if isinstance(learner, AvalancheEWCLearner):
        assert (
            plugin_by_class(EWCPlugin, avalanche_learner.plugins).ewc_lambda
            == learner_kwargs["ewc_lambda"]
        )
    elif isinstance(learner, AvalancheLwFLearner):
        assert (
            plugin_by_class(LwFPlugin, avalanche_learner.plugins).lwf.alpha
            == learner_kwargs["alpha"]
        )
        assert (
            plugin_by_class(LwFPlugin, avalanche_learner.plugins).lwf.temperature
            == learner_kwargs["temperature"]
        )


@pytest.mark.parametrize("learner_class", AVALANCHE_LEARNERS)
def test_update_settings(learner_class):
    """Test settings after Avalanche wrappers creation and update."""
    learner_kwargs = AVALANCHE_LEARNER_KWARGS[learner_class]
    expected_model = pytest.helpers.get_renate_module_mlp(
        num_inputs=1,
        num_outputs=1,
        hidden_size=1,
        num_hidden_layers=1,
        add_icarl_class_means=learner_class == AvalancheICaRLLearner,
    )
    plugins = []
    expected_max_epochs = 10
    expected_loss_fn = pytest.helpers.get_loss_fn("mean")
    expected_optimizer = SGD(expected_model.parameters(), lr=0.1)
    expected_device = torch.device("cpu")
    expected_eval_every = -1
    expected_batch_size = learner_kwargs["batch_size"]
    if "batch_memory_frac" in learner_kwargs:
        expected_batch_size = expected_batch_size - int(
            learner_kwargs["batch_memory_frac"] * expected_batch_size
        )
    learner = learner_class(
        model=expected_model,
        optimizer=None,
        loss_fn=expected_loss_fn,
        **learner_kwargs,
    )
    avalanche_learner = learner.create_avalanche_learner(
        plugins=plugins,
        optimizer=expected_optimizer,
        train_epochs=expected_max_epochs,
        device=expected_device,
        eval_every=expected_eval_every,
    )
    check_learner_settings(
        learner,
        learner_kwargs,
        avalanche_learner,
        expected_model=expected_model,
        expected_loss_fn=expected_loss_fn,
        expected_optimizer=expected_optimizer,
        expected_max_epochs=expected_max_epochs,
        expected_device=expected_device,
        expected_eval_every=expected_eval_every,
        expected_batch_size=expected_batch_size,
    )

    # Update
    expected_max_epochs = 20
    expected_optimizer = Adam(expected_model.parameters(), lr=0.3)
    expected_device = torch.device("cuda")
    expected_eval_every = 1
    for key, value in AVALANCHE_LEARNER_HYPERPARAMETER_UPDATES[learner_class].items():
        setattr(learner, f"_{key}", value)
    learner.update_settings(
        avalanche_learner=avalanche_learner,
        plugins=plugins,
        optimizer=expected_optimizer,
        max_epochs=expected_max_epochs,
        device=expected_device,
        eval_every=expected_eval_every,
    )
    learner_kwargs.update(AVALANCHE_LEARNER_HYPERPARAMETER_UPDATES[learner_class])
    check_learner_settings(
        learner,
        learner_kwargs,
        avalanche_learner,
        expected_model=expected_model,
        expected_loss_fn=expected_loss_fn,
        expected_optimizer=expected_optimizer,
        expected_max_epochs=expected_max_epochs,
        expected_device=expected_device,
        expected_eval_every=expected_eval_every,
        expected_batch_size=expected_batch_size,
    )
