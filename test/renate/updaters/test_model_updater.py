# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import warnings
from copy import deepcopy

import pytest
import torch
from torchvision.transforms import Lambda

from conftest import LEARNERS_USING_SIMPLE_UPDATER, LEARNER_KWARGS, check_learner_transforms
from renate import defaults
from renate.updaters.experimental.repeated_distill import RepeatedDistillationModelUpdater
from renate.updaters.learner import ReplayLearner


@pytest.mark.parametrize("provide_folder", [True, False])
def test_simple_model_updater(tmpdir, provide_folder):
    model, train_dataset, test_data = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=32,
        num_hidden_layers=3,
        train_num_samples=10,
        test_num_samples=5,
    )
    model_updater = pytest.helpers.get_simple_updater(
        model, output_state_folder=defaults.output_state_folder(tmpdir) if provide_folder else None
    )
    y_hat_before_train = model(test_data, task_id=defaults.TASK_ID)
    model = model_updater.update(train_dataset, task_id=defaults.TASK_ID)
    y_hat_after_train = model(test_data, task_id=defaults.TASK_ID)
    assert y_hat_before_train.shape[0] == y_hat_after_train.shape[0]
    assert not torch.allclose(y_hat_before_train, y_hat_after_train)


def test_deterministic_updater():
    # The behavior is always deterministic on CPU but it can become non-deterministic on GPU
    # When run on CPU this test never fails so it is only useful when tests are run on GPU
    model1, train_dataset, test_data = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=32,
        num_hidden_layers=3,
        train_num_samples=10,
        test_num_samples=5,
    )

    model2 = deepcopy(model1)

    model_updater1 = pytest.helpers.get_simple_updater(
        model1,
        deterministic_trainer=True,
    )

    model_updater2 = pytest.helpers.get_simple_updater(
        model2,
        deterministic_trainer=True,
    )

    model1 = model_updater1.update(train_dataset, task_id=defaults.TASK_ID)
    model2 = model_updater2.update(train_dataset, task_id=defaults.TASK_ID)

    y_hat_1 = model1(test_data, task_id=defaults.TASK_ID)
    y_hat_2 = model2(test_data, task_id=defaults.TASK_ID)

    assert torch.allclose(y_hat_1, y_hat_2)


@pytest.mark.parametrize("early_stopping_enabled", [True, False])
@pytest.mark.parametrize("use_val", [True, False])
@pytest.mark.parametrize("metric_monitored", [None, "val_accuracy"])
@pytest.mark.parametrize("updater_type", ["DMC", "SimpleUpdater"])
def test_model_updater_with_early_stopping(
    use_val, early_stopping_enabled, metric_monitored, updater_type
):
    model, train_dataset, val_dataset = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=8,
        num_hidden_layers=1,
        train_num_samples=10,
        test_num_samples=5,
        val_num_samples=10,
    )

    max_epochs = 10
    with warnings.catch_warnings(record=True) as warning_init:
        if updater_type == "DMC":
            model_updater = RepeatedDistillationModelUpdater(
                model=model,
                memory_size=50,
                max_epochs=max_epochs,
                early_stopping_enabled=early_stopping_enabled,
                metric=metric_monitored,
            )
        else:
            model_updater = pytest.helpers.get_simple_updater(
                model=model,
                max_epochs=max_epochs,
                early_stopping_enabled=early_stopping_enabled,
                metric=metric_monitored,
            )

    assert model_updater._num_epochs_trained == 0
    with warnings.catch_warnings(record=True) as warning_update:
        model_updater.update(
            train_dataset, val_dataset=val_dataset if use_val else None, task_id=defaults.TASK_ID
        )

    if metric_monitored and use_val and early_stopping_enabled:
        assert model_updater._num_epochs_trained < max_epochs
    else:
        assert model_updater._num_epochs_trained == max_epochs

    is_warning_metric_missing_sent = any(
        [
            str(w.message).startswith("Early stopping is enabled but no metric is specified")
            for w in warning_init
        ]
    )
    is_warning_early_stopping_without_val_set_sent = any(
        [
            str(w.message).startswith(
                "Early stopping is currently not supported without a validation set"
            )
            for w in warning_update
        ]
    )
    assert not metric_monitored and early_stopping_enabled or not is_warning_metric_missing_sent
    assert (
        metric_monitored
        and early_stopping_enabled
        and not use_val
        or not is_warning_early_stopping_without_val_set_sent
    )


@pytest.mark.parametrize("learner_class", LEARNERS_USING_SIMPLE_UPDATER)
def test_continuation_of_training_with_simple_model_updater(tmpdir, learner_class):
    model, train_dataset, _ = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=32,
        num_hidden_layers=1,
        train_num_samples=10,
        test_num_samples=5,
    )
    state_url = defaults.input_state_folder(tmpdir)
    model_updater = pytest.helpers.get_simple_updater(
        model,
        learner_class=learner_class,
        learner_kwargs=LEARNER_KWARGS[learner_class],
        output_state_folder=state_url,
        max_epochs=2,
    )
    model = model_updater.update(train_dataset, task_id=defaults.TASK_ID)
    model_updater = pytest.helpers.get_simple_updater(
        model,
        learner_class=learner_class,
        learner_kwargs=LEARNER_KWARGS[learner_class],
        input_state_folder=state_url,
        max_epochs=2,
    )
    model_updater.update(train_dataset, task_id=defaults.TASK_ID)


@pytest.mark.parametrize("learner_class", LEARNERS_USING_SIMPLE_UPDATER)
def test_transforms_passed_to_simple_model_updater_will_be_used_by_learner(tmpdir, learner_class):
    """Checks if all transforms are correctly forwarded to Learner and MemoryBuffer.

    There are two cases: passing transforms
        1) when creating Learner from scratch and
        2) when loading Learner from checkpoint
    """

    def identity_transform():
        return Lambda(lambda x: x)

    model, train_dataset, _ = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=32,
        num_hidden_layers=3,
        train_num_samples=10,
        test_num_samples=5,
    )
    state_url = defaults.input_state_folder(tmpdir)
    transforms_kwargs = {
        "train_transform": identity_transform(),
        "train_target_transform": identity_transform(),
        "test_transform": identity_transform(),
        "test_target_transform": identity_transform(),
    }
    if issubclass(learner_class, ReplayLearner):
        transforms_kwargs["buffer_transform"] = identity_transform()
        transforms_kwargs["buffer_target_transform"] = identity_transform()
    model_updater = pytest.helpers.get_simple_updater(
        model,
        output_state_folder=state_url,
        learner_kwargs=LEARNER_KWARGS[learner_class],
        learner_class=learner_class,
        **transforms_kwargs,
    )
    check_learner_transforms(model_updater._learner, transforms_kwargs)
    model = model_updater.update(train_dataset, task_id=defaults.TASK_ID)
    model_updater = pytest.helpers.get_simple_updater(
        model, input_state_folder=state_url, learner_class=learner_class, **transforms_kwargs
    )
    check_learner_transforms(model_updater._learner, transforms_kwargs)
