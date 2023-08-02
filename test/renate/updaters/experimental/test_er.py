# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from renate import defaults
from renate.updaters.experimental.er import (
    CLSExperienceReplayLearner,
    DarkExperienceReplayLearner,
    ExperienceReplayLearner,
    PooledOutputDistillationExperienceReplayLearner,
    SuperExperienceReplayLearner,
)


def get_model_and_dataset():
    model = pytest.helpers.get_renate_module_mlp(
        num_outputs=10, num_inputs=10, hidden_size=32, num_hidden_layers=3
    )
    dataset = torch.utils.data.TensorDataset(
        torch.rand((100, 10)),
        torch.randint(10, (100,)),
    )
    return model, dataset


@pytest.mark.parametrize(
    "batch_size,memory_size,batch_memory_frac",
    [[20, 10, 0.5], [30, 10, 0.34], [20, 100, 0.5], [10, 30, 0.1], [100, 10, 0.03]],
)
def test_er_overall_memory_size_after_update(batch_size, memory_size, batch_memory_frac):
    memory_batch_size = int(batch_memory_frac * batch_size)
    model, dataset = get_model_and_dataset()
    learner_kwargs = {
        "memory_size": memory_size,
        "batch_memory_frac": batch_memory_frac,
        "batch_size": batch_size,
    }
    model_updater = pytest.helpers.get_simple_updater(
        model=model,
        partial_optimizer=pytest.helpers.get_partial_optimizer(),
        learner_class=ExperienceReplayLearner,
        learner_kwargs=learner_kwargs,
        max_epochs=1,
    )
    model_updater.update(train_dataset=dataset, task_id=defaults.TASK_ID)
    memory, _ = next(iter(model_updater._learner._memory_loader))
    x, y = memory
    assert x.shape[0] == memory_batch_size and y.shape[0] == memory_batch_size
    assert len(model_updater._learner._memory_buffer) == memory_size


def test_er_validation_buffer(tmpdir):
    model, dataset_train, _ = pytest.helpers.get_renate_module_mlp_and_data(
        num_inputs=10,
        num_outputs=10,
        hidden_size=32,
        num_hidden_layers=3,
        train_num_samples=10,
        test_num_samples=5,
    )
    datasets_val = []
    state_folder = None
    next_state_folder = defaults.input_state_folder(tmpdir)
    for i in range(3):
        dataset_val = torch.utils.data.TensorDataset(
            torch.rand((100, 10)),
            torch.randint(10, (100,)),
        )
        model_updater = pytest.helpers.get_simple_updater(
            model, input_state_folder=state_folder, output_state_folder=next_state_folder
        )
        model_updater.update(
            train_dataset=dataset_train, val_dataset=dataset_val, task_id=defaults.TASK_ID
        )
        datasets_val.append(dataset_val)
        state_folder = next_state_folder

    model_updater = pytest.helpers.get_simple_updater(
        model, input_state_folder=state_folder, output_state_folder=next_state_folder
    )
    for i in range(3):
        for j in range(100):
            assert torch.allclose(
                datasets_val[i][j][0],
                model_updater._learner._val_memory_buffer[i * 100 + j][0][0],
                rtol=1e-3,
            )


def validate_common_args(model_updater, learner_kwargs):
    memory_batch_size = int(learner_kwargs["batch_memory_frac"] * learner_kwargs["batch_size"])
    batch_size = learner_kwargs["batch_size"] - memory_batch_size
    assert model_updater._learner._batch_size == batch_size
    assert model_updater._learner._memory_batch_size == memory_batch_size


def validate_cls_er(model_updater, learner_kwargs):
    validate_common_args(model_updater, learner_kwargs)
    assert model_updater._learner._components["memory_loss"].weight == learner_kwargs["alpha"]
    assert model_updater._learner._components["cls_loss"].weight == learner_kwargs["beta"]
    assert (
        model_updater._learner._components["cls_loss"]._stable_model_update_weight
        == learner_kwargs["stable_model_update_weight"]
    )
    assert (
        model_updater._learner._components["cls_loss"]._plastic_model_update_weight
        == learner_kwargs["plastic_model_update_weight"]
    )
    assert (
        model_updater._learner._components["cls_loss"]._stable_model_update_probability
        == learner_kwargs["stable_model_update_probability"]
    )
    assert (
        model_updater._learner._components["cls_loss"]._plastic_model_update_probability
        == learner_kwargs["plastic_model_update_probability"]
    )


def validate_dark_er(model_updater, learner_kwargs):
    validate_common_args(model_updater, learner_kwargs)
    assert model_updater._learner._components["memory_loss"].weight == learner_kwargs["beta"]
    assert model_updater._learner._components["mse_loss"].weight == learner_kwargs["alpha"]


def validate_pod_er(model_updater, learner_kwargs):
    validate_common_args(model_updater, learner_kwargs)
    assert model_updater._learner._components["pod_loss"].weight == learner_kwargs["alpha"]
    assert (
        model_updater._learner._components["pod_loss"]._distillation_type
        == learner_kwargs["distillation_type"]
    )
    assert model_updater._learner._components["pod_loss"]._normalize == learner_kwargs["normalize"]


def validate_super_er(model_updater, learner_kwargs):
    validate_common_args(model_updater, learner_kwargs)
    assert model_updater._learner._components["memory_loss"].weight == learner_kwargs["der_beta"]
    assert model_updater._learner._components["mse_loss"].weight == learner_kwargs["der_alpha"]
    assert model_updater._learner._components["cls_loss"].weight == learner_kwargs["cls_alpha"]
    assert (
        model_updater._learner._components["cls_loss"]._stable_model_update_weight
        == learner_kwargs["cls_stable_model_update_weight"]
    )
    assert (
        model_updater._learner._components["cls_loss"]._plastic_model_update_weight
        == learner_kwargs["cls_plastic_model_update_weight"]
    )
    assert (
        model_updater._learner._components["cls_loss"]._stable_model_update_probability
        == learner_kwargs["cls_stable_model_update_probability"]
    )
    assert (
        model_updater._learner._components["cls_loss"]._plastic_model_update_probability
        == learner_kwargs["cls_plastic_model_update_probability"]
    )
    assert model_updater._learner._components["pod_loss"].weight == learner_kwargs["pod_alpha"]
    assert (
        model_updater._learner._components["pod_loss"]._distillation_type
        == learner_kwargs["pod_distillation_type"]
    )
    assert (
        model_updater._learner._components["pod_loss"]._normalize == learner_kwargs["pod_normalize"]
    )
    assert (
        model_updater._learner._components["shrink_perturb"]._shrink_factor
        == learner_kwargs["sp_shrink_factor"]
    )
    assert model_updater._learner._components["shrink_perturb"]._sigma == learner_kwargs["sp_sigma"]


@pytest.mark.parametrize(
    "learner_class,validate_fct,learner_kwargs1,learner_kwargs2",
    [
        (
            CLSExperienceReplayLearner,
            validate_cls_er,
            {
                "memory_size": 30,
                "batch_memory_frac": 0.4,
                "batch_size": 50,
                "seed": 1,
                "alpha": 0.123,
                "beta": 2,
                "stable_model_update_weight": 0.2,
                "plastic_model_update_weight": 0.1,
                "stable_model_update_probability": 0.3,
                "plastic_model_update_probability": 0.4,
            },
            {
                "memory_size": 30,
                "batch_memory_frac": 0.1,
                "batch_size": 100,
                "seed": 1,
                "alpha": 2.3,
                "beta": 3,
                "stable_model_update_weight": 0.6,
                "plastic_model_update_weight": 0.5,
                "stable_model_update_probability": 0.7,
                "plastic_model_update_probability": 0.8,
            },
        ),
        (
            DarkExperienceReplayLearner,
            validate_dark_er,
            {
                "memory_size": 30,
                "batch_memory_frac": 0.4,
                "batch_size": 50,
                "seed": 1,
                "alpha": 0.123,
                "beta": 2,
            },
            {
                "memory_size": 30,
                "batch_memory_frac": 0.1,
                "batch_size": 100,
                "seed": 1,
                "alpha": 2.3,
                "beta": 3,
            },
        ),
        (
            PooledOutputDistillationExperienceReplayLearner,
            validate_pod_er,
            {
                "memory_size": 30,
                "batch_memory_frac": 0.4,
                "batch_size": 50,
                "seed": 1,
                "alpha": 0.123,
                "distillation_type": "spatial",
                "normalize": True,
            },
            {
                "memory_size": 30,
                "batch_memory_frac": 0.1,
                "batch_size": 100,
                "seed": 1,
                "alpha": 0.123,
                "distillation_type": "channel",
                "normalize": False,
            },
        ),
        (
            SuperExperienceReplayLearner,
            validate_super_er,
            {
                "memory_size": 30,
                "batch_memory_frac": 0.4,
                "batch_size": 50,
                "seed": 1,
                "der_alpha": 0.123,
                "der_beta": 2,
                "sp_shrink_factor": 0.33,
                "sp_sigma": 0.11,
                "cls_alpha": 2.3,
                "cls_stable_model_update_weight": 0.6,
                "cls_plastic_model_update_weight": 0.5,
                "cls_stable_model_update_probability": 0.7,
                "cls_plastic_model_update_probability": 0.8,
                "pod_alpha": 0.13,
                "pod_distillation_type": "spatial",
                "pod_normalize": True,
            },
            {
                "memory_size": 30,
                "batch_memory_frac": 0.1,
                "batch_size": 100,
                "seed": 1,
                "der_alpha": 2.3,
                "der_beta": 3,
                "sp_shrink_factor": 0.66,
                "sp_sigma": 0.22,
                "cls_alpha": 2.3,
                "cls_stable_model_update_weight": 0.6,
                "cls_plastic_model_update_weight": 0.5,
                "cls_stable_model_update_probability": 0.7,
                "cls_plastic_model_update_probability": 0.8,
                "pod_alpha": 0.123,
                "pod_distillation_type": "channel",
                "pod_normalize": False,
            },
        ),
    ],
)
def test_saving_and_loading_of_er_methods(
    tmpdir, learner_class, validate_fct, learner_kwargs1, learner_kwargs2
):
    """ER saving and loading test.

    The ER methods partially have custom saving and loading functions (CLS-ER). Furthermore, the
    components used to be Modules that effectively did not allow for changing hyperparameter
    settings.
    """
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
        learner_kwargs=learner_kwargs1,
        output_state_folder=state_url,
        max_epochs=2,
    )

    model = model_updater.update(train_dataset, task_id=defaults.TASK_ID)
    validate_fct(model_updater, learner_kwargs1)
    model_updater = pytest.helpers.get_simple_updater(
        model,
        learner_class=learner_class,
        learner_kwargs=learner_kwargs2,
        input_state_folder=state_url,
        max_epochs=2,
    )
    validate_fct(model_updater, learner_kwargs2)
