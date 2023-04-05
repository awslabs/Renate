# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

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
    "batch_size,memory_size,memory_batch_size",
    [[10, 10, 10], [20, 10, 10], [10, 100, 10], [10, 30, 1], [100, 10, 3]],
)
def test_er_overall_memory_size_after_update(batch_size, memory_size, memory_batch_size):
    model, dataset = get_model_and_dataset()
    learner_kwargs = {
        "memory_size": memory_size,
        "memory_batch_size": memory_batch_size,
        "batch_size": batch_size,
    }
    model_updater = pytest.helpers.get_simple_updater(
        model=model,
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


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        [ExperienceReplayLearner, {"alpha": 0.2, "memory_size": 10, "memory_batch_size": 10}],
        [DarkExperienceReplayLearner, {"alpha": 0.1, "beta": 0.3, "memory_size": 42}],
        [
            CLSExperienceReplayLearner,
            {
                "alpha": 0.3,
                "beta": 0.1,
                "stable_model_update_weight": 0.3,
                "plastic_model_update_weight": 0.3,
                "stable_model_update_probability": 0.3,
                "plastic_model_update_probability": 0.5,
                "memory_size": 42,
            },
        ],
        [
            PooledOutputDistillationExperienceReplayLearner,
            {"alpha": 0.3, "distillation_type": "pixel", "normalize": False, "memory_size": 42},
        ],
        [
            SuperExperienceReplayLearner,
            {
                "der_alpha": 0.2,
                "der_beta": 0.3,
                "sp_shrink_factor": 0.1,
                "sp_sigma": 0.3,
                "cls_alpha": 0.3,
                "cls_stable_model_update_weight": 0.4,
                "cls_plastic_model_update_weight": 0.4,
                "cls_stable_model_update_probability": 0.3,
                "cls_plastic_model_update_probability": 0.5,
                "pod_alpha": 0.1,
                "pod_distillation_type": "pixel",
                "pod_normalize": False,
                "memory_size": 42,
            },
        ],
    ],
)
def test_er_components_save_and_load(tmpdir, cls, kwargs):
    """This test saves the learner state and reloads it and verifies that the hyperparameters
    for the components were correctly set, saved and loaded."""
    model = pytest.helpers.get_renate_module_mlp(
        num_inputs=10, num_outputs=10, hidden_size=32, num_hidden_layers=3
    )
    learner = cls(model=model, **kwargs)
    torch.save(learner.state_dict(), os.path.join(tmpdir, "learner.pt"))
    learner = cls.__new__(cls)
    learner.load_state_dict(model, torch.load(os.path.join(tmpdir, "learner.pt")))
    if isinstance(learner, ExperienceReplayLearner) and not isinstance(
        learner, DarkExperienceReplayLearner
    ):
        assert learner._components["memory_loss"]._weight == kwargs["alpha"]
    if isinstance(learner, DarkExperienceReplayLearner):
        assert learner._components["mse_loss"]._weight == kwargs["alpha"]
        assert learner._components["memory_loss"]._weight == kwargs["beta"]
    elif isinstance(learner, PooledOutputDistillationExperienceReplayLearner):
        assert learner._components["pod_loss"]._weight == kwargs["alpha"]
        assert learner._components["pod_loss"]._distillation_type == kwargs["distillation_type"]
        assert learner._components["pod_loss"]._normalize == kwargs["normalize"]
    elif isinstance(learner, CLSExperienceReplayLearner):
        assert learner._components["memory_loss"]._weight == kwargs["alpha"]
        assert learner._components["cls_loss"]._weight == kwargs["beta"]
        assert (
            learner._components["cls_loss"]._stable_model_update_weight
            == kwargs["stable_model_update_weight"]
        )
        assert (
            learner._components["cls_loss"]._plastic_model_update_weight
            == kwargs["plastic_model_update_weight"]
        )
        assert (
            learner._components["cls_loss"]._stable_model_update_probability
            == kwargs["stable_model_update_probability"]
        )
        assert (
            learner._components["cls_loss"]._plastic_model_update_probability
            == kwargs["plastic_model_update_probability"]
        )
    elif isinstance(learner, SuperExperienceReplayLearner):
        assert learner._components["mse_loss"]._weight == kwargs["der_alpha"]
        assert learner._components["memory_loss"]._weight == kwargs["der_beta"]
        assert learner._components["shrink_perturb"]._shrink_factor == kwargs["sp_shrink_factor"]
        assert learner._components["shrink_perturb"]._sigma == kwargs["sp_sigma"]
        assert learner._components["cls_loss"]._weight == kwargs["cls_alpha"]
        assert (
            learner._components["cls_loss"]._stable_model_update_weight
            == kwargs["cls_stable_model_update_weight"]
        )
        assert (
            learner._components["cls_loss"]._plastic_model_update_weight
            == kwargs["cls_plastic_model_update_weight"]
        )
        assert (
            learner._components["cls_loss"]._stable_model_update_probability
            == kwargs["cls_stable_model_update_probability"]
        )
        assert (
            learner._components["cls_loss"]._plastic_model_update_probability
            == kwargs["cls_plastic_model_update_probability"]
        )
        assert learner._components["pod_loss"]._weight == kwargs["pod_alpha"]
        assert learner._components["pod_loss"]._distillation_type == kwargs["pod_distillation_type"]
        assert learner._components["pod_loss"]._normalize == kwargs["pod_normalize"]
