# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import os

import pytest
import torch

from renate.benchmark.models.mlp import MultiLayerPerceptron
from renate.benchmark.models.resnet import ResNet
from renate.benchmark.models.vision_transformer import VisionTransformer
from renate.defaults import TASK_ID
from renate.models import RenateModule
from renate.models.renate_module import RenateWrapper
from renate.utils.deepspeed import recover_object_from_tensor


def test_failing_to_init_abs_class():
    with pytest.raises(TypeError):
        RenateModule({"toy_hyperparam": 1.0}, pytest.helpers.get_loss_fn())


@pytest.mark.parametrize(
    "model",
    [
        pytest.helpers.get_renate_module_mlp(
            num_inputs=10, num_outputs=10, hidden_size=32, num_hidden_layers=3
        ),
        pytest.helpers.get_renate_vision_module(model="resnet", sub_class="resnet18cifar"),
        pytest.helpers.get_renate_vision_module(
            model="visiontransformer", sub_class="visiontransformercifar"
        ),
    ],
)
def test_renate_model_save(tmpdir, model):
    torch.save(model.state_dict(), os.path.join(tmpdir, "test_model.pt"))
    state = torch.load(os.path.join(tmpdir, "test_model.pt"))
    os.remove(os.path.join(tmpdir, "test_model.pt"))

    state["_extra_state"] = recover_object_from_tensor(state["_extra_state"])
    assert "constructor_arguments" in state["_extra_state"].keys()
    assert "tasks_params_ids" in state["_extra_state"].keys()


@pytest.mark.parametrize(
    "test_case,test_cls",
    [
        [
            pytest.helpers.get_renate_module_mlp_and_data(
                num_inputs=10,
                num_outputs=10,
                hidden_size=32,
                num_hidden_layers=3,
                train_num_samples=5,
                test_num_samples=5,
            ),
            MultiLayerPerceptron,
        ],
        [
            pytest.helpers.get_renate_vision_module_and_data(
                model="resnet",
                sub_class="resnet18cifar",
                input_size=(3, 32, 32),
                num_outputs=10,
                train_num_samples=5,
                test_num_samples=5,
            ),
            ResNet,
        ],
        [
            pytest.helpers.get_renate_vision_module_and_data(
                model="visiontransformer",
                sub_class="visiontransformercifar",
                input_size=(3, 32, 32),
                num_outputs=10,
                train_num_samples=5,
                test_num_samples=5,
            ),
            VisionTransformer,
        ],
    ],
)
def test_renate_model_singlehead_save_and_load(tmpdir, test_case, test_cls):
    model, _, test_data = test_case
    model.eval()

    y = torch.randint(3, 8, (5,))
    y_hat_pre_save = model(test_data)
    loss_pre_save = pytest.helpers.get_loss_fn("mean")(y_hat_pre_save, y)

    torch.save(model.state_dict(), os.path.join(tmpdir, "test_model.pt"))
    state = torch.load(os.path.join(tmpdir, "test_model.pt"))
    os.remove(os.path.join(tmpdir, "test_model.pt"))

    model2 = test_cls.from_state_dict(state)
    model2.eval()
    y_hat_post_load = model2(test_data)
    loss_post_load = pytest.helpers.get_loss_fn("mean")(y_hat_post_load, y)

    assert torch.allclose(y_hat_pre_save, y_hat_post_load)
    assert torch.allclose(loss_post_load, loss_pre_save)


@pytest.mark.parametrize(
    "test_case,test_cls",
    [
        [
            pytest.helpers.get_renate_module_mlp_and_data(
                num_inputs=10,
                num_outputs=10,
                hidden_size=32,
                num_hidden_layers=3,
                train_num_samples=5,
                test_num_samples=5,
            ),
            MultiLayerPerceptron,
        ],
        [
            pytest.helpers.get_renate_vision_module_and_data(
                model="resnet",
                sub_class="resnet18cifar",
                input_size=(3, 32, 32),
                num_outputs=10,
                train_num_samples=5,
                test_num_samples=5,
            ),
            ResNet,
        ],
        [
            pytest.helpers.get_renate_vision_module_and_data(
                model="visiontransformer",
                sub_class="visiontransformercifar",
                input_size=(3, 32, 32),
                num_outputs=10,
                train_num_samples=5,
                test_num_samples=5,
            ),
            VisionTransformer,
        ],
    ],
)
def test_renate_model_multihead_save_and_load(tmpdir, test_case, test_cls):
    model, _, test_data = test_case

    model.add_task_params(task_id="first_task")
    model.add_task_params(task_id="second_task")
    model.add_task_params(task_id="third_task")
    model.eval()

    y_hat_first = model(test_data, task_id="first_task")
    y_hat_second = model(test_data, task_id="second_task")
    y_hat_third = model(test_data, task_id="third_task")

    torch.save(model.state_dict(), os.path.join(tmpdir, "test_multihead_model.pt"))
    state = torch.load(os.path.join(tmpdir, "test_multihead_model.pt"))
    os.remove(os.path.join(tmpdir, "test_multihead_model.pt"))

    model2 = test_cls.from_state_dict(state)
    model2.eval()

    y_hat_first_loaded = model2(test_data, task_id="first_task")
    y_hat_second_loaded = model2(test_data, task_id="second_task")
    y_hat_third_loaded = model2(test_data, task_id="third_task")

    assert torch.allclose(y_hat_first_loaded, y_hat_first)
    assert torch.allclose(y_hat_second_loaded, y_hat_second)
    assert torch.allclose(y_hat_third_loaded, y_hat_third)


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.helpers.get_renate_module_mlp_and_data(
            num_inputs=10,
            num_outputs=10,
            hidden_size=32,
            num_hidden_layers=3,
            train_num_samples=5,
            test_num_samples=5,
        ),
        pytest.helpers.get_renate_vision_module_and_data(
            model="resnet",
            sub_class="resnet18cifar",
            input_size=(3, 32, 32),
            num_outputs=10,
            train_num_samples=5,
            test_num_samples=5,
        ),
        pytest.helpers.get_renate_vision_module_and_data(
            model="visiontransformer",
            sub_class="visiontransformercifar",
            input_size=(3, 32, 32),
            num_outputs=10,
            train_num_samples=32,
            test_num_samples=32,
        ),
    ],
)
def test_renate_model_add_params(test_case):
    model, train_dataset, _ = test_case
    model_updater = pytest.helpers.get_simple_updater(model)
    model = model_updater.update(train_dataset=train_dataset, task_id="second_task")
    assert model._tasks_params_ids == {TASK_ID, "second_task"}
    model = model_updater.update(train_dataset=train_dataset, task_id="second_task")
    assert model._tasks_params_ids == {TASK_ID, "second_task"}
    model = model_updater.update(train_dataset=train_dataset, task_id="third_task")
    assert model._tasks_params_ids == {TASK_ID, "second_task", "third_task"}


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.helpers.get_renate_module_mlp_and_data(
            num_inputs=10,
            num_outputs=10,
            hidden_size=32,
            num_hidden_layers=3,
            train_num_samples=5,
            test_num_samples=5,
        ),
        pytest.helpers.get_renate_vision_module_and_data(
            model="resnet",
            sub_class="resnet18cifar",
            input_size=(3, 32, 32),
            num_outputs=10,
            train_num_samples=5,
            test_num_samples=5,
        ),
        pytest.helpers.get_renate_vision_module_and_data(
            model="visiontransformer",
            sub_class="visiontransformercifar",
            input_size=(3, 32, 32),
            num_outputs=10,
            train_num_samples=5,
            test_num_samples=5,
        ),
    ],
)
def test_renate_model_singlehead_single_param_update(test_case):
    model, train_dataset, _ = test_case
    original_model = copy.deepcopy(model)

    model_updater = pytest.helpers.get_simple_updater(model)
    model = model_updater.update(train_dataset=train_dataset, task_id=TASK_ID)
    state_dict = model.state_dict()

    for name, p in original_model.named_parameters():
        assert not torch.equal(p, state_dict[name])


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.helpers.get_renate_module_mlp_and_data(
            num_inputs=10,
            num_outputs=10,
            hidden_size=32,
            num_hidden_layers=3,
            train_num_samples=5,
            test_num_samples=5,
        ),
        pytest.helpers.get_renate_vision_module_and_data(
            model="resnet",
            sub_class="resnet18cifar",
            input_size=(3, 32, 32),
            num_outputs=10,
            train_num_samples=5,
            test_num_samples=5,
        ),
        pytest.helpers.get_renate_vision_module_and_data(
            model="visiontransformer",
            sub_class="visiontransformercifar",
            input_size=(3, 32, 32),
            num_outputs=10,
            train_num_samples=5,
            test_num_samples=5,
        ),
    ],
)
def test_renate_multihead_multi_param_update(test_case):
    model, train_dataset, _ = test_case
    original_model = copy.deepcopy(model)
    model_updater = pytest.helpers.get_simple_updater(model)
    model = model_updater.update(train_dataset=train_dataset, task_id=TASK_ID)
    first_update_model = copy.deepcopy(model)
    model = model_updater.update(train_dataset=train_dataset, task_id="second_task")
    second_update_model = model

    first_update_state_dict = first_update_model.state_dict()
    second_update_state_dict = second_update_model.state_dict()

    for name, p in original_model.named_parameters():
        assert not torch.equal(p, first_update_state_dict[name])
        assert not torch.equal(p, second_update_state_dict[name])

    assert torch.equal(
        first_update_model._tasks_params[TASK_ID].weight,
        second_update_model._tasks_params[TASK_ID].weight,
    )
    assert torch.equal(
        first_update_model._tasks_params[TASK_ID].bias,
        second_update_model._tasks_params[TASK_ID].bias,
    )

    for name, p in first_update_model.named_parameters():
        if TASK_ID in name:
            continue
        assert not torch.equal(p, second_update_state_dict[name])


@pytest.mark.parametrize(
    "torch_model",
    [torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 3))],
)
@torch.no_grad()
def test_renate_wrapper_save_and_load(tmpdir, torch_model):
    renate_module = RenateWrapper(torch_model)
    X = torch.randn(10, 3)
    output_before = renate_module(X)
    assert torch.equal(output_before, torch_model(X))

    torch.save(renate_module.state_dict(), os.path.join(tmpdir, "test_model.pt"))
    del renate_module
    state = torch.load(os.path.join(tmpdir, "test_model.pt"))
    os.remove(os.path.join(tmpdir, "test_model.pt"))

    renate_module = RenateWrapper(torch_model)
    renate_module.load_state_dict(state)

    output_after = renate_module(X)
    assert torch.equal(output_after, output_before)


def test_renate_wrapper_forbids_from_state_dict():
    renate_module = RenateWrapper(torch.nn.Linear(1, 1))
    state_dict = renate_module.state_dict()
    del renate_module
    with pytest.raises(NotImplementedError):
        RenateWrapper.from_state_dict(state_dict)
