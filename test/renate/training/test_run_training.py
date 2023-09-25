# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from syne_tune.config_space import loguniform, uniform
from syne_tune.optimizer.baselines import ASHA
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.transfer_learning import RUSHScheduler

from renate import defaults
from renate.training.training import (
    RENATE_CONFIG_COLUMNS,
    _create_scheduler,
    _get_transfer_learning_task_evaluations,
    _load_tuning_history,
    _merge_tuning_history,
    _verify_validation_set_for_hpo_and_checkpointing,
    run_training_job,
)
from renate.utils.syne_tune import is_syne_tune_config_space

config_file = str(Path(__file__).parent.parent / "renate_config_files" / "config.py")
config_file_custom_optimizer = str(
    Path(__file__).parent.parent / "renate_config_files" / "config_custom_optimizer.py"
)


@pytest.mark.parametrize(
    "num_chunks, val_size, raises, fixed_search_space, scheduler, configuration_file",
    [
        (2, 0.9, False, False, "rush", config_file),
        (1, 0.0, True, False, "rush", config_file),
        (2, 0.9, False, True, None, config_file),
        (2, 0.0, False, True, None, config_file),
        (1, 0.0, False, True, None, config_file_custom_optimizer),
    ],
    ids=[
        "transfer-hpo-with-val",
        "transfer-hpo-without-val-raises-exception",
        "training-single-config-with-val",
        "training-single-config-without-val",
        "training-single-config-without-val-custom-optimizer",
    ],
)
@pytest.mark.parametrize("updater", ("ER", "Avalanche-iCaRL"))
def test_run_training_job(
    tmpdir, num_chunks, val_size, raises, fixed_search_space, scheduler, updater, configuration_file
):
    """Simply running tuning job to check if anything fails.

    Case 1: Standard HPO setup with transfer learning in second step.
    Case 2: HPO without validation set fails.
    Case 3: Training of single configuration with validation set.
    Case 4: Training of single configuration without validation set.
    Case 5: Training of single configuration without validation set using custom optimizer and
        learning rate scheduler..
    """
    state_url = None
    tmpdir = str(tmpdir)
    for _ in range(num_chunks):

        def execute_job():
            config_space = {"val_size": val_size}
            if configuration_file == config_file:
                config_space["learning_rate"] = (
                    0.1 if fixed_search_space else loguniform(10e-5, 0.1)
                )
            run_training_job(
                updater=updater,
                max_epochs=5,
                config_file=configuration_file,
                input_state_url=state_url,
                output_state_url=tmpdir,
                backend="local",
                mode="max",
                config_space=config_space,
                metric="val_accuracy",
                max_time=35,
                scheduler=scheduler,
            )

        if raises:
            with pytest.raises(
                AssertionError, match="Provide a validation set to optimize hyperparameters."
            ):
                execute_job()
            return
        execute_job()
        state_url = tmpdir


@pytest.mark.parametrize(
    "data_old, data_new, expected_data_first_row, expected_data_second_row",
    [
        (
            [[0, 0.001, 0.01]],
            [[0.002, 0.8, 0.02]],
            [0, 0.001, np.nan, 0.01],
            [1, 0.002, 0.8, 0.02],
        ),
        (None, [[0.002, 0.8, 0.02]], [0, 0.002, 0.8, 0.02], None),
    ],
)
def test_merge_tuning_history(
    data_old, data_new, expected_data_first_row, expected_data_second_row
):
    """Test whether HPO results are merged correctly.

    Testing two cases:
        1. Old results exist and new ones are added.
        2. No old results exist.
    """
    results_old = None
    if data_old is not None:
        results_old = pd.DataFrame(data_old, columns=["update_id", "val_loss", "config_lr"])
    results_new = pd.DataFrame(data_new, columns=["val_loss", "val_accuracy", "config_lr"])
    merged_results = _merge_tuning_history(
        new_tuning_results=results_new, old_tuning_results=results_old
    )
    np.testing.assert_array_equal(
        merged_results[["update_id", "val_loss", "val_accuracy", "config_lr"]].iloc[0],
        expected_data_first_row,
    )
    if expected_data_second_row is not None:
        np.testing.assert_array_equal(
            merged_results[["update_id", "val_loss", "val_accuracy", "config_lr"]].iloc[1],
            expected_data_second_row,
        )


@pytest.mark.parametrize("val_size", [0.9, 0.0])
@pytest.mark.parametrize("tune_hyperparameters", [True, False])
def test_verify_validation_set_for_hpo_and_checkpointing(tmpdir, val_size, tune_hyperparameters):
    """Check if misconfigurations are spotted and config_space is updated correctly.

    If tune_hyperparameters is `True` (hyperparameter optimization is enabled), a validation set
    must exist.
    If a validation set exists, the `config_space` must be changed such that the right metric and
    mode for checkpointing and hyperparameter optimization is used.
    """
    config_space = {"val_size": val_size}
    expected_metric = "val_accuracy"
    expected_mode: defaults.SUPPORTED_TUNING_MODE_TYPE = "max"

    def verify_validation_set():
        return _verify_validation_set_for_hpo_and_checkpointing(
            config_space=config_space,
            config_file=config_file,
            tune_hyperparameters=tune_hyperparameters,
            metric=expected_metric,
            mode=expected_mode,
            working_directory=tmpdir,
        )

    if tune_hyperparameters and val_size == 0:
        with pytest.raises(AssertionError):
            verify_validation_set()
    else:
        metric, mode = verify_validation_set()
        if val_size > 0:
            assert config_space.get("metric") == expected_metric
            assert config_space.get("mode") == expected_mode
            assert metric == expected_metric
            assert mode == expected_mode
        else:
            assert metric == "train_loss"
            assert mode == "min"


def test_is_syne_tune_config_space():
    """Function should return True if any entry in the `config_space` is a Syne Tune search
    space.
    """
    assert is_syne_tune_config_space({"a": uniform(0, 1), "b": 1})
    assert not is_syne_tune_config_space({"a": 0.5, "b": 1})


def _get_transfer_learning_task_evaluations_input():
    """Helper function which returns the input required for the unit tests below."""
    hyperparameter_names = ["config_lr"]
    objectives_names = ["train_loss", "val_loss"]

    tuning_results = pd.DataFrame(
        [
            [[0.3, 0.2], [0.33, 0.23], 0.1, "irrelevant"],
            [[0.2, 0.1, 0.01], [0.23, 0.13, 0.03], 0.01, "irrelevant"],
            [[0.4], [0.5], 0.4, "irrelevant"],
            [[0.5], [0.6], 0.5, "irrelevant"],
            [[0.51], [0.61], 0.5, "irrelevant"],  # duplicate hyperparameter
            [[0.5], [np.nan], 0.3, "irrelevant"],  # missing metric
            [[0.5], [0.6], np.nan, "irrelevant"],  # missing hyperparameter
        ],
        columns=objectives_names + hyperparameter_names + ["irrelevant_columns"],
    )
    config_space = {"lr": uniform(0.001, 1)}
    for renate_config in RENATE_CONFIG_COLUMNS:
        if renate_config != "state_url":
            tuning_results[f"config_{renate_config}"] = f"{renate_config}_old"
        config_space[renate_config] = f"{renate_config}_new"

    return {
        "tuning_results": tuning_results,
        "config_space": config_space,
        "metric": objectives_names[1],
        "max_epochs": 3,
    }


def test_get_transfer_learning_task_evaluations():
    """Case that should successfully return a result.

    Function must drop duplicate hyperparameters, rows where metric or hyperparameter are missing.
    Only the relevant metric is kept. Renate configurations are replaced with the current ones
    (e.g. working space).
    """
    input_dict = _get_transfer_learning_task_evaluations_input()
    expected_objectives_evaluations = np.array(
        [
            [[[0.33], [0.23], [np.nan]]],
            [[[0.23], [0.13], [0.03]]],
            [[[0.5], [np.nan], [np.nan]]],
            [[[0.6], [np.nan], [np.nan]]],
        ]
    )
    expected_hyperparameters = [0.1, 0.01, 0.4, 0.5]
    task_evaluations = _get_transfer_learning_task_evaluations(
        tuning_results=input_dict["tuning_results"],
        config_space=input_dict["config_space"],
        metric=input_dict["metric"],
        max_epochs=input_dict["max_epochs"],
    )
    assert task_evaluations is not None
    assert task_evaluations.objectives_names == [input_dict["metric"]]
    assert task_evaluations.objectives_evaluations.shape == (4, 1, 3, 1)
    np.testing.assert_array_equal(
        task_evaluations.objectives_evaluations, expected_objectives_evaluations
    )
    assert list(task_evaluations.hyperparameters.columns) == ["lr"] + RENATE_CONFIG_COLUMNS
    assert task_evaluations.hyperparameters["lr"].to_list() == expected_hyperparameters
    for hyperparameter in task_evaluations.hyperparameters:
        if hyperparameter != "lr":
            assert (
                task_evaluations.hyperparameters[hyperparameter] == f"{hyperparameter}_new"
            ).all()


@pytest.mark.parametrize(
    "updated_input_dict",
    [
        {"config_space": {"unknown_hyperparameter": uniform(0, 1), "lr": uniform(0, 1)}},
        {"metric": "val_unknown_metric"},
        {"tuning_results": pd.DataFrame(columns=["train_loss", "val_loss", "config_lr"])},
        {
            "tuning_results": _get_transfer_learning_task_evaluations_input()[
                "tuning_results"
            ].iloc[5:6]
        },
        {
            "tuning_results": _get_transfer_learning_task_evaluations_input()[
                "tuning_results"
            ].iloc[6:7]
        },
    ],
)
def test_get_transfer_learning_task_evaluations_failure_cases(updated_input_dict):
    """Cases in which the function should return `None` since metadata cannot be used.

    Case 1: The `config_space` contains a new hyperparameter.
    Case 2: No evaluations for the given `metric` exist.
    Case 3: No evaluations exist.
    Case 4: Metric column is all `np.nan`.
    Case 5: Any hyperparameter column contains only `np.nan`.
    """
    input_dict = _get_transfer_learning_task_evaluations_input()
    input_dict.update(updated_input_dict)
    task_evaluations = _get_transfer_learning_task_evaluations(
        tuning_results=input_dict["tuning_results"],
        config_space=input_dict["config_space"],
        metric=input_dict["metric"],
        max_epochs=input_dict["max_epochs"],
    )
    assert task_evaluations is None


def test_load_tuning_history_with_no_tuning_history_should_return_empty_dict(tmpdir):
    assert _load_tuning_history(tmpdir, {}, "val_loss") == {}


def _get_tuning_results():
    return pd.DataFrame(
        [
            [0, 0, 1, 0.3, 0.4, 0.1, np.nan],
            [0, 0, 2, 0.2, 0.3, 0.1, np.nan],
            [0, 0, 3, 0.1, 0.2, 0.1, np.nan],
            [1, 0, 1, 0.3, 0.4, 0.1, 0.8],
            [1, 1, 1, 0.2, 0.3, 0.05, 0.9],
            [1, 0, 2, 0.25, 0.45, 0.1, 0.8],
            [1, 2, 1, 0.1, 0.2, 0.05, 0.8],
        ],
        columns=[
            "update_id",
            "trial_id",
            "epoch",
            "train_loss",
            "val_loss",
            "config_lr",
            "config_momentum",
        ],
    )


@pytest.mark.parametrize("hyperparameter, num_task_evaluations", [("lr", 2), ("momentum", 1)])
def test_load_tuning_history(tmpdir, hyperparameter, num_task_evaluations):
    """Check if tuning results can be successfully converted to a list of
    TransferLearningTaskEvaluations.

    Case 1: Evaluations for hyperparameter `lr` is available for both updates. Return both.
    Case 2: Evaluations for hyperparameter `momentum` is only available for the second update.
        Drop first data.
    """
    tuning_results = _get_tuning_results()
    config_space = {hyperparameter: uniform(0.001, 1)}
    tuning_results.to_csv(defaults.hpo_file(tmpdir), index=False)
    task_evaluations = _load_tuning_history(tmpdir, config_space, "val_loss")
    assert len(task_evaluations) == num_task_evaluations
    assert None not in task_evaluations.values()


@pytest.mark.parametrize("use_dir", [True, False])
def test_load_tuning_history_when_no_previous_history_exists(tmpdir, use_dir):
    """Check if function returns empty dict in case no previous history exists.

    Case 1: state_url is None. Return empty dict.
    Case 2: state_url is a path which does not contain a tuning history. Return empty dict.
    """
    state_url = None
    if use_dir:
        state_url = tmpdir
    task_evaluations = _load_tuning_history(state_url, {"lr": 0.01}, "val_loss")
    assert task_evaluations == {}


@pytest.mark.parametrize(
    "tuning_results_exist", [True, False], ids=["with_tuning_results", "without_tuning_results"]
)
@pytest.mark.parametrize(
    "scheduler,scheduler_kwargs",
    [
        ("random", None),
        ("asha", None),
        ("rush", None),
        (FIFOScheduler, None),
        (ASHA, {"max_resource_attr": "max_epochs", "resource_attr": "epoch"}),
        (RUSHScheduler, {"max_resource_attr": "max_epochs", "resource_attr": "epoch"}),
    ],
    ids=["str_random", "str_asha", "str_rush", "class_random", "class_asha", "class_rush"],
)
def test_create_scheduler(tmpdir, scheduler, scheduler_kwargs, tuning_results_exist):
    """Test various ways of creating a scheduler with and without previous tuning results.

    Cases:
        1. Check if schedulers can be created by using a string only.
        2. Custom Syne Tune schedulers can be created by passing class and kwargs.
        3. If any of the above methods works with and without previous tuning results.
    """
    tuning_results = _get_tuning_results()
    state_url = None
    if tuning_results_exist:
        state_url = tmpdir
        tuning_results.to_csv(defaults.hpo_file(state_url), index=False)
    _create_scheduler(
        scheduler=scheduler,
        config_space={"lr": 0.01, "max_epochs": 3},
        metric="val_loss",
        mode="min",
        seed=0,
        scheduler_kwargs=scheduler_kwargs,
        input_state_url=state_url,
    )
