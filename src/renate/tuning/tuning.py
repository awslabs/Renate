# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect
import json
import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch
from syne_tune.backend.local_backend import LocalBackend
from syne_tune.config_space import Domain
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import ASHA
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers import FIFOScheduler
from syne_tune.optimizer.schedulers.transfer_learning import (
    RUSHScheduler,
    TransferLearningTaskEvaluations,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from syne_tune.tuner_callback import StoreResultsCallback
from syne_tune.util import experiment_path

import renate
from renate import defaults
from renate.cli.parsing_functions import get_data_module_fn_args
from renate.utils.file import move_to_uri
from renate.utils.module import get_and_prepare_data_module, import_module
from renate.utils.syne_tune import (
    TrainingLoggerCallback,
    TuningLoggerCallback,
    best_hyperparameters,
    config_space_to_dict,
    is_syne_tune_config_space,
    redirect_to_tmp,
)

logger = logging.getLogger(__name__)

RENATE_CONFIG_COLUMNS = [
    "config_file",
    "prepare_data",
    "chunk_id",
    "task_id",
    "working_directory",
    "seed",
    "accelerator",
    "devices",
    "metric",
    "mode",
    "state_url",
]


def execute_tuning_job(
    mode: defaults.SUPPORTED_TUNING_MODE_TYPE,
    config_space: Dict,
    metric: str,
    backend: defaults.SUPPORTED_BACKEND_TYPE,
    updater: str = defaults.LEARNER,
    max_epochs: int = defaults.MAX_EPOCHS,
    task_id: str = defaults.TASK_ID,
    chunk_id: int = defaults.CHUNK_ID,
    state_url: Optional[str] = None,
    next_state_url: Optional[str] = None,
    working_directory: Optional[str] = defaults.WORKING_DIRECTORY,
    source_dir: Optional[str] = None,
    config_file: Optional[str] = None,
    requirements_file: Optional[str] = None,
    role: Optional[str] = None,
    instance_type: str = defaults.INSTANCE_TYPE,
    instance_count: int = defaults.INSTANCE_COUNT,
    instance_max_time: float = defaults.INSTANCE_MAX_TIME,
    max_time: Optional[float] = None,
    max_num_trials_started: Optional[int] = None,
    max_num_trials_completed: Optional[int] = None,
    max_num_trials_finished: Optional[int] = None,
    max_cost: Optional[float] = None,
    n_workers: int = defaults.N_WORKERS,
    scheduler: Optional[Union[str, Type[TrialScheduler]]] = None,
    scheduler_kwargs: Optional[Dict] = None,
    seed: int = defaults.SEED,
    accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
    devices: int = defaults.DEVICES,
    job_name: str = defaults.JOB_NAME,
) -> Optional[Tuner]:
    """Starts updating the model including hyperparameter optimization.

    Args:
        mode: Declares the type of optimization problem: `min` or `max`.
        config_space: Details for defining your own search space is provided in the
            `Syne Tune Documentation <https://github.com/awslabs/syne-tune/blob/main/docs/search_space.md>`_.
        metric: Name of metric to optimize.
        backend: Whether to run jobs locally (`local`) or on SageMaker (`sagemaker`).
        updater: Updater used for model update.
        max_epochs: Maximum number of epochs the model is trained.
        task_id: Unique identifier for the current task.
        chunk_id: Unique identifier for the current data chunk.
        state_url: Path to the Renate model state.
        next_state_url: Path where Renate model state will be stored.
        working_directory: Path to the working directory.
        source_dir: (SageMaker backend only) Root directory which will be moved to SageMaker.
        config_file: File containing the definition of `model_fn` and `data_module_fn`.
        requirements_file: (SageMaker backend only) Path to requirements.txt containing environment dependencies.
        role: (SageMaker backend only) An AWS IAM role (either name or full ARN).
        instance_type: (SageMaker backend only) Sagemaker instance type for each worker.
        instance_count: (SageMaker backend only) Number of instances for each worker.
        instance_max_time: (SageMaker backend only) Requested maximum wall_clock time for each worker.
        max_time: Stopping criterion: wall clock time.
        max_num_trials_started: Stopping criterion: trials started.
        max_num_trials_completed: Stopping criterion: trials completed.
        max_num_trials_finished: Stopping criterion: trials finished.
        max_cost: (SageMaker backend only) Stopping criterion: SageMaker cost.
        n_workers: Number of workers running in parallel.
        scheduler: Default is random search, you can change it by providing a either a string (`random`, `bo`, `asha` or
            `rush`) or scheduler class and its corresponding scheduler_kwargs if required. For latter option,
            `see details at <https://github.com/awslabs/syne-tune/blob/main/docs/schedulers.md>`_ .
        scheduler_kwargs: Only required if custom scheduler is provided.
        seed: Seed used for ensuring reproducibility.
        accelerator: Type of accelerator to use.
        devices: Number of devices to use.
        job_name: Name of the tuning job.
    """
    assert (
        mode in defaults.SUPPORTED_TUNING_MODE
    ), f"Mode {mode} is not in {defaults.SUPPORTED_TUNING_MODE}."
    assert (
        backend in defaults.SUPPORTED_BACKEND
    ), f"Backend {backend} is not in {defaults.SUPPORTED_BACKEND}."
    if backend == "local":
        return _execute_tuning_job_locally(
            state_url=state_url,
            next_state_url=next_state_url,
            working_directory=working_directory,
            config_file=config_file,
            mode=mode,
            config_space=config_space,
            metric=metric,
            updater=updater,
            max_epochs=max_epochs,
            task_id=task_id,
            chunk_id=chunk_id,
            max_time=max_time,
            max_num_trials_started=max_num_trials_started,
            max_num_trials_completed=max_num_trials_completed,
            max_num_trials_finished=max_num_trials_finished,
            max_cost=max_cost,
            n_workers=n_workers,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            seed=seed,
            accelerator=accelerator,
            devices=devices,
        )
    _execute_tuning_job_remotely(
        state_url=state_url,
        next_state_url=next_state_url,
        working_directory=working_directory,
        config_file=config_file,
        mode=mode,
        config_space=config_space,
        metric=metric,
        updater=updater,
        max_epochs=max_epochs,
        task_id=task_id,
        chunk_id=chunk_id,
        source_dir=source_dir,
        requirements_file=requirements_file,
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        instance_max_time=instance_max_time,
        max_time=max_time,
        max_num_trials_started=max_num_trials_started,
        max_num_trials_completed=max_num_trials_completed,
        max_num_trials_finished=max_num_trials_finished,
        max_cost=max_cost,
        n_workers=n_workers,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        seed=seed,
        accelerator=accelerator,
        devices=devices,
        job_name=job_name,
    )


def _prepare_remote_tuning_job(
    tmp_dir: str, requirements_file: Optional[str], **job_kwargs: Any
) -> List[str]:
    """Prepares a SageMaker tuning job."""
    dependencies = list(renate.__path__ + [job_kwargs["config_file"]])

    if "state_url" in job_kwargs and job_kwargs["state_url"] is None:
        del job_kwargs["state_url"]
    job_kwargs["config_file"] = os.path.basename(job_kwargs["config_file"])
    job_kwargs["config_space"] = config_space_to_dict(job_kwargs["config_space"])

    jobs_kwargs_file = os.path.join(tmp_dir, defaults.JOB_KWARGS_FILE)
    with open(jobs_kwargs_file, "w") as f:
        json.dump(job_kwargs, f)
    dependencies.append(jobs_kwargs_file)

    if requirements_file is None:
        requirements_file = os.path.join(tmp_dir, "requirements.txt")
        with open(requirements_file, "w") as f:
            f.write(f"renate=={renate.__version__}")
    dependencies.append(requirements_file)
    return dependencies


def _get_transfer_learning_task_evaluations(
    tuning_results: pd.DataFrame,
    config_space: Dict[str, Any],
    metric: str,
    max_epochs: int,
) -> Optional[TransferLearningTaskEvaluations]:
    """Converts data frame with tuning results of a single update step into `TransferLearningTaskEvaluations`.

    Args:
        tuning_results: Results of previous hyperparameter optimization runs.
        config_space: Configuration space used.
        metric: The metric to be optimized. This will be the only metric added to `TransferLearningTaskEvaluations`.
        max_epochs: Length of the learning curve. Learning curves will be padded with `np.nan` to this length.
    Returns:
        `TransferLearningTaskEvaluations` contains a `pd.DataFrame` of hyperparameters, a `np.array` of shape
        `(num_hyperparameters, 1, max_epochs, 1)` which contains the learning curves for `metric` for each
        hyperparameter.
    """
    for config in RENATE_CONFIG_COLUMNS:
        config_with_prefix = f"config_{config}"
        if config in config_space:
            tuning_results[config_with_prefix] = config_space[config]
        elif config_with_prefix in tuning_results:
            del tuning_results[config_with_prefix]
    if (
        metric not in tuning_results
        or not len(tuning_results)
        or not set([f"config_{config_name}" for config_name in config_space]).issubset(
            tuning_results
        )
    ):
        return None

    def pad_same_length(x):
        if len(x) == max_epochs:
            return x
        return x + (max_epochs - len(x)) * [np.nan]

    hyperparameter_names = [
        hyperparameter for hyperparameter in tuning_results if hyperparameter.startswith("config_")
    ]
    hyperparameters = tuning_results[hyperparameter_names]
    hyperparameters.columns = [
        hyperparameter_name[7:] for hyperparameter_name in hyperparameters.columns
    ]
    hyperparameters = hyperparameters[list(config_space)]
    hyperparameter_rows_with_nan = hyperparameters.isna().any(axis=1)
    hyperparameter_duplicates = hyperparameters.duplicated()
    objectives_evaluations = np.array(tuning_results[metric].apply(pad_same_length).to_list())
    objectives_evaluations = np.expand_dims(objectives_evaluations, axis=(1, 3))
    objectives_evaluations_with_nan_only = np.all(np.isnan(objectives_evaluations), axis=2).reshape(
        -1
    )
    valid_hyperparameters = np.logical_and(
        np.logical_and(~hyperparameter_rows_with_nan, ~objectives_evaluations_with_nan_only),
        ~hyperparameter_duplicates,
    )
    if not np.any(valid_hyperparameters):
        return None
    hyperparameters = hyperparameters[valid_hyperparameters]
    objectives_evaluations = objectives_evaluations[valid_hyperparameters]
    return TransferLearningTaskEvaluations(
        configuration_space=config_space,
        hyperparameters=hyperparameters,
        objectives_names=[metric],
        objectives_evaluations=objectives_evaluations,
    )


def _load_tuning_history(
    state_url: str, config_space: Dict[str, Any], metric: str
) -> Dict[str, TransferLearningTaskEvaluations]:
    """Loads the tuning history in a list where each entry of the list is the tuning history of one update.

    Args:
        state_url: Location of state. Will check at this location of a tuning history exists.
        config_space: The configuration space defines which parts of the tuning history to load.
        metric: Only the defined metric of the tuning history will be loaded.
    Returns:
        Returns an empty list if no previous tuning history exists or it does not match the current `config_space`.
        The list contains an instance of `TransferLearningTaskEvaluations` for each update that contains matching data.
    """
    if state_url is None or not Path(defaults.hpo_file(state_url)).exists():
        return {}
    tuning_results = pd.read_csv(defaults.hpo_file(state_url))
    hyperparameter_names = [
        hyperparameter for hyperparameter in tuning_results if hyperparameter.startswith("config_")
    ]
    agg_dict = {
        hyperparameter_name: lambda x: x.iloc[0] for hyperparameter_name in hyperparameter_names
    }
    agg_dict[metric] = list
    max_epochs = tuning_results["epoch"].max()
    task_evaluations = {}
    for update_id in tuning_results["update_id"].unique():
        task_evaluation = _get_transfer_learning_task_evaluations(
            tuning_results=tuning_results[tuning_results["update_id"] == update_id]
            .groupby(["trial_id"])
            .agg(agg_dict),
            config_space=config_space,
            metric=metric,
            max_epochs=max_epochs,
        )
        if task_evaluation is not None:
            task_evaluations[f"update_id_{update_id}"] = task_evaluation
    return task_evaluations


def _merge_tuning_history(
    new_tuning_results: pd.DataFrame, old_tuning_results: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Merges old tuning history with tuning results from current chunk.

    `update_id` identifies the update step in the csv file. This allows creating the metadata required for transfer
        hyperparameter optimization.
    """
    update_id = 0 if old_tuning_results is None else old_tuning_results["update_id"].max() + 1
    new_tuning_results.insert(0, "update_id", update_id)
    if old_tuning_results is not None:
        return pd.concat([old_tuning_results, new_tuning_results], axis=0, ignore_index=True)
    return new_tuning_results


def _teardown_tuning_job(
    backend: LocalBackend,
    config_space: Dict[str, Union[Domain, int, float, str]],
    job_name: str,
    state_url: Optional[str] = None,
    next_state_url: Optional[str] = None,
) -> None:
    """Update lifelong hyperparameter optimization results, save state and clean up disk."""
    experiment_folder = redirect_to_tmp(str(experiment_path(job_name)))
    if next_state_url is not None:
        experiment = load_experiment(job_name)
        try:
            best_trial_id = experiment.best_config()["trial_id"]
            if is_syne_tune_config_space(config_space):
                logger.info(
                    f"Best hyperparameter settings: {best_hyperparameters(experiment, config_space)}"
                )
        except AttributeError:
            raise RuntimeError(
                "Not a single training run finished. This may have two reasons:\n"
                "1) The provided tuning time is too short.\n"
                "2) There is a bug in the training script."
                + "\n\nLogs (stdout):\n\n{}".format("".join(backend.stdout(0)))
                + "\n\nLogs (stderr):\n\n{}".format("".join(backend.stderr(0)))
            )
        next_state_folder = defaults.next_state_folder(
            f"{experiment_folder}/{best_trial_id}/checkpoints"
        )
        old_tuning_results = (
            pd.read_csv(defaults.hpo_file(state_url))
            if state_url is not None and os.path.exists(defaults.hpo_file(state_url))
            else None
        )
        tuning_results = _merge_tuning_history(experiment.results, old_tuning_results)
        tuning_results.to_csv(defaults.hpo_file(next_state_folder), index=False)
        move_to_uri(next_state_folder, next_state_url)
        logger.info(f"Renate state is available at {next_state_url}.")
    shutil.rmtree(experiment_folder, ignore_errors=True)
    shutil.rmtree(experiment_path(job_name), ignore_errors=True)


def _verify_validation_set_for_hpo_and_checkpointing(
    config_space: Dict[str, Any],
    config_file: str,
    tune_hyperparameters: bool,
    metric: str,
    mode: defaults.SUPPORTED_TUNING_MODE_TYPE,
    working_directory: str,
    chunk_id: int,
    seed: int,
) -> Tuple[str, defaults.SUPPORTED_TUNING_MODE_TYPE]:
    """Checks if validation set is provided when needed and updates config_space such that checkpointing works.

    If a validation set exists, the metric is forwarded such that we store the checkpoint which performs best on
    validation. This is a side effect changing `config_space`. Otherwise, the checkpoint of the last epoch is used.

    Returns:
        Metric and mode used by the Syne Tune tuner. If there is no validation set, returns `("train_loss", "min")`.
    Raises:
        AssertionError: If `tune_hyperparameters` is True but no validation set is provided.
    """
    config_module = import_module("config_module", config_file)
    data_module = get_and_prepare_data_module(
        config_module,
        data_path=defaults.data_folder(working_directory),
        chunk_id=chunk_id,
        seed=seed,
        **get_data_module_fn_args(config_space),
    )
    data_module.setup("val")
    val_exists = data_module.val_data() is not None
    assert (
        val_exists or not tune_hyperparameters
    ), "Provide a validation set to optimize hyperparameters."
    if val_exists:
        config_space["metric"] = metric
        config_space["mode"] = mode
        return metric, mode
    elif not val_exists and metric is not None:
        warnings.warn("No need to pass `metric`. Without a validation set, it is not used.")
    return "train_loss", "min"


def _create_scheduler(
    scheduler: Union[str, Type[TrialScheduler]],
    config_space: Dict[str, Any],
    metric: str,
    mode: defaults.SUPPORTED_TUNING_MODE_TYPE,
    max_epochs: int,
    seed: int,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    state_url: Optional[str] = None,
) -> TrialScheduler:
    scheduler_kwargs = scheduler_kwargs or {}
    if isinstance(scheduler, str):
        hyperband_scheduler_kwargs = {"max_t": max_epochs, "resource_attr": "epoch"}
        scheduler_classes = {
            "asha": {"scheduler": ASHA, "scheduler_kwargs": hyperband_scheduler_kwargs},
            "bo": {"scheduler": FIFOScheduler, "scheduler_kwargs": {"searcher": "bayesopt"}},
            "rush": {"scheduler": RUSHScheduler, "scheduler_kwargs": hyperband_scheduler_kwargs},
            "random": {"scheduler": FIFOScheduler, "scheduler_kwargs": {"searcher": "random"}},
        }
        assert (
            scheduler in scheduler_classes
        ), f"Unknown scheduler {scheduler}. Options: {list(scheduler_classes)}."
        scheduler_kwargs.update(scheduler_classes[scheduler]["scheduler_kwargs"])
        scheduler = scheduler_classes[scheduler]["scheduler"]
    if "transfer_learning_evaluations" in inspect.getfullargspec(scheduler.__init__).args:
        scheduler_kwargs["transfer_learning_evaluations"] = _load_tuning_history(
            state_url=state_url, config_space=config_space, metric=metric
        )
        if scheduler_kwargs["transfer_learning_evaluations"]:
            logger.info(
                f"Using information of {len(scheduler_kwargs['transfer_learning_evaluations'])} previous tuning "
                "jobs to accelerate this job."
            )
    return scheduler(
        config_space=config_space,
        mode=mode,
        metric=metric,
        random_seed=seed,
        **scheduler_kwargs,
    )


def _execute_tuning_job_locally(
    state_url: Optional[str],
    next_state_url: Optional[str],
    working_directory: Optional[str],
    config_file: str,
    mode: defaults.SUPPORTED_TUNING_MODE_TYPE,
    config_space: Dict[str, Any],
    metric: str,
    updater: str,
    max_epochs: int,
    task_id: str,
    chunk_id: int,
    max_time: float,
    max_num_trials_started: int,
    max_num_trials_completed: int,
    max_num_trials_finished: int,
    max_cost: float,
    n_workers: int,
    scheduler: Union[str, Type[TrialScheduler]],
    scheduler_kwargs: Dict[str, Any],
    seed: int,
    accelerator: str,
    devices: int,
):
    """Executes the tuning job locally.

    See renate.tuning.execute_tuning_job for a description of arguments.
    """
    tune_hyperparameters = is_syne_tune_config_space(config_space)
    config_space["updater"] = updater
    config_space["max_epochs"] = max_epochs
    config_space["config_file"] = config_file
    config_space["prepare_data"] = 0
    config_space["chunk_id"] = chunk_id
    config_space["task_id"] = task_id
    config_space["working_directory"] = working_directory
    config_space["seed"] = seed
    config_space["accelerator"] = accelerator
    config_space["devices"] = devices
    if state_url is not None:
        config_space["state_url"] = state_url

    metric, mode = _verify_validation_set_for_hpo_and_checkpointing(
        config_space=config_space,
        config_file=config_file,
        tune_hyperparameters=tune_hyperparameters,
        metric=metric,
        mode=mode,
        working_directory=working_directory,
        chunk_id=chunk_id,
        seed=seed,
    )

    training_script = str(Path(renate.__path__[0]) / "cli" / "run_training.py")
    assert Path(training_script).is_file(), f"Could not find training script {training_script}."
    logger.info("Start updating the model.")
    if tune_hyperparameters:
        logger.info(
            f"Tuning hyperparameters with respect to {metric} ({mode}) for {max_time} seconds on {n_workers} worker(s)."
        )
    backend = LocalBackend(entry_point=training_script)
    if scheduler is None or not tune_hyperparameters:
        if scheduler is not None:
            warnings.warn(
                "Configuration space contains exactly one configuration, custom scheduler is ignored."
            )
        scheduler = defaults.scheduler(config_space=config_space, mode=mode, metric=metric)
    else:
        scheduler = _create_scheduler(
            scheduler=scheduler,
            config_space=config_space,
            metric=metric,
            mode=mode,
            max_epochs=max_epochs,
            seed=seed,
            scheduler_kwargs=scheduler_kwargs,
            state_url=state_url,
        )
    logging_callback = (
        TuningLoggerCallback(mode=mode, metric=metric)
        if tune_hyperparameters
        else TrainingLoggerCallback()
    )
    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(
            max_wallclock_time=max_time,
            max_num_trials_started=max_num_trials_started,
            max_num_trials_completed=max_num_trials_completed,
            max_num_trials_finished=max_num_trials_finished,
            max_cost=max_cost,
        ),
        n_workers=n_workers,
        callbacks=[StoreResultsCallback(), logging_callback],
    )

    tuner.run()

    logger.info("All training is completed. Saving state...")

    _teardown_tuning_job(
        backend=backend,
        config_space=config_space,
        job_name=tuner.name,
        state_url=state_url,
        next_state_url=next_state_url,
    )

    logger.info("Renate update completed successfully.")

    return tuner


def _execute_tuning_job_remotely(
    state_url: Optional[str],
    next_state_url: Optional[str],
    working_directory: Optional[str],
    config_file: str,
    mode: str,
    config_space: Dict[str, Any],
    metric: str,
    updater: str,
    max_epochs: int,
    task_id: str,
    chunk_id: int,
    source_dir: str,
    requirements_file: str,
    role: str,
    instance_type: str,
    instance_count: int,
    instance_max_time: float,
    max_time: float,
    max_num_trials_started: int,
    max_num_trials_completed: int,
    max_num_trials_finished: int,
    max_cost: float,
    n_workers: int,
    scheduler: Union[str, Type[TrialScheduler]],
    scheduler_kwargs: Dict[str, Any],
    seed: int,
    accelerator: str,
    devices: int,
    job_name: str,
) -> None:
    """Executes the tuning job on SageMaker.

    See renate.tuning.execute_tuning_job for a description of arguments."""
    tuning_script = str(Path(renate.__path__[0]) / "cli" / "run_training_with_tuning.py")
    job_timestamp = defaults.current_timestamp()
    job_name = f"{job_name}-{job_timestamp}"
    tmp_dir = tempfile.mkdtemp()
    dependencies = _prepare_remote_tuning_job(
        tmp_dir=tmp_dir,
        requirements_file=requirements_file,
        state_url=state_url,
        next_state_url=next_state_url,
        working_directory=working_directory,
        config_file=config_file,
        mode=mode,
        config_space=config_space,
        metric=metric,
        updater=updater,
        max_epochs=max_epochs,
        task_id=task_id,
        chunk_id=chunk_id,
        max_time=max_time,
        max_num_trials_started=max_num_trials_started,
        max_num_trials_completed=max_num_trials_completed,
        max_num_trials_finished=max_num_trials_finished,
        max_cost=max_cost,
        n_workers=n_workers,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        seed=seed,
        accelerator=accelerator,
        devices=devices,
    )
    PyTorch(
        entry_point=tuning_script,
        source_dir=None if source_dir is None else str(source_dir),
        instance_type=instance_type,
        instance_count=instance_count,
        py_version=defaults.PYTHON_VERSION,
        framework_version=defaults.FRAMEWORK_VERSION,
        max_run=instance_max_time,
        role=role or get_execution_role(),
        dependencies=dependencies,
        hyperparameters={},
        volume_size=defaults.VOLUME_SIZE,
    ).fit(wait=False, job_name=job_name)
    shutil.rmtree(tmp_dir)
