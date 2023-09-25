# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from pytorch_lightning import seed_everything

import renate
import renate.defaults as defaults
from renate.cli.parsing_functions import (
    get_data_module_fn_kwargs,
    get_metrics_fn_kwargs,
    get_model_fn_kwargs,
    get_scheduler_kwargs,
    get_transforms_kwargs,
)
from renate.evaluation.metrics.classification import (
    average_accuracy,
    backward_transfer,
    forgetting,
    forward_transfer,
    micro_average_accuracy,
)
from renate.training import run_training_job
from renate.training.training import submit_remote_job
from renate.utils.file import (
    copy_to_uri,
    is_s3_uri,
    move_to_uri,
    save_pandas_df_to_csv,
)
from renate.utils.module import (
    evaluate_and_record_results,
    get_and_prepare_data_module,
    get_metrics,
    get_model,
    import_module,
)

logger = logging.getLogger(__name__)


def experiment_config_file():
    return str(Path(renate.__path__[0]) / "benchmark" / "experiment_config.py")


def create_cumulative_metrics() -> List[Tuple[str, Callable]]:
    """Gets the cumulative metrics for a given task along with a name of the metric to include in
    any potential results table.
    """
    return [
        ("Average Accuracy", average_accuracy),
        ("Micro Average Accuracy", micro_average_accuracy),
        ("Forgetting", forgetting),
        ("Forward Transfer", forward_transfer),
        ("Backward Transfer", backward_transfer),
    ]


def cumulative_metrics_summary(
    results: Dict[str, List[List[float]]],
    cumulative_metrics: List[Tuple[str, Callable]],
    num_tasks: int,
    num_instances: List[int],
) -> pd.DataFrame:
    """Creates a pandas DataFrame summary with respect to the observed tasks, specified by
    `num_tasks`.

    Args:
        results: The results dictionary holding all the results with respect to all recorded
            metrics.
        cumulative_metrics: The list of (name, metric) tuples.
        num_tasks: The total number of tasks.
        num_instances: Count of test data points for each task.
    """
    data = []
    for task_id in range(num_tasks):
        row = [task_id + 1]
        for _, metric in cumulative_metrics:
            row.append(metric(results, task_id, num_instances))
        data.append(row)

    column_names = ["Task ID"] + [name for name, _ in cumulative_metrics]
    df = pd.DataFrame(data, columns=column_names)
    return df


def individual_metrics_summary(
    results: Dict[str, List[List[float]]],
    current_task: int,
    num_tasks: int,
) -> pd.DataFrame:
    """Creates a pandas DataFrame summary for all individual metrics with respect to all observed
    tasks.

    Args:
        results: The results dictionary holding all the results with respect to all recorded
            metrics.
        current_task: The current task ID.
        num_tasks: The total number of tasks.
    """
    data = []
    metric_columns = [k for k in results.keys() if "_init" not in k]

    for task_id in range(current_task):
        row = [task_id + 1]
        for key in metric_columns:
            value = results[key]
            for v in value[task_id]:
                row.append(v)
        data.append(row)

    sub_columns = [f"Task {i}" for i in range(1, num_tasks + 1)]
    mux = pd.MultiIndex.from_product([metric_columns, sub_columns])
    mux = mux.insert(0, "Task ID")
    df = pd.DataFrame(data, columns=mux)
    return df


def execute_experiment_job(
    backend: defaults.SUPPORTED_BACKEND_TYPE,
    config_file: str,
    config_space: Dict[str, Any],
    experiment_outputs_url: str,
    mode: defaults.SUPPORTED_TUNING_MODE_TYPE,
    metric: str,
    num_updates: int,
    working_directory: Optional[str] = defaults.WORKING_DIRECTORY,
    requirements_file: Optional[str] = None,
    dependencies: Optional[List[str]] = None,
    role: Optional[str] = None,
    instance_type: str = defaults.INSTANCE_TYPE,
    instance_count: int = defaults.INSTANCE_COUNT,
    instance_max_time: float = defaults.INSTANCE_MAX_TIME,
    max_time: Optional[float] = None,
    max_num_trials_started: Optional[int] = None,
    max_num_trials_completed: Optional[int] = None,
    max_num_trials_finished: Optional[int] = None,
    n_workers: int = defaults.N_WORKERS,
    seed: int = defaults.SEED,
    accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
    devices: int = defaults.DEVICES,
    deterministic_trainer: bool = True,
    gradient_clip_val: Optional[float] = defaults.GRADIENT_CLIP_VAL,
    gradient_clip_algorithm: Optional[str] = defaults.GRADIENT_CLIP_ALGORITHM,
    job_name: str = defaults.JOB_NAME,
    strategy: str = defaults.DISTRIBUTED_STRATEGY,
    precision: str = defaults.PRECISION,
    save_state: bool = defaults.SAVE_BENCHMARK_STATE,
) -> None:
    """Executes the experiment job.

    Args:
        backend: Backend of the experiment job.
        config_file: Path to the Renate config file.
        config_space: Details for defining your own search space is provided in the
            `Syne Tune Documentation
            <https://github.com/awslabs/syne-tune/blob/main/docs/search_space.md>`_.
        experiment_outputs_url: Path to the experiment outputs.
        mode: Whether to minimize or maximize the metric.
        metric: Metric of the experiment job.
        num_updates: Number of updates of the experiment job.
        working_directory: Path to the working directory.
        requirements_file: Path to the requirements file.
        dependencies: (SageMaker backend only) List of strings containing absolute or relative paths
            to files and directories that will be uploaded as part of the SageMaker training job.
        role: Role of the experiment job.
        instance_type: Instance type of the experiment job.
        instance_count: Instance count of the experiment job.
        instance_max_time: Instance max time of the experiment job.
        max_time: Max time of the experiment job.
        max_num_trials_started: Max number of trials started of the experiment job.
        max_num_trials_completed: Max number of trials completed of the experiment job.
        max_num_trials_finished: Max number of trials finished of the experiment job.
        n_workers: Number of workers of the experiment job.
        seed: Seed of the experiment job.
        accelerator: Type of accelerator to use.
        devices: Number of devices to use.
        deterministic_trainer: When true the Trainer adopts a deterministic behaviour also on GPU.
            In this function this parameter is set to True by default.
        gradient_clip_val: The value at which to clip gradients. Passing None disables it.
            `More details <https://lightning.ai/docs/pytorch/stable/common/trainer.html#init>`__
        gradient_clip_algorithm: The gradient clipping algorithm to use. Can be norm or value.
            `More details <https://lightning.ai/docs/pytorch/stable/common/trainer.html#init>`__
        job_name: Name of the experiment job.
        strategy: Name of the distributed training strategy to use.
            `More details <https://lightning.ai/docs/pytorch/stable/extensions/strategy.html>`__
        precision: Type of bit precision to use.
            `More details <https://lightning.ai/docs/pytorch/stable/common/precision_basic.html>`__
        save_state: Flag to retain models and buffer states of each update step. Disable to save
            storage.
    """
    assert (
        mode in defaults.SUPPORTED_TUNING_MODE
    ), f"Mode {mode} is not in {defaults.SUPPORTED_TUNING_MODE}."
    assert (
        backend in defaults.SUPPORTED_BACKEND
    ), f"Backend {backend} is not in {defaults.SUPPORTED_BACKEND}."
    if backend == "local":
        return _execute_experiment_job_locally(
            config_file=config_file,
            experiment_outputs_url=experiment_outputs_url,
            mode=mode,
            config_space=config_space,
            metric=metric,
            working_directory=working_directory,
            num_updates=num_updates,
            max_time=max_time,
            max_num_trials_started=max_num_trials_started,
            max_num_trials_completed=max_num_trials_completed,
            max_num_trials_finished=max_num_trials_finished,
            n_workers=n_workers,
            accelerator=accelerator,
            devices=devices,
            deterministic_trainer=deterministic_trainer,
            seed=seed,
            strategy=strategy,
            precision=precision,
            save_state=save_state,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
    _execute_experiment_job_remotely(
        job_name=job_name,
        config_file=config_file,
        experiment_outputs_url=experiment_outputs_url,
        mode=mode,
        metric=metric,
        num_updates=num_updates,
        working_directory=working_directory,
        dependencies=dependencies or [],
        config_space=config_space,
        max_time=max_time,
        max_num_trials_started=max_num_trials_started,
        max_num_trials_completed=max_num_trials_completed,
        max_num_trials_finished=max_num_trials_finished,
        n_workers=n_workers,
        accelerator=accelerator,
        devices=devices,
        deterministic_trainer=deterministic_trainer,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        seed=seed,
        requirements_file=requirements_file,
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        instance_max_time=instance_max_time,
        strategy=strategy,
        precision=precision,
        save_state=save_state,
    )


def _execute_experiment_job_locally(
    config_file: str,
    experiment_outputs_url: str,
    num_updates: int,
    mode: defaults.SUPPORTED_TUNING_MODE_TYPE,
    config_space: Dict[str, Any],
    metric: str,
    working_directory: str,
    seed: int,
    accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE,
    devices: int,
    max_time: float,
    max_num_trials_started: int,
    max_num_trials_completed: int,
    max_num_trials_finished: int,
    n_workers: int,
    deterministic_trainer: bool,
    strategy: str,
    precision: str,
    save_state: bool,
    gradient_clip_val: Optional[float],
    gradient_clip_algorithm: Optional[str],
) -> None:
    """Runs an experiment, combining hyperparameter tuning and model for multiple updates.

    See renate.benchmark.experimentation.execute_experiment_job for more details.
    """
    logger.info("Start experiment.")
    seed_everything(seed, True)

    input_state_url = defaults.input_state_folder(working_directory)
    output_state_url = defaults.output_state_folder(working_directory)
    data_url = defaults.data_folder(working_directory)
    model_url = defaults.model_file(input_state_url)
    logs_url = defaults.logs_folder(working_directory)

    for url in [input_state_url, output_state_url, logs_url]:
        if os.path.exists(url):
            shutil.rmtree(url)
        Path(url).mkdir(parents=True, exist_ok=True)

    config_module = import_module("config_module", config_file)
    scheduler, scheduler_kwargs = get_scheduler_kwargs(config_module)
    model_fn_kwargs = get_model_fn_kwargs(config_module, config_space)
    logger.info(f"Loading model {model_fn_kwargs.get('model_name', '')}")
    model = get_model(config_module, **model_fn_kwargs)
    data_module_fn_kwargs = get_data_module_fn_kwargs(config_module, config_space)
    logger.info(f"Prepare dataset {data_module_fn_kwargs.get('dataset_name', '')}")
    data_module = get_and_prepare_data_module(
        config_module,
        data_path=data_url,
        chunk_id=0,
        seed=seed,
        **data_module_fn_kwargs,
    )
    data_module.setup()
    assert num_updates == len(
        data_module.test_data()
    ), f"The dataset has {len(data_module.test_data())} chunks, expected {num_updates}."
    num_instances = [len(data_chunk) for data_chunk in data_module.test_data()]
    transforms = get_transforms_kwargs(config_module, config_space)
    metrics_fn_kwargs = get_metrics_fn_kwargs(config_module, config_space)
    metrics = get_metrics(config_module, **metrics_fn_kwargs)

    torch.save(
        model.state_dict(),
        model_url,
    )

    # TODO: evaluate's trainer has to use devices=1:
    # See https://github.com/Lightning-AI/lightning/issues/2537
    # The fix is to launch evaluation in a separate process like training.
    results = evaluate_and_record_results(
        {},
        model=model,
        data_module=data_module,
        transform=transforms.get("test_transform"),
        target_transform=transforms.get("target_test_transform"),
        logged_metrics=metrics,
        metric_postfix="_init",
        accelerator=accelerator,
        devices=1,
        strategy=strategy,
        precision=precision,
    )

    for update_id in range(num_updates):
        logger.info(f"Starting Update {update_id + 1}/{num_updates}.")
        update_url = os.path.join(experiment_outputs_url, f"update_{update_id}")
        run_training_job(
            mode=mode,
            config_space=config_space,
            metric=metric,
            backend="local",
            updater=config_space["updater"],
            max_epochs=config_space["max_epochs"],
            chunk_id=update_id,
            input_state_url=input_state_url,
            output_state_url=output_state_url,
            working_directory=working_directory,
            config_file=config_file,
            max_time=max_time,
            max_num_trials_started=max_num_trials_started,
            max_num_trials_completed=max_num_trials_completed,
            max_num_trials_finished=max_num_trials_finished,
            n_workers=n_workers,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            seed=seed,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            strategy=strategy,
            deterministic_trainer=deterministic_trainer,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
        )
        move_to_uri(output_state_url, input_state_url)
        if save_state:
            copy_to_uri(input_state_url, update_url)
        model = get_model(
            config_module,
            model_state_url=model_url,
            **get_model_fn_kwargs(config_module, config_space),
        )

        results = evaluate_and_record_results(
            results,
            model=model,
            data_module=data_module,
            transform=transforms.get("test_transform"),
            target_transform=transforms.get("target_test_transform"),
            logged_metrics=metrics,
            accelerator=accelerator,
            devices=1,
        )
        df = individual_metrics_summary(results, update_id + 1, num_updates)
        save_pandas_df_to_csv(
            df, defaults.metric_summary_file(logs_url, special_str=f"_update_{update_id}")
        )
        logger.info(f"### Results after update {update_id + 1}: ###")
        logger.info(df)

    cumulative_metrics = create_cumulative_metrics()
    df = cumulative_metrics_summary(results, cumulative_metrics, num_updates, num_instances)
    save_pandas_df_to_csv(df, defaults.metric_summary_file(logs_url))
    logger.info("### Cumulative results: ###")
    logger.info(df)

    if not save_state:
        move_to_uri(defaults.hpo_file(input_state_url), str(experiment_outputs_url))
    move_to_uri(logs_url, defaults.logs_folder(experiment_outputs_url))

    shutil.rmtree(working_directory)
    logger.info("Experiment completed successfully.")


def _execute_experiment_job_remotely(experiment_outputs_url: str, **job_kwargs: Any) -> str:
    """Executes the experiment job on SageMaker.

    See renate.benchmark.experimentation.execute_experiment_job for more details.
    """
    assert is_s3_uri(
        experiment_outputs_url
    ), f"experiment_outputs_url {experiment_outputs_url} is not on S3."
    return submit_remote_job(
        experiment_outputs_url=experiment_outputs_url,
        optional_dependencies="benchmark",
        **job_kwargs,
    )
