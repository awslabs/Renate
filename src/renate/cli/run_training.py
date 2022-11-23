# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import shutil
from pathlib import Path

from pytorch_lightning import seed_everything

from renate import defaults
from renate.cli.parsing_functions import (
    get_data_module_fn_args,
    get_model_fn_args,
    get_transforms_kwargs,
    get_updater_and_learner_kwargs,
    parse_by_updater,
    parse_hyperparameters,
    parse_unknown_args,
)
from renate.utils.file import maybe_download_from_s3, move_to_uri
from renate.utils.module import get_and_setup_data_module, get_metrics, get_model, import_module
from renate.utils.syne_tune import redirect_to_tmp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelUpdaterCLI:
    """Entry point to update a model.

    Given a dataset, a model and a training configuration, this class will update the model with the given data.

    It will create following folder structure:
        args.working_dir
            data (filled by the DataModule)
            state_dir (content provided in args.state_url is copied to this location)
                learner.pkl
                model.pt
            next_state_dir (outcome of the model update)
                learner.pkl
                model.pt

    All the arguments whose names start with 'model_fn_' or 'data_module_fn_' will be passed to the
    model function (`model_fn`) and the data module function (`data_module_fn`) respectively.
    """

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()

        argument_group = parser.add_argument_group("Required Parameters")
        argument_group.add_argument(
            "--updater",
            type=str,
            required=True,
            choices=list(parse_by_updater),
            help="Select the type of model update strategy.",
        )
        argument_group.add_argument(
            "--config_file",
            type=str,
            required=True,
            help="Location of python file containing model_fn and data_module_fn.",
        )

        argument_group = parser.add_argument_group("Renate State")
        argument_group.add_argument(
            "--state_url",
            type=str,
            help="Location of previous Renate state (if available).",
        )
        argument_group.add_argument(
            "--next_state_url",
            type=str,
            help="Location where to store the next Renate state.",
        )

        argument_group = parser.add_argument_group("Hyperparameters")
        parse_hyperparameters(argument_group)

        argument_group = parser.add_argument_group("Optional Arguments")
        argument_group.add_argument(
            "--max_epochs",
            type=int,
            default=defaults.MAX_EPOCHS,
            help=f"Number of epochs trained at most. Default: {defaults.MAX_EPOCHS}",
        )
        argument_group.add_argument(
            "--task_id",
            type=str,
            default=defaults.TASK_ID,
            help="Task ID matching the current dataset. If you do not distinguish between different tasks, ignore this"
            f" argument. Default: {defaults.TASK_ID}.",
        )
        argument_group.add_argument(
            "--chunk_id",
            type=int,
            default=defaults.CHUNK_ID,
            help=f"The chunk ID specifying which data chunk to load in case of a benchmark"
            f"scenario. Default: {defaults.CHUNK_ID}.",
        )
        argument_group.add_argument(
            "--metric",
            type=str,
            help="Metric monitored during training to save checkpoints.",
        )
        argument_group.add_argument(
            "--mode",
            choices=defaults.SUPPORTED_TUNING_MODE,
            default="min",
            help="Indicate whether a smaller `metric` is better (`min`) or a larger (`max`).",
        )
        argument_group.add_argument(
            "--working_directory",
            type=str,
            default=defaults.WORKING_DIRECTORY,
            help=f"Folder used by Renate to store files temporarily. Default: {defaults.WORKING_DIRECTORY}.",
        )
        argument_group.add_argument(
            "--seed",
            type=int,
            default=defaults.SEED,
            help=f"Seed used for this job. Default: {defaults.SEED}.",
        )
        argument_group.add_argument(
            "--accelerator",
            type=str,
            default=defaults.ACCELERATOR,
            help=f"Accelerator used for this job. Default: {defaults.ACCELERATOR}.",
        )
        argument_group.add_argument(
            "--devices",
            type=int,
            default=defaults.DEVICES,
            help=f"Devices used for this job. Default: {defaults.DEVICES} device.",
        )
        argument_group.add_argument(
            "--early_stopping",
            type=str,
            default=str(defaults.EARLY_STOPPING),
            choices=["True", "False"],
            help=f"Enables the early stopping of the optimization. Default: {defaults.EARLY_STOPPING}.",
        )

        argument_group = parser.add_argument_group("DO NOT CHANGE")
        argument_group.add_argument(
            "--prepare_data",
            type=int,
            default=1,
            help="Whether to call DataModule.prepare_data(). Default: 1.",
        )
        argument_group.add_argument(
            "--st_checkpoint_dir",
            type=str,
            help="Location for checkpoints.",
        )
        return parser

    def _copy_state_to_working_directory(self, state_url: str, current_state_folder: str) -> None:
        """Copies current state into the working directory.

        If state is on s3, download directly from there to current_state_folder.
        If state is already in local directory, copy to right folder unless it is already in the right folder.
        If current_state_folder exists but state_url is not provided, current_state_folder will be removed.
        """
        local_dir = maybe_download_from_s3(state_url, current_state_folder)
        folder_downloaded_from_s3 = local_dir == current_state_folder
        if not folder_downloaded_from_s3:
            if local_dir != current_state_folder:
                shutil.rmtree(current_state_folder, ignore_errors=True)
                if local_dir is not None:
                    shutil.copytree(
                        local_dir,
                        current_state_folder,
                        ignore=shutil.ignore_patterns("*.sagemaker-uploading"),
                        dirs_exist_ok=True,
                    )

    def _prepare_data_state_model(self, args: argparse.Namespace) -> None:
        """Assigns locations for data, state and model."""
        self._data_folder = defaults.data_folder(args.working_directory)
        self._current_state_folder = defaults.current_state_folder(args.working_directory)
        working_directory = redirect_to_tmp(args.st_checkpoint_dir or args.working_directory)
        self._next_state_folder = defaults.next_state_folder(working_directory)
        self._current_model_file = defaults.model_file(self._current_state_folder)
        self._next_model_file = defaults.model_file(self._next_state_folder)
        self._copy_state_to_working_directory(args.state_url, self._current_state_folder)
        if not Path(self._current_state_folder).is_dir():
            self._current_state_folder = None
        if not Path(self._current_model_file).is_file():
            self._current_model_file = None

    def run(self):
        parser = self._create_parser()
        known_args, unknown_args_list = parser.parse_known_args()
        additional_args = parse_unknown_args(unknown_args_list)

        args = argparse.Namespace(**vars(known_args), **additional_args)
        args.early_stopping = args.early_stopping == "True"

        seed_everything(args.seed)
        self._prepare_data_state_model(args)
        config_module = import_module("config_module", args.config_file)

        data_module = get_and_setup_data_module(
            config_module,
            data_path=self._data_folder,
            prepare_data=args.prepare_data,
            chunk_id=args.chunk_id,
            seed=args.seed,
            **get_data_module_fn_args(args),
        )

        model = get_model(
            config_module,
            model_state_url=self._current_model_file,
            **get_model_fn_args(args),
        )

        metrics = get_metrics(config_module)

        model_updater_class, learner_kwargs = get_updater_and_learner_kwargs(args)

        model_updater = model_updater_class(
            model=model,
            current_state_folder=self._current_state_folder,
            next_state_folder=self._next_state_folder,
            max_epochs=args.max_epochs,
            metric=args.metric,
            mode=args.mode,
            logged_metrics=metrics,
            accelerator=args.accelerator,
            devices=args.devices,
            early_stopping_enabled=bool(args.early_stopping),
            **learner_kwargs,
            **get_transforms_kwargs(config_module, args),
        )

        model_updater.update(
            train_dataset=data_module.train_data(),
            val_dataset=data_module.val_data(),
            task_id=args.task_id,
        )

        if args.next_state_url is not None:
            move_to_uri(self._next_state_folder, args.next_state_url)


if __name__ == "__main__":
    ModelUpdaterCLI().run()
