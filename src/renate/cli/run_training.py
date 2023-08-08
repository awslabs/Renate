# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import shutil
import sys
from pathlib import Path

from pytorch_lightning import seed_everything

from renate import defaults
from renate.cli.parsing_functions import (
    get_function_kwargs,
    get_transforms_dict,
    get_updater_and_learner_kwargs,
    parse_arguments,
)
from renate.utils.file import maybe_download_from_s3, move_to_uri
from renate.utils.module import (
    get_and_setup_data_module,
    get_learning_rate_scheduler,
    get_loss_fn,
    get_metrics,
    get_model,
    get_optimizer,
    import_module,
)
from renate.utils.optimizer import create_partial_optimizer
from renate.utils.syne_tune import redirect_to_tmp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelUpdaterCLI:
    """Entry point to update a model.

    Given a dataset, a model and a training configuration, this class will update the model with the
    given data.

    It will create following folder structure:
        args.working_dir
            data (filled by the DataModule)
            input_state_dir (content provided in args.state_url is copied to this location)
                learner.pkl
                model.pt
            output_state_dir (outcome of the model update)
                learner.pkl
                model.pt
    """

    def _copy_state_to_working_directory(
        self, input_state_url: str, input_state_folder: str
    ) -> None:
        """Copies current state into the working directory.

        If state is on s3, download directly from there to `input_state_folder`.
        If state is already in local directory, copy to right folder unless it is already in the
        right folder. If `input_state_folder` exists but `input_state_url` is not provided,
        `input_state_folder` will be removed.
        """
        local_dir = maybe_download_from_s3(input_state_url, input_state_folder)
        folder_downloaded_from_s3 = local_dir == input_state_folder
        if not folder_downloaded_from_s3:
            if local_dir != input_state_folder:
                shutil.rmtree(input_state_folder, ignore_errors=True)
                if local_dir is not None:
                    shutil.copytree(
                        local_dir,
                        input_state_folder,
                        ignore=shutil.ignore_patterns("*.sagemaker-uploading"),
                        dirs_exist_ok=True,
                    )

    def _prepare_data_state_model(self, args: argparse.Namespace) -> None:
        """Assigns locations for data, state and model."""
        self._data_folder = defaults.data_folder(args.working_directory)
        self._input_state_folder = defaults.input_state_folder(args.working_directory)
        working_directory = redirect_to_tmp(args.st_checkpoint_dir or args.working_directory)
        self._output_state_folder = defaults.output_state_folder(working_directory)
        self._current_model_file = defaults.model_file(self._input_state_folder)
        self._next_model_file = defaults.model_file(self._output_state_folder)
        self._copy_state_to_working_directory(args.input_state_url, self._input_state_folder)
        if not Path(self._input_state_folder).is_dir():
            self._input_state_folder = None
        if not Path(self._current_model_file).is_file():
            self._current_model_file = None

    def run(self):
        config_module = None
        for i, arg_value in enumerate(sys.argv):
            if arg_value == "--config_file":
                config_module = import_module("config_module", sys.argv[i + 1])
                break
        if config_module is None:
            raise RuntimeError("The following argument is required: --config_file")
        args, function_args = parse_arguments(
            config_module=config_module,
            function_names=[
                "model_fn",
                "data_module_fn",
                "train_transform",
                "test_transform",
                "buffer_transform",
                "scheduler_fn",
                "loss_fn",
                "optimizer_fn",
                "lr_scheduler_fn",
                "metrics_fn",
            ],
            ignore_args=["data_path", "model_state_url"],
        )

        seed_everything(args.seed, True)
        self._prepare_data_state_model(args)

        data_module = get_and_setup_data_module(
            config_module,
            data_path=self._data_folder,
            prepare_data=args.prepare_data,
            **get_function_kwargs(args=args, function_args=function_args["data_module_fn"]),
        )
        model = get_model(
            config_module,
            model_state_url=None if getattr(args, "reset", False) else self._current_model_file,
            **get_function_kwargs(args=args, function_args=function_args["model_fn"]),
        )
        loss_fn = get_loss_fn(
            config_module,
            not args.updater.startswith("Avalanche-"),
            **get_function_kwargs(args=args, function_args=function_args["loss_fn"]),
        )
        partial_optimizer = get_optimizer(
            config_module,
            **get_function_kwargs(args=args, function_args=function_args["optimizer_fn"]),
        )
        if partial_optimizer is None:
            partial_optimizer = create_partial_optimizer(
                optimizer=args.optimizer,
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        lr_scheduler_config = get_learning_rate_scheduler(
            config_module,
            **get_function_kwargs(args=args, function_args=function_args["lr_scheduler_fn"]),
        )
        lr_scheduler_kwargs = {}
        if lr_scheduler_config is not None:
            lr_scheduler_kwargs["learning_rate_scheduler"] = lr_scheduler_config[0]
            lr_scheduler_kwargs["learning_rate_scheduler_interval"] = lr_scheduler_config[1]
        metrics = get_metrics(
            config_module,
            **get_function_kwargs(args=args, function_args=function_args["metrics_fn"]),
        )

        model_updater_class, learner_kwargs = get_updater_and_learner_kwargs(args)

        model_updater = model_updater_class(
            model=model,
            optimizer=partial_optimizer,
            input_state_folder=self._input_state_folder,
            output_state_folder=self._output_state_folder,
            max_epochs=args.max_epochs,
            metric=args.metric,
            mode=args.mode,
            logged_metrics=metrics,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            strategy=args.strategy,
            gradient_clip_algorithm=args.gradient_clip_algorithm,
            gradient_clip_val=args.gradient_clip_val,
            early_stopping_enabled=args.early_stopping,
            deterministic_trainer=args.deterministic_trainer,
            loss_fn=loss_fn,
            **learner_kwargs,
            **lr_scheduler_kwargs,
            **get_transforms_dict(config_module, args, function_args),
        )

        model_updater.update(
            train_dataset=data_module.train_data(),
            val_dataset=data_module.val_data(),
            train_dataset_collate_fn=data_module.train_collate_fn(),
            val_dataset_collate_fn=data_module.val_collate_fn(),
            task_id=args.task_id,
        )

        if args.output_state_url is not None:
            move_to_uri(self._output_state_folder, args.output_state_url)


if __name__ == "__main__":
    ModelUpdaterCLI().run()
