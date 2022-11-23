# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from syne_tune.optimizer.scheduler import TrialScheduler

from renate import defaults
from renate.updaters.experimental.repeated_distill import RepeatedDistillationModelUpdater
from renate.updaters.experimental.er import (
    CLSExperienceReplayModelUpdater,
    DarkExperienceReplayModelUpdater,
    ExperienceReplayModelUpdater,
    PooledOutputDistillationExperienceReplayModelUpdater,
    SuperExperienceReplayModelUpdater,
)
from renate.updaters.experimental.gdumb import GDumbModelUpdater
from renate.updaters.experimental.joint import JointModelUpdater
from renate.updaters.experimental.offline_er import OfflineExperienceReplayModelUpdater
from renate.updaters.model_updater import ModelUpdater


def get_updater_and_learner_kwargs(
    args: argparse.Namespace,
) -> Tuple[Type[ModelUpdater], Dict[str, Any]]:
    """Returns the model updater class and the keyword arguments for the learner."""
    learner_args = [
        "optimizer",
        "learning_rate",
        "learning_rate_scheduler",
        "learning_rate_scheduler_step_size",
        "learning_rate_scheduler_gamma",
        "momentum",
        "weight_decay",
        "batch_size",
        "seed",
    ]
    base_er_args = learner_args + [
        "loss_weight",
        "ema_memory_update_gamma",
        "memory_size",
        "memory_batch_size",
        "loss_normalization",
    ]
    updater_class = None
    if args.updater == "ER":
        learner_args = base_er_args + ["alpha"]
        updater_class = ExperienceReplayModelUpdater
    elif args.updater == "DER":
        learner_args = base_er_args + ["alpha", "beta"]
        updater_class = DarkExperienceReplayModelUpdater
    elif args.updater == "POD-ER":
        learner_args = base_er_args + ["alpha", "distillation_type", "normalize"]
        updater_class = PooledOutputDistillationExperienceReplayModelUpdater
    elif args.updater == "CLS-ER":
        learner_args = base_er_args + [
            "alpha",
            "beta",
            "stable_model_update_weight",
            "plastic_model_update_weight",
            "stable_model_update_probability",
            "plastic_model_update_probability",
        ]
        updater_class = CLSExperienceReplayModelUpdater
    elif args.updater == "Super-ER":
        learner_args = base_er_args + [
            "der_alpha",
            "der_beta",
            "sp_shrink_factor",
            "sp_sigma",
            "cls_alpha",
            "cls_stable_model_update_weight",
            "cls_plastic_model_update_weight",
            "cls_stable_model_update_probability",
            "cls_plastic_model_update_probability",
            "pod_alpha",
            "pod_distillation_type",
            "pod_normalize",
        ]
        updater_class = SuperExperienceReplayModelUpdater
    elif args.updater == "OfflineER":
        learner_args = learner_args + ["loss_weight_new_data", "memory_size", "memory_batch_size"]
        updater_class = OfflineExperienceReplayModelUpdater
    elif args.updater == "RD":
        learner_args = learner_args + ["memory_size"]
        updater_class = RepeatedDistillationModelUpdater
    elif args.updater == "GDumb":
        learner_args = learner_args + ["memory_size"]
        updater_class = GDumbModelUpdater
    elif args.updater == "Joint":
        learner_args = learner_args
        updater_class = JointModelUpdater
    if updater_class is None:
        raise ValueError(f"Unknown learner {args.updater}.")
    learner_kwargs = {arg: value for arg, value in vars(args).items() if arg in learner_args}
    return updater_class, learner_kwargs


def parse_hyperparameters(parser) -> None:
    """Adds arguments for the specified updater."""
    updater: Optional[str] = None
    for i, arg in enumerate(sys.argv):
        if arg == "--updater" and len(sys.argv) > i:
            updater = sys.argv[i + 1]
            break
    if updater is None:
        return

    assert updater in parse_by_updater, f"Unknown updater {updater}."
    parse_by_updater[updater](parser)
    parse_optimizer_arguments(parser)


def parse_unknown_args(unknown_args_list: List[str]) -> Dict[str, str]:
    """Parses arguments provided using the command line which are not in the standard list of args.

    For example these args can be used to pass hyperparameters to the model function.
    """
    args: Dict[str, str] = {}

    if len(unknown_args_list) % 2 != 0:
        raise ValueError(
            "Error: unable to parse the additional arguments. "
            "Please make sure arguments are specified in the format: --arg_name value. "
            f"The following list of arguments was received: {unknown_args_list}."
        )

    for i in range(0, len(unknown_args_list), 2):
        if not unknown_args_list[i].startswith("--"):
            raise ValueError(
                "Please make sure arguments are specified in the format: --arg_name value. "
                f"Failing to parse: {unknown_args_list[i]} due to the missing '--'."
            )
        arg_name = unknown_args_list[i][2:]
        arg_val = unknown_args_list[i + 1]
        args[arg_name] = arg_val

    return args


def parse_optimizer_arguments(parser: argparse.Namespace) -> None:
    """A helper function that adds optimizer arguments."""
    parser.add_argument(
        "--optimizer",
        type=str,
        default=defaults.OPTIMIZER,
        help=f"Optimizer used for training. Options: SGD or Adam. Default: {defaults.OPTIMIZER}.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=defaults.LEARNING_RATE,
        help=f"Learning rate used during model update. Default: {defaults.LEARNING_RATE}.",
    )
    parser.add_argument(
        "--learning_rate_scheduler",
        type=str,
        default=defaults.LEARNING_RATE_SCHEDULER,
        help=f"Learning rate scheduler used during model update. Default: {defaults.LEARNING_RATE_SCHEDULER}.",
    )
    parser.add_argument(
        "--learning_rate_scheduler_step_size",
        type=int,
        default=defaults.LEARNING_RATE_SCHEDULER_STEP_SIZE,
        help=f"Step size for learning rate scheduler. Default: {defaults.LEARNING_RATE_SCHEDULER_STEP_SIZE}.",
    )
    parser.add_argument(
        "--learning_rate_scheduler_gamma",
        type=float,
        default=defaults.LEARNING_RATE_SCHEDULER_GAMMA,
        help=f"Gamma for learning rate scheduler. Default: {defaults.LEARNING_RATE_SCHEDULER_GAMMA}.",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=defaults.MOMENTUM,
        help=f"Momentum used during model update. Default: {defaults.MOMENTUM}.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=defaults.WEIGHT_DECAY,
        help=f"Weight decay used during model update. Default: {defaults.WEIGHT_DECAY}.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=defaults.BATCH_SIZE,
        help=f"Batch size used during model update for the new data. Default: {defaults.BATCH_SIZE}.",
    )

    parser.add_argument(
        "--loss_weight",
        type=float,
        default=defaults.LOSS_WEIGHT,
        help=f"Loss weight used during model update. Default: {defaults.LOSS_WEIGHT}.",
    )


def parse_replay_learner_arguments(parser: argparse.Namespace) -> None:
    """A helper function that adds Replay Learner arguments."""
    parser.add_argument(
        "--memory_size",
        type=int,
        default=defaults.MEMORY_SIZE,
        help=f"Memory size available for the memory buffer. Default: {defaults.MEMORY_SIZE}.",
    )
    parser.add_argument(
        "--memory_batch_size",
        type=int,
        default=defaults.BATCH_SIZE,
        help=f"Batch size used during model update for the memory buffer. Default: {defaults.BATCH_SIZE}.",
    )


def _parse_base_experience_replay_arguments(parser: argparse.Namespace) -> None:
    """A helper function that adds Base Experience Replay arguments."""
    parser.add_argument(
        "--ema_memory_update_gamma",
        type=float,
        default=defaults.EMA_MEMORY_UPDATE_GAMMA,
        help=f"Exponential moving average factor to update logits. Default: {defaults.EMA_MEMORY_UPDATE_GAMMA}.",
    )
    parser.add_argument(
        "--loss_normalization",
        type=int,
        choices=[0, 1],
        default=defaults.LOSS_NORMALIZATION,
        help="Whether to normalize the loss with respect to the loss weights. "
        f"Default: {bool(defaults.LOSS_NORMALIZATION)}.",
    )
    parse_replay_learner_arguments(parser)


def parse_gdumb_arguments(parser: argparse.Namespace) -> None:
    """A helper function that adds GDumb arguments."""
    parse_replay_learner_arguments(parser)


def parse_joint_arguments(parser: argparse.Namespace) -> None:
    """A helper function that adds Joint Learner arguments."""
    pass


def parse_experience_replay_arguments(parser: argparse.Namespace) -> None:
    """A helper function that adds Experience Replay arguments."""
    parser.add_argument(
        "--alpha",
        type=float,
        default=defaults.ER_ALPHA,
        help=f"Weight for the loss of the buffer data. Default: {defaults.ER_ALPHA}.",
    )
    _parse_base_experience_replay_arguments(parser)


def parse_dark_experience_replay_arguments(parser: argparse.Namespace) -> None:
    """A helper function that adds Dark Experience Replay arguments."""
    parser.add_argument(
        "--alpha",
        type=float,
        default=defaults.DER_ALPHA,
        help=f"Weight for logit regularization term. Default: {defaults.DER_ALPHA}.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=defaults.DER_ALPHA,
        help=f"Weight for memory loss term. Default: {defaults.DER_BETA}.",
    )
    _parse_base_experience_replay_arguments(parser)


def parse_pod_experience_replay_arguments(parser: argparse.Namespace) -> None:
    """A helper function that adds PODNet Experience Replay arguments."""
    parser.add_argument(
        "--alpha",
        type=float,
        default=defaults.POD_ALPHA,
        help=f"Weight for intermediate representation regularization term. Default: {defaults.POD_ALPHA}.",
    )
    parser.add_argument(
        "--distillation_type",
        type=str,
        default=defaults.POD_DISTILLATION_TYPE,
        help="Distillation type to apply with respect to the intermediate representation. "
        f"Default: {defaults.POD_DISTILLATION_TYPE}.",
    )
    parser.add_argument(
        "--normalize",
        type=int,
        default=defaults.POD_NORMALIZE,
        help="Whether to normalize both the current and cached features before computing the Frobenius norm. "
        f"Default: {defaults.POD_NORMALIZE}.",
    )
    _parse_base_experience_replay_arguments(parser)


def parse_cls_experience_replay_arguments(parser: argparse.Namespace) -> None:
    """A helper function that adds CLS Experience Replay arguments."""
    parser.add_argument(
        "--alpha",
        type=float,
        default=defaults.CLS_ALPHA,
        help=f"Weight for the cross-entropy loss term. Default: {defaults.CLS_ALPHA}.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=defaults.CLS_BETA,
        help=f"Weight for the consistency memory loss term. Default: {defaults.CLS_BETA}.",
    )
    parser.add_argument(
        "--stable_model_update_weight",
        type=float,
        default=defaults.CLS_STABLE_MODEL_UPDATE_WEIGHT,
        help="The starting weight for the exponential moving average to update the stable model. "
        f"Default: {defaults.CLS_STABLE_MODEL_UPDATE_WEIGHT}.",
    )
    parser.add_argument(
        "--plastic_model_update_weight",
        type=float,
        default=defaults.CLS_PLASTIC_MODEL_UPDATE_WEIGHT,
        help="The starting weight for the exponential moving average to update the plastic model. "
        f"Default: {defaults.CLS_PLASTIC_MODEL_UPDATE_WEIGHT}.",
    )
    parser.add_argument(
        "--stable_model_update_probability",
        type=float,
        default=defaults.CLS_STABLE_MODEL_UPDATE_PROBABILITY,
        help=f"Probability to update the stable model. Default: {defaults.CLS_STABLE_MODEL_UPDATE_PROBABILITY}.",
    )
    parser.add_argument(
        "--plastic_model_update_probability",
        type=float,
        default=defaults.CLS_PLASTIC_MODEL_UPDATE_PROBABILITY,
        help=f"Probability to update the plastic model. Default: {defaults.CLS_PLASTIC_MODEL_UPDATE_PROBABILITY}.",
    )
    _parse_base_experience_replay_arguments(parser)


def parse_super_experience_replay_arguments(parser: argparse.Namespace) -> None:
    """A helper function that adds Super Experience Replay arguments."""
    parser.add_argument(
        "--der_alpha",
        type=float,
        default=defaults.SER_DER_ALPHA,
        help=f"Weight for logit regularization term. Default: {defaults.SER_DER_ALPHA}.",
    )
    parser.add_argument(
        "--der_beta",
        type=float,
        default=defaults.SER_DER_ALPHA,
        help=f"Weight for memory loss term. Default: {defaults.SER_DER_BETA}.",
    )
    parser.add_argument(
        "--sp_shrink_factor",
        type=float,
        default=defaults.SER_SP_SHRINK_FACTOR,
        help=f"Weight for logit regularization term. Default: {defaults.SER_SP_SHRINK_FACTOR}.",
    )
    parser.add_argument(
        "--sp_sigma",
        type=float,
        default=defaults.SER_SP_SIGMA,
        help=f"Weight for memory loss term. Default: {defaults.SER_SP_SIGMA}.",
    )
    parser.add_argument(
        "--pod_alpha",
        type=float,
        default=defaults.SER_POD_ALPHA,
        help=f"Weight for intermediate representation regularization term. Default: {defaults.SER_POD_ALPHA}.",
    )
    parser.add_argument(
        "--pod_distillation_type",
        type=str,
        default=defaults.SER_POD_DISTILLATION_TYPE,
        help="Distillation type to apply with respect to the intermediate representation. "
        f"Default: {defaults.SER_POD_DISTILLATION_TYPE}.",
    )
    parser.add_argument(
        "--pod_normalize",
        type=int,
        default=defaults.SER_POD_NORMALIZE,
        help="Whether to normalize both the current and cached features before computing the Frobenius norm. "
        f"Default: {defaults.SER_POD_NORMALIZE}.",
    )
    parser.add_argument(
        "--cls_alpha",
        type=float,
        default=defaults.SER_CLS_ALPHA,
        help=f"Weight for the consistency loss term. Default: {defaults.SER_CLS_ALPHA}.",
    )
    parser.add_argument(
        "--cls_stable_model_update_weight",
        type=float,
        default=defaults.SER_CLS_STABLE_MODEL_UPDATE_WEIGHT,
        help="The starting weight for the exponential moving average to update the stable model. "
        f"Default: {defaults.SER_CLS_STABLE_MODEL_UPDATE_WEIGHT}.",
    )
    parser.add_argument(
        "--cls_plastic_model_update_weight",
        type=float,
        default=defaults.SER_CLS_PLASTIC_MODEL_UPDATE_WEIGHT,
        help="The starting weight for the exponential moving average to update the plastic model. "
        f"Default: {defaults.SER_CLS_PLASTIC_MODEL_UPDATE_WEIGHT}.",
    )
    parser.add_argument(
        "--cls_stable_model_update_probability",
        type=float,
        default=defaults.SER_CLS_STABLE_MODEL_UPDATE_PROBABILITY,
        help=f"Probability to update the stable model. Default: {defaults.SER_CLS_STABLE_MODEL_UPDATE_PROBABILITY}.",
    )
    parser.add_argument(
        "--cls_plastic_model_update_probability",
        type=float,
        default=defaults.SER_CLS_PLASTIC_MODEL_UPDATE_PROBABILITY,
        help=f"Probability to update the plastic model. Default: {defaults.SER_CLS_PLASTIC_MODEL_UPDATE_PROBABILITY}.",
    )
    _parse_base_experience_replay_arguments(parser)


def parse_rd_learner_arguments(parser: argparse.Namespace) -> None:
    """A helper function that adds Repeated Distill Learner arguments."""
    parser.add_argument(
        "--memory_size",
        type=int,
        default=defaults.MEMORY_SIZE,
        help=f"Memory size available for the memory buffer. Default: {defaults.MEMORY_SIZE}.",
    )


def _get_args_by_prefix(
    args: Union[argparse.Namespace, Dict[str, str]], prefix: str
) -> Dict[str, str]:
    """Returns a dictionary containing all key/value pairs from `args` whose arguments start with `prefix`."""
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    return {k: v for k, v in args.items() if k.startswith(prefix)}


def get_model_fn_args(args: Union[argparse.Namespace, Dict[str, str]]) -> Dict[str, str]:
    """Returns all arguments from `args` which should be passed to `model_fn`."""
    return _get_args_by_prefix(args, "model_fn_")


def get_data_module_fn_args(args: Union[argparse.Namespace, Dict[str, str]]) -> Dict[str, str]:
    """Returns all arguments from `args` which should be passed to `data_module_fn`."""
    return _get_args_by_prefix(args, "data_module_fn_")


def get_transform_args(args: Union[argparse.Namespace, Dict[str, str]]) -> Dict[str, str]:
    """Returns all arguments from `args` which should be passed to each `transform` function."""
    return _get_args_by_prefix(args, "transform_")


def get_transforms_kwargs(
    config_module: ModuleType, args: Union[argparse.Namespace, Dict[str, str]]
) -> Dict[str, Callable]:
    """Creates and returns data transforms kwargs for updater."""
    transform_fn_names = [
        "train_transform",
        "train_target_transform",
        "test_transform",
        "test_target_transform",
        "buffer_transform",
        "buffer_target_transform",
    ]
    transforms = {}
    for transform_fn_name in transform_fn_names:
        if transform_fn_name in vars(config_module):
            transforms[transform_fn_name] = getattr(config_module, transform_fn_name)(
                **get_transform_args(args)
            )
    return transforms


def get_scheduler_kwargs(
    config_module: ModuleType,
) -> Tuple[Optional[Type[TrialScheduler]], Optional[Dict[str, Any]]]:
    """Creates and returns scheduler type and kwargs for the scheduler."""
    scheduler_fn_name = "scheduler_fn"
    if scheduler_fn_name in vars(config_module):
        return getattr(config_module, scheduler_fn_name)()
    return None, None


parse_by_updater = {
    "ER": parse_experience_replay_arguments,
    "DER": parse_dark_experience_replay_arguments,
    "POD-ER": parse_pod_experience_replay_arguments,
    "CLS-ER": parse_cls_experience_replay_arguments,
    "Super-ER": parse_super_experience_replay_arguments,
    "GDumb": parse_gdumb_arguments,
    "Joint": parse_joint_arguments,
    "RD": parse_rd_learner_arguments,
    "OfflineER": parse_replay_learner_arguments,
}
