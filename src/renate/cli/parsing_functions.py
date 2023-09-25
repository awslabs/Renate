# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import ast
import inspect
import sys
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
from syne_tune.optimizer.scheduler import TrialScheduler

from renate import defaults
from renate.updaters.experimental.er import (
    CLSExperienceReplayModelUpdater,
    DarkExperienceReplayModelUpdater,
    ExperienceReplayModelUpdater,
    PooledOutputDistillationExperienceReplayModelUpdater,
    SuperExperienceReplayModelUpdater,
)
from renate.updaters.experimental.fine_tuning import FineTuningModelUpdater
from renate.updaters.experimental.gdumb import GDumbModelUpdater
from renate.updaters.experimental.joint import JointModelUpdater
from renate.updaters.experimental.l2p import (
    LearningToPromptModelUpdater,
    LearningToPromptReplayModelUpdater,
)
from renate.updaters.experimental.offline_er import OfflineExperienceReplayModelUpdater
from renate.updaters.experimental.repeated_distill import RepeatedDistillationModelUpdater
from renate.updaters.model_updater import ModelUpdater

REQUIRED_ARGS_GROUP = "Required Arguments"
RENATE_STATE_ARGS_GROUP = "Renate State"
HYPERPARAMETER_ARGS_GROUP = "Hyperparameters"
CUSTOM_ARGS_GROUP = "Custom Arguments"
OPTIONAL_ARGS_GROUP = "Optional Arguments"
DO_NOT_CHANGE_GROUP = "DO NOT CHANGE"
ARGS_GROUP_ORDER = [
    REQUIRED_ARGS_GROUP,
    RENATE_STATE_ARGS_GROUP,
    HYPERPARAMETER_ARGS_GROUP,
    CUSTOM_ARGS_GROUP,
    OPTIONAL_ARGS_GROUP,
    DO_NOT_CHANGE_GROUP,
]


def get_updater_and_learner_kwargs(
    args: argparse.Namespace,
) -> Tuple[Type[ModelUpdater], Dict[str, Any]]:
    """Returns the model updater class and the keyword arguments for the learner."""
    if args.updater.startswith("Avalanche-") and find_spec("avalanche", None) is None:
        raise ImportError("Avalanche is not installed. Please run `pip install Renate[avalanche]`.")
    learner_args = ["batch_size", "seed", "mask_unused_classes"]
    base_er_args = learner_args + [
        "loss_weight",
        "ema_memory_update_gamma",
        "memory_size",
        "batch_memory_frac",
        "loss_normalization",
    ]
    updater_class = None
    if args.updater == "ER":
        learner_args = base_er_args + ["alpha"]
        updater_class = ExperienceReplayModelUpdater
    elif args.updater == "LearningToPrompt":
        learner_args = learner_args + ["prompt_sim_loss_weight"]
        updater_class = LearningToPromptModelUpdater
    elif args.updater == "LearningToPromptReplay":
        learner_args = learner_args + ["prompt_sim_loss_weight", "memory_size", "memory_batch_size"]
        updater_class = LearningToPromptReplayModelUpdater
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
    elif args.updater == "Offline-ER":
        learner_args = learner_args + ["loss_weight_new_data", "memory_size", "batch_memory_frac"]
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
    elif args.updater == "FineTuning":
        learner_args = learner_args
        updater_class = FineTuningModelUpdater
    elif args.updater == "Avalanche-ER":
        learner_args = learner_args + ["memory_size", "batch_memory_frac"]
        from renate.updaters.avalanche.model_updater import ExperienceReplayAvalancheModelUpdater

        updater_class = ExperienceReplayAvalancheModelUpdater
    elif args.updater == "Avalanche-EWC":
        learner_args = learner_args + ["ewc_lambda"]
        from renate.updaters.avalanche.model_updater import ElasticWeightConsolidationModelUpdater

        updater_class = ElasticWeightConsolidationModelUpdater
    elif args.updater == "Avalanche-LwF":
        learner_args = learner_args + ["alpha", "temperature"]
        from renate.updaters.avalanche.model_updater import LearningWithoutForgettingModelUpdater

        updater_class = LearningWithoutForgettingModelUpdater
    elif args.updater == "Avalanche-iCaRL":
        learner_args = learner_args + ["memory_size", "batch_memory_frac"]
        from renate.updaters.avalanche.model_updater import ICaRLModelUpdater

        updater_class = ICaRLModelUpdater
    if updater_class is None:
        raise ValueError(f"Unknown learner {args.updater}.")
    learner_kwargs = {arg: value for arg, value in vars(args).items() if arg in learner_args}
    return updater_class, learner_kwargs


def _cast_arguments_to_true_types(
    arguments: Dict[str, Dict[str, Any]], args_dict: Dict[str, str]
) -> None:
    """Casts all values in args_dict to the value specified in arguments.

    Booleans, lists, None, and tuples are passed as a string to the script and argsparse will not
    convert them for us. After detecting the true types according to typing (typing is saved in
    ``arguments``), we now explicitly cast our arguments.
    After this step, ``args`` as a results of argparse.parse will contain the correct types and no
    further typecasting is required.
    """
    for argument_name, argument_kwargs in arguments.items():
        if args_dict[argument_name] == "None":
            args_dict[argument_name] = None
        if args_dict[argument_name] is None:
            continue
        if argument_kwargs.get("true_type") == bool:
            args_dict[argument_name] = args_dict[argument_name] == "True"
        elif argument_kwargs.get("true_type") in [list, tuple]:
            args_dict[argument_name] = ast.literal_eval(args_dict[argument_name])


def parse_arguments(
    config_module: ModuleType, function_names: List[str], ignore_args: List[str]
) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Parses all input arguments.

    Combines standard arguments with custom arguments in functions of Renate config.

    Args:
        config_module: Module containing function of which we extract arguments.
        function_names: List of function names in module for which extract arguments.
        ignore_args: List of arguments to be ignored since they are not passed via CLI.
    Returns:
        First return value are the passed arguments where strings are converted to Booleans, lists
        and tuples. Second return value is a dictionary containing the arguments that will be passed
        to all functions specified in ``function_names``.
    """
    arguments = _standard_arguments()
    _add_hyperparameter_arguments(arguments, "optimizer_fn" not in vars(config_module))
    function_args = {}
    for function_name in function_names:
        function_args[function_name] = get_function_args(
            config_module=config_module,
            function_name=function_name,
            all_args=arguments,
            ignore_args=ignore_args,
        )

    parser = argparse.ArgumentParser()

    for argument_group_name in ARGS_GROUP_ORDER:
        argument_group = parser.add_argument_group(argument_group_name)
        for argument_name, argument_kwargs in arguments.items():
            if argument_kwargs["argument_group"] == argument_group_name:
                argument_group.add_argument(
                    f"--{argument_name}",
                    **{
                        key: value
                        for key, value in argument_kwargs.items()
                        if key != "argument_group" and key != "true_type"
                    },
                )
    args = parser.parse_args()
    _cast_arguments_to_true_types(arguments=arguments, args_dict=vars(args))
    return args, function_args


def _standard_arguments() -> Dict[str, Dict[str, Any]]:
    """Returns information about the minimum number of arguments accepted by ``run_training.py``."""
    return {
        "updater": {
            "type": str,
            "required": True,
            "choices": list(parse_by_updater),
            "help": "Select the type of model update strategy.",
            "argument_group": REQUIRED_ARGS_GROUP,
        },
        "config_file": {
            "type": str,
            "required": True,
            "help": "Location of python file containing model_fn and data_module_fn.",
            "argument_group": REQUIRED_ARGS_GROUP,
        },
        "input_state_url": {
            "type": str,
            "help": "Location of previous Renate state (if available).",
            "argument_group": RENATE_STATE_ARGS_GROUP,
        },
        "output_state_url": {
            "type": str,
            "help": "Location where to store the next Renate state.",
            "argument_group": RENATE_STATE_ARGS_GROUP,
        },
        "max_epochs": {
            "type": int,
            "default": defaults.MAX_EPOCHS,
            "help": "Maximum number of (finetuning-equivalent) epochs. "
            f"Default: {defaults.MAX_EPOCHS}",
            "argument_group": OPTIONAL_ARGS_GROUP,
        },
        "task_id": {
            "type": str,
            "default": defaults.TASK_ID,
            "help": "Task ID matching the current dataset. If you do not distinguish between "
            "different tasks, ignore this"
            f" argument. Default: {defaults.TASK_ID}.",
            "argument_group": OPTIONAL_ARGS_GROUP,
        },
        "metric": {
            "type": str,
            "help": "Metric monitored during training to save checkpoints.",
            "argument_group": OPTIONAL_ARGS_GROUP,
        },
        "mode": {
            "type": str,
            "default": "min",
            "help": "Indicate whether a smaller `metric` is better (`min`) or a larger (`max`).",
            "argument_group": OPTIONAL_ARGS_GROUP,
        },
        "working_directory": {
            "type": str,
            "default": defaults.WORKING_DIRECTORY,
            "help": "Folder used by Renate to store files temporarily. Default: "
            f"{defaults.WORKING_DIRECTORY}.",
            "argument_group": OPTIONAL_ARGS_GROUP,
        },
        "seed": {
            "type": int,
            "default": defaults.SEED,
            "help": f"Seed used for this job. Default: {defaults.SEED}.",
            "argument_group": OPTIONAL_ARGS_GROUP,
        },
        "accelerator": {
            "type": str,
            "default": defaults.ACCELERATOR,
            "help": f"Accelerator used for this job. Default: {defaults.ACCELERATOR}.",
            "argument_group": OPTIONAL_ARGS_GROUP,
        },
        "devices": {
            "type": int,
            "default": defaults.DEVICES,
            "help": f"Devices used for this job. Default: {defaults.DEVICES} device.",
            "argument_group": OPTIONAL_ARGS_GROUP,
        },
        "strategy": {
            "type": str,
            "default": defaults.DISTRIBUTED_STRATEGY,
            "help": "Distributed training strategy when devices > 1. Default:"
            + f"{defaults.DISTRIBUTED_STRATEGY}.",
            "argument_group": OPTIONAL_ARGS_GROUP,
            "choices": list(pl.strategies.StrategyRegistry.keys()),
        },
        "precision": {
            "type": str,
            "default": defaults.PRECISION,
            "help": f"Distributed training precision. Default: {defaults.PRECISION}.",
            "argument_group": OPTIONAL_ARGS_GROUP,
            "choices": ("16", "32", "64", "bf16"),
        },
        "early_stopping": {
            "type": str,
            "default": str(defaults.EARLY_STOPPING),
            "choices": ["True", "False"],
            "help": "Enables the early stopping of the optimization. Default: "
            f"{defaults.EARLY_STOPPING}.",
            "argument_group": OPTIONAL_ARGS_GROUP,
            "true_type": bool,
        },
        "deterministic_trainer": {
            "type": str,
            "default": str(defaults.DETERMINISTIC_TRAINER),
            "choices": ["True", "False"],
            "help": "Enables deterministic training which may be slower. Default: "
            f"{defaults.DETERMINISTIC_TRAINER}.",
            "argument_group": OPTIONAL_ARGS_GROUP,
            "true_type": bool,
        },
        "gradient_clip_val": {
            "type": lambda x: None if x == "None" else float(x),
            "default": defaults.GRADIENT_CLIP_VAL,
            "help": "The value at which to clip gradients. None disables clipping.",
            "argument_group": OPTIONAL_ARGS_GROUP,
        },
        "gradient_clip_algorithm": {
            "type": lambda x: None if x == "None" else x,
            "default": defaults.GRADIENT_CLIP_ALGORITHM,
            "help": "Gradient clipping algorithm to use.",
            "choices": ["norm", "value", None],
            "argument_group": OPTIONAL_ARGS_GROUP,
        },
        "mask_unused_classes": {
            "default": str(defaults.MASK_UNUSED_CLASSES),
            "type": str,
            "choices": ["True", "False"],
            "help": "Whether to use a class mask to kill the unused logits. Useful possibly for "
            "class incremental learning methods. ",
            "argument_group": OPTIONAL_ARGS_GROUP,
            "true_type": bool,
        },
        "prepare_data": {
            "type": str,
            "default": "True",
            "choices": ["True", "False"],
            "help": "Whether to call DataModule.prepare_data(). Default: True.",
            "argument_group": DO_NOT_CHANGE_GROUP,
            "true_type": bool,
        },
        "st_checkpoint_dir": {
            "type": str,
            "help": "Location for checkpoints.",
            "argument_group": DO_NOT_CHANGE_GROUP,
        },
    }


def _add_hyperparameter_arguments(
    arguments: Dict[str, Dict[str, Any]], add_optimizer_args: bool
) -> None:
    """Adds arguments for the specified updater."""
    updater: Optional[str] = None
    for i, arg in enumerate(sys.argv):
        if arg == "--updater" and len(sys.argv) > i:
            updater = sys.argv[i + 1]
            break
    if updater is None:
        return

    assert updater in parse_by_updater, f"Unknown updater {updater}."
    parse_by_updater[updater](arguments)
    _add_optimizer_arguments(arguments, add_optimizer_args)
    for value in arguments.values():
        if "argument_group" not in value:
            value["argument_group"] = HYPERPARAMETER_ARGS_GROUP


def _add_optimizer_arguments(
    arguments: Dict[str, Dict[str, Any]], add_optimizer_args: bool
) -> None:
    """A helper function that adds optimizer arguments."""
    if add_optimizer_args:
        arguments.update(
            {
                "optimizer": {
                    "type": str,
                    "default": defaults.OPTIMIZER,
                    "help": "Optimizer used for training. Options: SGD or Adam. Default: "
                    f"{defaults.OPTIMIZER}.",
                },
                "learning_rate": {
                    "type": float,
                    "default": defaults.LEARNING_RATE,
                    "help": "Learning rate used during model update. Default: "
                    f"{defaults.LEARNING_RATE}.",
                },
                "momentum": {
                    "type": float,
                    "default": defaults.MOMENTUM,
                    "help": f"Momentum used during model update. Default: {defaults.MOMENTUM}.",
                },
                "weight_decay": {
                    "type": float,
                    "default": defaults.WEIGHT_DECAY,
                    "help": "Weight decay used during model update. Default: "
                    f"{defaults.WEIGHT_DECAY}.",
                },
            }
        )
    arguments.update(
        {
            "batch_size": {
                "type": int,
                "default": defaults.BATCH_SIZE,
                "help": "Batch size used during model update for the new data. Default: "
                f"{defaults.BATCH_SIZE}.",
            },
            "loss_weight": {
                "type": float,
                "default": defaults.LOSS_WEIGHT,
                "help": f"Loss weight used during model update. Default: {defaults.LOSS_WEIGHT}.",
            },
        }
    )


def _add_replay_learner_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds Replay Learner arguments."""
    arguments.update(
        {
            "memory_size": {
                "type": int,
                "default": defaults.MEMORY_SIZE,
                "help": "Memory size available for the memory buffer. Default: "
                f"{defaults.MEMORY_SIZE}.",
            },
            "batch_memory_frac": {
                "type": float,
                "default": defaults.BATCH_MEMORY_FRAC,
                "help": "Fraction of the batch populated with memory data. Default: "
                f"{defaults.BATCH_MEMORY_FRAC}.",
            },
        }
    )


def _add_base_experience_replay_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds Base Experience Replay arguments."""
    arguments.update(
        {
            "ema_memory_update_gamma": {
                "type": float,
                "default": defaults.EMA_MEMORY_UPDATE_GAMMA,
                "help": "Exponential moving average factor to update logits. Default: "
                f"{defaults.EMA_MEMORY_UPDATE_GAMMA}.",
            },
            "loss_normalization": {
                "type": int,
                "choices": [0, 1],
                "default": defaults.LOSS_NORMALIZATION,
                "help": "Whether to normalize the loss with respect to the loss weights. "
                f"Default: {bool(defaults.LOSS_NORMALIZATION)}.",
            },
        }
    )
    _add_replay_learner_arguments(arguments)


def _add_l2p_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    arguments.update(
        {
            "prompt_sim_loss_weight": {
                "type": float,
                "default": defaults.PROMPT_SIM_LOSS_WEIGHT,
                "help": "Prompt key similarity regularization weight. "
                f"Default: {defaults.PROMPT_SIM_LOSS_WEIGHT}",
            }
        }
    )


def _add_l2preplay_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    _add_l2p_arguments(arguments)
    _add_offline_er_arguments(arguments)


def _add_gdumb_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds GDumb arguments."""
    _add_replay_learner_arguments(arguments)
    _add_joint_arguments(arguments)


def _add_joint_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds Joint Learner arguments."""
    arguments.update(
        {
            "reset": {
                "type": str,
                "default": "True",
                "choices": ["True", "False"],
                "help": "Resets the model before the update. Default: True",
                "true_type": bool,
            },
        }
    )


def _add_finetuning_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds Fine Tuning arguments."""
    pass


def _add_offline_er_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    _add_replay_learner_arguments(arguments)
    arguments.update(
        {
            "loss_weight_new_data": {
                "type": float,
                "default": None,
                "help": "Weight assigned to loss on the new data.",
            }
        }
    )


def _add_experience_replay_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds Experience Replay arguments."""
    arguments.update(
        {
            "alpha": {
                "type": float,
                "default": defaults.ER_ALPHA,
                "help": f"Weight for the loss of the buffer data. Default: {defaults.ER_ALPHA}.",
            }
        }
    )
    _add_base_experience_replay_arguments(arguments)


def _add_dark_experience_replay_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds Dark Experience Replay arguments."""
    arguments.update(
        {
            "alpha": {
                "type": float,
                "default": defaults.DER_ALPHA,
                "help": f"Weight for logit regularization term. Default: {defaults.DER_ALPHA}.",
            },
            "beta": {
                "type": float,
                "default": defaults.DER_BETA,
                "help": f"Weight for memory loss term. Default: {defaults.DER_BETA}.",
            },
        }
    )
    _add_base_experience_replay_arguments(arguments)


def _add_pod_experience_replay_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds PODNet Experience Replay arguments."""
    arguments.update(
        {
            "alpha": {
                "type": float,
                "default": defaults.POD_ALPHA,
                "help": "Weight for intermediate representation regularization term. Default: "
                f"{defaults.POD_ALPHA}.",
            },
            "distillation_type": {
                "type": str,
                "default": defaults.POD_DISTILLATION_TYPE,
                "help": "Distillation type to apply with respect to the intermediate "
                f"representation. Default: {defaults.POD_DISTILLATION_TYPE}.",
            },
            "normalize": {
                "type": int,
                "default": defaults.POD_NORMALIZE,
                "help": "Whether to normalize both the current and cached features before computing"
                f" the Frobenius norm. Default: {defaults.POD_NORMALIZE}.",
            },
        }
    )
    _add_base_experience_replay_arguments(arguments)


def _add_cls_experience_replay_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds CLS Experience Replay arguments."""
    arguments.update(
        {
            "alpha": {
                "type": float,
                "default": defaults.CLS_ALPHA,
                "help": f"Weight for the cross-entropy loss term. Default: {defaults.CLS_ALPHA}.",
            },
            "beta": {
                "type": float,
                "default": defaults.CLS_BETA,
                "help": "Weight for the consistency memory loss term. Default: "
                f"{defaults.CLS_BETA}.",
            },
            "stable_model_update_weight": {
                "type": float,
                "default": defaults.CLS_STABLE_MODEL_UPDATE_WEIGHT,
                "help": "The starting weight for the exponential moving average to update the "
                f"stable model. Default: {defaults.CLS_STABLE_MODEL_UPDATE_WEIGHT}.",
            },
            "plastic_model_update_weight": {
                "type": float,
                "default": defaults.CLS_PLASTIC_MODEL_UPDATE_WEIGHT,
                "help": "The starting weight for the exponential moving average to update the "
                f"plastic model. Default: {defaults.CLS_PLASTIC_MODEL_UPDATE_WEIGHT}.",
            },
            "stable_model_update_probability": {
                "type": float,
                "default": defaults.CLS_STABLE_MODEL_UPDATE_PROBABILITY,
                "help": "Probability to update the stable model. Default: "
                f"{defaults.CLS_STABLE_MODEL_UPDATE_PROBABILITY}.",
            },
            "plastic_model_update_probability": {
                "type": float,
                "default": defaults.CLS_PLASTIC_MODEL_UPDATE_PROBABILITY,
                "help": "Probability to update the plastic model. Default: "
                f"{defaults.CLS_PLASTIC_MODEL_UPDATE_PROBABILITY}.",
            },
        }
    )
    _add_base_experience_replay_arguments(arguments)


def _add_super_experience_replay_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds Super Experience Replay arguments."""
    arguments.update(
        {
            "der_alpha": {
                "type": float,
                "default": defaults.SER_DER_ALPHA,
                "help": f"Weight for logit regularization term. Default: {defaults.SER_DER_ALPHA}.",
            },
            "der_beta": {
                "type": float,
                "default": defaults.SER_DER_BETA,
                "help": f"Weight for memory loss term. Default: {defaults.SER_DER_BETA}.",
            },
            "sp_shrink_factor": {
                "type": float,
                "default": defaults.SER_SP_SHRINK_FACTOR,
                "help": "Weight for logit regularization term. Default: "
                f"{defaults.SER_SP_SHRINK_FACTOR}.",
            },
            "sp_sigma": {
                "type": float,
                "default": defaults.SER_SP_SIGMA,
                "help": f"Weight for memory loss term. Default: {defaults.SER_SP_SIGMA}.",
            },
            "pod_alpha": {
                "type": float,
                "default": defaults.SER_POD_ALPHA,
                "help": "Weight for intermediate representation regularization term. Default: "
                f"{defaults.SER_POD_ALPHA}.",
            },
            "pod_distillation_type": {
                "type": str,
                "default": defaults.SER_POD_DISTILLATION_TYPE,
                "help": "Distillation type to apply with respect to the intermediate "
                f"intermediate representation. Default: {defaults.SER_POD_DISTILLATION_TYPE}.",
            },
            "pod_normalize": {
                "type": int,
                "default": defaults.SER_POD_NORMALIZE,
                "help": "Whether to normalize both the current and cached features before "
                f"computing the Frobenius norm. Default: {defaults.SER_POD_NORMALIZE}.",
            },
            "cls_alpha": {
                "type": float,
                "default": defaults.SER_CLS_ALPHA,
                "help": f"Weight for the consistency loss term. Default: {defaults.SER_CLS_ALPHA}.",
            },
            "cls_stable_model_update_weight": {
                "type": float,
                "default": defaults.SER_CLS_STABLE_MODEL_UPDATE_WEIGHT,
                "help": "The starting weight for the exponential moving average to update the "
                f"stable model. Default: {defaults.SER_CLS_STABLE_MODEL_UPDATE_WEIGHT}.",
            },
            "cls_plastic_model_update_weight": {
                "type": float,
                "default": defaults.SER_CLS_PLASTIC_MODEL_UPDATE_WEIGHT,
                "help": "The starting weight for the exponential moving average to update the "
                f"plastic model. Default: {defaults.SER_CLS_PLASTIC_MODEL_UPDATE_WEIGHT}.",
            },
            "cls_stable_model_update_probability": {
                "type": float,
                "default": defaults.SER_CLS_STABLE_MODEL_UPDATE_PROBABILITY,
                "help": "Probability to update the stable model. Default: "
                f"{defaults.SER_CLS_STABLE_MODEL_UPDATE_PROBABILITY}.",
            },
            "cls_plastic_model_update_probability": {
                "type": float,
                "default": defaults.SER_CLS_PLASTIC_MODEL_UPDATE_PROBABILITY,
                "help": "Probability to update the plastic model. Default: "
                f"{defaults.SER_CLS_PLASTIC_MODEL_UPDATE_PROBABILITY}.",
            },
        }
    )
    _add_base_experience_replay_arguments(arguments)


def _add_rd_learner_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds Repeated Distill Learner arguments."""
    arguments.update(
        {
            "memory_size": {
                "type": int,
                "default": defaults.MEMORY_SIZE,
                "help": f"Memory size available for the memory buffer. Default: "
                f"{defaults.MEMORY_SIZE}.",
            }
        }
    )


def _add_avalanche_ewc_learner_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds EWC arguments."""
    arguments.update(
        {
            "ewc_lambda": {
                "type": float,
                "default": defaults.EWC_LAMBDA,
                "help": f"EWC regularization hyperparameter. Default: {defaults.EWC_LAMBDA}.",
            }
        }
    )


def _add_avalanche_lwf_learner_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds LwF arguments."""
    arguments.update(
        {
            "alpha": {
                "type": float,
                "default": defaults.LWF_ALPHA,
                "help": f"Distillation loss weight. Default: {defaults.LWF_ALPHA}.",
            },
            "temperature": {
                "type": float,
                "default": defaults.LWF_TEMPERATURE,
                "help": "Temperature of the softmax function. Default: "
                f"{defaults.LWF_TEMPERATURE}.",
            },
        },
    )


def _add_avalanche_icarl_learner_arguments(arguments: Dict[str, Dict[str, Any]]) -> None:
    """A helper function that adds iCarl arguments."""
    arguments.update(
        {
            "memory_size": {
                "type": int,
                "default": defaults.MEMORY_SIZE,
                "help": f"Number of exemplars being stored. Default: {defaults.MEMORY_SIZE}.",
            }
        },
    )


def get_function_kwargs(args: argparse.Namespace, function_args: Dict[str, Any]) -> Dict[str, Any]:
    """Returns the kwargs for a function with defined arguments based on provided values.

    Args:
        args: Values for specific keys.
        function_args: Arguments of the function for which we extract values.
    """
    return {key: value for key, value in vars(args).items() if key in function_args}


def get_data_module_fn_kwargs(
    config_module, config_space: Dict[str, Any], cast_arguments: Optional[bool] = False
) -> Dict[str, Any]:
    """Returns the kwargs for a ``data_module_fn`` with defined arguments based on config_space."""
    return _get_function_kwargs_helper(
        config_module, config_space, "data_module_fn", ["data_path"], cast_arguments
    )


def get_model_fn_kwargs(
    config_module, config_space: Dict[str, Any], cast_arguments: Optional[bool] = False
) -> Dict[str, Any]:
    """Returns the kwargs for a ``model_fn`` with defined arguments based on config_space."""
    return _get_function_kwargs_helper(config_module, config_space, "model_fn", [], cast_arguments)


def get_transforms_kwargs(
    config_module, config_space: Dict[str, Any], cast_arguments: Optional[bool] = False
) -> Dict[str, Callable]:
    """Returns the transforms based on config_space."""
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
        if hasattr(config_module, transform_fn_name):
            transforms[transform_fn_name] = getattr(config_module, transform_fn_name)(
                **_get_function_kwargs_helper(
                    config_module, config_space, transform_fn_name, [], cast_arguments
                )
            )
    return transforms


def get_metrics_fn_kwargs(
    config_module, config_space: Dict[str, Any], cast_arguments: Optional[bool] = False
) -> Dict[str, Any]:
    """Returns the kwargs for a ``metrics_fn`` with defined arguments based on config_space."""
    return _get_function_kwargs_helper(
        config_module, config_space, "metrics_fn", [], cast_arguments
    )


def _get_function_kwargs_helper(
    config_module,
    config_space: Dict[str, Any],
    function_name: str,
    ignore_args: List[str],
    cast_arguments: Optional[bool] = False,
) -> Dict[str, Any]:
    """Returns kwargs for function based on its interface."""
    all_args = {}
    function_args = get_function_args(config_module, function_name, all_args, ignore_args)
    filtered_args = {key: value for key, value in config_space.items() if key in function_args}
    if cast_arguments:
        filtered_all_args = {key: value for key, value in all_args.items() if key in filtered_args}
        _cast_arguments_to_true_types(arguments=filtered_all_args, args_dict=filtered_args)
    return filtered_args


def get_transforms_dict(
    config_module: ModuleType,
    args: Union[argparse.Namespace, Dict[str, str]],
    function_args: Dict[str, Dict[str, Any]],
) -> Dict[str, Callable]:
    """Creates and returns data transforms for updater."""
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
        if hasattr(config_module, transform_fn_name):
            transforms[transform_fn_name] = getattr(config_module, transform_fn_name)(
                **get_function_kwargs(args, function_args[transform_fn_name])
            )
    return transforms


def get_argument_type(arg_spec: inspect.FullArgSpec, argument_name: str) -> Type:
    """Returns the type of the argument with ``argument_name``."""

    def raise_error(arg_type, arg_name):
        raise TypeError(f"Type {arg_type} is not supported (argument {arg_name}).")

    if argument_name not in arg_spec.annotations:
        raise TypeError(f"Missing type annotation for argument {argument_name}.")
    if hasattr(arg_spec.annotations[argument_name], "__origin__"):
        if arg_spec.annotations[argument_name].__origin__ == Union:
            if len(arg_spec.annotations[argument_name].__args__) != 2 or arg_spec.annotations[
                argument_name
            ].__args__[1] != type(None):
                raise_error(arg_spec.annotations[argument_name], argument_name)
            if hasattr(arg_spec.annotations[argument_name].__args__[0], "__origin__"):
                if arg_spec.annotations[argument_name].__args__[0].__origin__ in [list, tuple]:
                    argument_type = arg_spec.annotations[argument_name].__args__[0].__origin__
                else:
                    raise_error(arg_spec.annotations[argument_name], argument_name)
            else:
                argument_type = arg_spec.annotations[argument_name].__args__[0]
            if arg_spec.annotations[argument_name].__origin__ in [list, tuple]:
                argument_type = arg_spec.annotations[argument_name].__origin__
        elif arg_spec.annotations[argument_name].__origin__ in [list, tuple]:
            argument_type = arg_spec.annotations[argument_name].__origin__
        else:
            raise_error(arg_spec.annotations[argument_name], argument_name)
    else:
        argument_type = arg_spec.annotations[argument_name]
    if argument_type not in [int, bool, float, str, list, tuple]:
        raise_error(argument_type, argument_name)
    return argument_type


def to_dense_str(value: Union[bool, List, Tuple, None]) -> str:
    """Converts a variable to string without empty spaces."""
    return str(value).replace(" ", "")


def get_function_args(
    config_module: ModuleType,
    function_name: str,
    all_args: Dict[str, Dict[str, Any]],
    ignore_args: List[str],
) -> List[str]:
    """Returns all argument names of the function and appends new arguments to ``all_args``.

    Args:
        config_module: Renate config file containing functions to access the model and data.
        function_name: Function of which we want to return the arguments.
        all_args: List of all arguments that will be an input to the script.
        ignore_args: Arguments which will not be added to ``all_args`` because these values will
            be created on the fly and are no input to the script.
    """
    if not hasattr(config_module, function_name):
        return []
    arg_spec = inspect.getfullargspec(getattr(config_module, function_name))
    known_args = [arg for arg in arg_spec.args if arg in all_args]
    new_args = [arg for arg in arg_spec.args if arg not in all_args and arg not in ignore_args]
    if arg_spec.defaults is None:
        default_values = {}
    else:
        default_values = {
            arg_spec.args[len(arg_spec.args) - len(arg_spec.defaults) + i]: default
            for i, default in enumerate(arg_spec.defaults)
        }
    for argument_name in known_args:
        argument_type = get_argument_type(arg_spec, argument_name)
        expected_type = all_args[argument_name]["type"]
        if argument_type != expected_type:
            raise TypeError(
                f"Types of `{argument_name}` are not consistent. Defined as type `{argument_type}`"
                f" as well as `{expected_type}`."
            )
        if argument_name not in default_values:
            all_args[argument_name]["required"] = True

    for argument_name in new_args:
        true_type = get_argument_type(arg_spec, argument_name)
        all_args[argument_name] = {
            "type": str if true_type in [bool, list, tuple] else true_type,
            "argument_group": CUSTOM_ARGS_GROUP,
            "true_type": true_type,
        }
        if argument_name in default_values:
            default_value = default_values[argument_name]
            if true_type in [bool, list, tuple] and default_value is not None:
                default_value = to_dense_str(default_value)
            all_args[argument_name]["default"] = default_value
        else:
            all_args[argument_name]["required"] = True
    return arg_spec.args


def get_scheduler_kwargs(
    config_module: ModuleType,
) -> Tuple[Optional[Type[TrialScheduler]], Optional[Dict[str, Any]]]:
    """Creates and returns scheduler type and kwargs for the HPO scheduler."""
    scheduler_fn_name = "scheduler_fn"
    if scheduler_fn_name in vars(config_module):
        return getattr(config_module, scheduler_fn_name)()
    return None, None


parse_by_updater = {
    "ER": _add_experience_replay_arguments,
    "LearningToPrompt": _add_l2p_arguments,
    "LearningToPromptReplay": _add_l2preplay_arguments,
    "DER": _add_dark_experience_replay_arguments,
    "POD-ER": _add_pod_experience_replay_arguments,
    "CLS-ER": _add_cls_experience_replay_arguments,
    "Super-ER": _add_super_experience_replay_arguments,
    "GDumb": _add_gdumb_arguments,
    "Joint": _add_joint_arguments,
    "FineTuning": _add_finetuning_arguments,
    "RD": _add_rd_learner_arguments,
    "Offline-ER": _add_offline_er_arguments,
    "Avalanche-ER": _add_experience_replay_arguments,
    "Avalanche-EWC": _add_avalanche_ewc_learner_arguments,
    "Avalanche-LwF": _add_avalanche_lwf_learner_arguments,
    "Avalanche-iCaRL": _add_avalanche_icarl_learner_arguments,
}
