# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Union

import numpy as np
from syne_tune.config_space import Domain, choice, loguniform, uniform


def _get_range(start, end, step):
    return [np.round(i, 8) for i in np.arange(start, end + 1e-8, step)]


_learner_config_space = {
    "optimizer": "SGD",
    "momentum": 0.0,
    "weight_decay": loguniform(1e-6, 1e-2),
    "learning_rate": loguniform(0.001, 0.5),
    "batch_size": 32,
    "max_epochs": 50,
}
_replay_config_space = {
    **_learner_config_space,
    **{
        "memory_batch_size": 32,
        "memory_size": 500,
    },
}
_er_config_space = {
    **_replay_config_space,
    **{"updater": "ER", "alpha": 0.5, "loss_normalization": 0, "loss_weight": 0.5},
}
_der_config_space = {
    **_replay_config_space,
    **{
        "updater": "DER",
        "alpha": uniform(0.0, 1.0),
        "beta": uniform(0.0, 1.0),
        "loss_normalization": 0,
        "loss_weight": 1.0,
    },
}
_super_er_config_space = {
    **_replay_config_space,
    **{
        "updater": "Super-ER",
        "der_alpha": uniform(0.0, 1.0),
        "der_beta": uniform(0.0, 1.0),
        "sp_shrink_factor": choice(_get_range(0.3, 1, 0.1)),
        "sp_sigma": choice([0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
        "cls_alpha": choice(_get_range(0, 1, 0.05)),
        "cls_stable_model_update_weight": 0.999,
        "cls_plastic_model_update_weight": 0.999,
        "cls_stable_model_update_probability": uniform(0.01, 0.49),
        "cls_plastic_model_update_probability": uniform(0.5, 1.0),
        "loss_normalization": 1,
        "ema_memory_update_gamma": uniform(0.95, 1.0),
    },
}

_repeated_distill_config_space = {
    "optimizer": choice(["SGD", "Adam"]),
    "momentum": loguniform(1e-6, 1e-1),
    "weight_decay": loguniform(1e-6, 1e-2),
    "learning_rate": loguniform(0.001, 0.5),
    "batch_size": choice([32, 64, 128]),
    "max_epochs": 50,
    "memory_size": 1000,
}


def config_space(updater: str) -> Dict[str, Union[Domain, str, int, float]]:
    """Returns the default configuration space for the updater."""
    config_spaces = {
        "ER": _er_config_space,
        "DER": _der_config_space,
        "SUPER-ER": _super_er_config_space,
        "RD": _repeated_distill_config_space,
    }
    return config_spaces[updater.upper()]
