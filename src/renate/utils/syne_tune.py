# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Dict, Union

from syne_tune.config_space import Domain, from_dict, to_dict
from syne_tune.experiments import ExperimentResult


def redirect_to_tmp(uri: str) -> str:
    """Changes uri in /opt/ml to /tmp.

    Syne Tune stores checkpoints by default in /opt/ml when running on SageMaker. While we want to store checkpoints,
    we have no interest in uploading them to S3. Therefore, this function changes the location to /tmp instead.
    """
    if "SM_MODEL_DIR" in os.environ:  # If running on sagemaker, redirect checkpoints to /tmp
        assert uri.startswith("/opt/ml")
        uri = "/tmp" + uri[7:]
    return uri


def config_space_to_dict(
    config_space: Dict[str, Union[Domain, int, float, str]]
) -> Dict[str, Union[int, float, str]]:
    """Converts `config_space` into a dictionary that can be saved as a json file."""
    # TODO: remove with Syne Tune 0.3.3
    return {k: to_dict(v) if isinstance(v, Domain) else v for k, v in config_space.items()}


def config_space_from_dict(
    config_space_dict: Dict[str, Union[int, float, str]]
) -> Dict[str, Union[Domain, int, float, str]]:
    """Converts the given dictionary into a Syne Tune search space."""
    # TODO: remove with Syne Tune 0.3.3
    return {k: from_dict(v) if isinstance(v, dict) else v for k, v in config_space_dict.items()}


def best_hyperparameters(
    experiment: ExperimentResult, config_space: Dict[str, Union[Domain, int, float, str]]
) -> Dict[str, Union[int, float, str]]:
    """Returns the values of all keys in the `config_space` that belong to a Syne Tune search space."""
    return {
        k[7:]: v
        for k, v in experiment.best_config().items()
        if k.startswith("config_") and isinstance(config_space[k[7:]], Domain)
    }


def is_syne_tune_config_space(config_space: Dict[str, Any]) -> bool:
    """Returns `True` if any value in the configuration space defines a Syne Tune search space."""
    return any(
        [
            isinstance(hyperparameter_instance, Domain)
            for hyperparameter_instance in config_space.values()
        ]
    )
