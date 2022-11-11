# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Dict, Union

from syne_tune.config_space import Domain, from_dict, to_dict


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
