# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import renate
from renate.benchmark.experimentation import execute_experiment_job


def experimentation_config():
    return str(Path(renate.__path__[0]) / "benchmark" / "experiment_config.py")
