# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This script is used to launch model tuning on SageMaker. Previously passed arguments on a different machine are loaded
again and the update process is started with these parameters.
Arguments expected are described in renate.tuning.execute_tuning_job.
"""
import json

import renate.defaults as defaults
from renate.benchmark.experimentation import execute_experiment_job
from renate.tuning import execute_tuning_job
from renate.utils.syne_tune import config_space_from_dict

if __name__ == "__main__":
    with open(defaults.JOB_KWARGS_FILE, "r") as f:
        job_kwargs = json.load(f)
    job_kwargs["config_space"] = config_space_from_dict(job_kwargs["config_space"])
    if "experiment_outputs_url" in job_kwargs:
        execute_experiment_job(backend="local", **job_kwargs)
    else:
        execute_tuning_job(backend="local", **job_kwargs)
