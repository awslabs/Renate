# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This script is used to launch model tuning on SageMaker. Previously passed arguments on a different machine are loaded
from S3 and the update process is started with these parameters.
Arguments expected are described in renate.tuning.execute_tuning_job.
"""
import argparse
import json

import renate.defaults as defaults
from renate.tuning import execute_tuning_job
from renate.utils.file import download_file_from_s3
from renate.utils.syne_tune import config_space_from_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_kwargs_bucket", type=str, help="Bucket in which file is stored.")
    parser.add_argument(
        "--job_kwargs_object_name", type=str, help="Location of pickled dictionary on S3."
    )
    args = parser.parse_args()
    download_file_from_s3(
        args.job_kwargs_bucket, args.job_kwargs_object_name, defaults.JOB_KWARGS_FILE
    )
    with open(defaults.JOB_KWARGS_FILE, "r") as f:
        job_kwargs = json.load(f)
    job_kwargs["config_space"] = config_space_from_dict(job_kwargs["config_space"])
    # Tuning on SageMaker is only possible with local backend: https://github.com/awslabs/syne-tune/issues/214
    execute_tuning_job(backend="local", **job_kwargs)
