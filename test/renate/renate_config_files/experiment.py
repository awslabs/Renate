# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from renate.benchmark.experimentation import execute_experiment_job


def test_execute_experiment_job():
    execute_experiment_job(backend="local")
