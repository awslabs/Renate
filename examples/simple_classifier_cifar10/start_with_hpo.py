# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.config_space import choice, loguniform, uniform

from renate.tuning import execute_tuning_job

config_space = {
    "optimizer": "SGD",
    "momentum": uniform(0.1, 0.9),
    "weight_decay": 0.0,
    "learning_rate": loguniform(1e-4, 1e-1),
    "alpha": uniform(0.0, 1.0),
    "batch_size": choice([32, 64, 128, 256]),
    "memory_batch_size": 32,
    "memory_size": 1000,
    "loss_normalization": 0,
    "loss_weight": uniform(0.0, 1.0),
}

if __name__ == "__main__":

    AWSID = "387922178948"
    AWSREGION = "us-west-2"

    execute_tuning_job(
        config_space=config_space,
        mode="max",
        metric="val_accuracy",
        updater="ER",  # we train with Experience Replay
        max_epochs=50,
        chunk_id=0,  # we select the first chunk of our dataset, you will probably not need this in practice
        config_file="renate_config.py",
        requirements_file="requirements.txt",
        # uncomment the line below only if you already created a model with this script
        # state_url=f"s3://sagemaker-{AWSREGION}-{AWSID}/renate-training-test-cifar10/",
        # replace the url below with a different one, for example by appending "updated" to the folder name
        next_state_url=f"s3://sagemaker-{AWSREGION}-{AWSID}/renate-training-test-cifar10/",
        backend="sagemaker",  # we will run this on SageMaker, but you can select "local" to run this locally
        role=get_execution_role(),
        instance_count=0,
        instance_type="ml.g4dn.2xlarge",
        max_num_trials_finished=100,
        scheduler="asha",  # we will run ASHA to optimize our hyperparameters
        n_workers=4,
        job_name="testjob",
    )
