# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
import os
from typing import Any, Dict, Literal

from pytorch_lightning.loggers import TensorBoardLogger
from syne_tune.optimizer.schedulers import FIFOScheduler

OPTIMIZER = "Adam"
SUPPORTED_OPTIMIZERS = ["Adam", "SGD"]
SUPPORTED_OPTIMIZERS_TYPE = Literal["Adam", "SGD"]
LR_SCHEDULER_INTERVAL = "epoch"
SUPPORTED_LR_SCHEDULER_INTERVAL = ["epoch", "step"]
SUPPORTED_LR_SCHEDULER_INTERVAL_TYPE = Literal["epoch", "step"]
LEARNING_RATE = 3e-4
MOMENTUM = 0.0
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 50
BATCH_SIZE = 32
BATCH_MEMORY_FRAC = 0.5
LOSS_WEIGHT = 1.0
SEED = 0
EMA_MEMORY_UPDATE_GAMMA = 1.0
VALIDATION_SIZE = 0.0
LOSS_NORMALIZATION = 1
EARLY_STOPPING = False
DETERMINISTIC_TRAINER = False

ACCELERATOR = "auto"
SUPPORTED_ACCELERATORS = ["auto", "cpu", "gpu", "tpu"]
SUPPORTED_ACCELERATORS_TYPE = Literal["auto", "cpu", "gpu", "tpu"]
DEVICES = 1
VOLUME_SIZE = 60
DISTRIBUTED_STRATEGY = "ddp"
PRECISION = "32"
GRADIENT_CLIP_VAL = None
GRADIENT_CLIP_ALGORITHM = None

LEARNER = "ER"
INSTANCE_COUNT = 1
INSTANCE_MAX_TIME = 3 * 24 * 3600
N_WORKERS = 1
INSTANCE_TYPE = "ml.c5.xlarge"
PYTHON_VERSION = "py39"
FRAMEWORK_VERSION = "1.13.1"

TASK_ID = "default_task"
MASK_UNUSED_CLASSES = False
WORKING_DIRECTORY = "renate_working_dir"
LOGGER = TensorBoardLogger
LOGGER_KWARGS = {
    "save_dir": os.path.expanduser(os.path.join("~", ".renate", WORKING_DIRECTORY)),
    "version": 1,
    "name": "lightning_logs",
}
JOB_KWARGS_FILE = "job_kwargs.json"
JOB_NAME = "renate"
SUPPORTED_TUNING_MODE = ["min", "max"]
SUPPORTED_TUNING_MODE_TYPE = Literal["min", "max"]
SAVE_BENCHMARK_STATE = True

SUPPORTED_BACKEND = ["local", "sagemaker"]
SUPPORTED_BACKEND_TYPE = Literal["local", "sagemaker"]

# ER
ER_ALPHA = 1.0

# DER
DER_ALPHA = 1.0
DER_BETA = 1.0

# POD-ER
POD_ALPHA = 1.0
POD_DISTILLATION_TYPE = "spatial"
POD_NORMALIZE = 1

# CLS-ER
CLS_ALPHA = 0.5
CLS_BETA = 0.1
CLS_STABLE_MODEL_UPDATE_WEIGHT = 0.999
CLS_STABLE_MODEL_UPDATE_PROBABILITY = 0.7
CLS_PLASTIC_MODEL_UPDATE_WEIGHT = 0.999
CLS_PLASTIC_MODEL_UPDATE_PROBABILITY = 0.9

# SuperER
SER_DER_ALPHA = 1.0
SER_DER_BETA = 1.0
SER_SP_SHRINK_FACTOR = 0.95
SER_SP_SIGMA = 0.001
SER_POD_ALPHA = 1.0
SER_POD_DISTILLATION_TYPE = "spatial"
SER_POD_NORMALIZE = 1
SER_CLS_ALPHA = 0.1
SER_CLS_STABLE_MODEL_UPDATE_WEIGHT = 0.999
SER_CLS_STABLE_MODEL_UPDATE_PROBABILITY = 0.7
SER_CLS_PLASTIC_MODEL_UPDATE_WEIGHT = 0.999
SER_CLS_PLASTIC_MODEL_UPDATE_PROBABILITY = 0.9

# EWC
EWC_LAMBDA = 0.4

# LwF
LWF_ALPHA = 1
LWF_TEMPERATURE = 2

MEMORY_SIZE = 32

# Benchmark datasets/models
TOKENIZER_KWARGS = {"padding": "max_length", "max_length": 128, "truncation": True}

# L2p
PROMPT_SIM_LOSS_WEIGHT = 0.5


def scheduler(config_space: Dict[str, Any], mode: str, metric: str):
    return FIFOScheduler(
        config_space=config_space,
        searcher="random",
        mode=mode,
        metric=metric,
    )


def current_timestamp() -> str:
    return str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))


def data_folder(working_directory: str) -> str:
    return os.path.join(working_directory, "data")


def input_state_folder(working_directory: str) -> str:
    return os.path.join(working_directory, "input_state")


def output_state_folder(working_directory: str) -> str:
    return os.path.join(working_directory, "output_state")


def logs_folder(working_directory: str) -> str:
    return os.path.join(working_directory, "logs")


def model_file(state_folder: str) -> str:
    return os.path.join(state_folder, "model.ckpt")


LEARNER_CHECKPOINT_NAME = "learner.ckpt"
AVALANCHE_CHECKPOINT_NAME = "avalanche.ckpt"


def learner_state_file(state_folder: str) -> str:
    return os.path.join(state_folder, LEARNER_CHECKPOINT_NAME)


def avalanche_state_file(state_folder: str) -> str:
    return os.path.join(state_folder, AVALANCHE_CHECKPOINT_NAME)


def metric_summary_file(logs_folder: str, special_str: str = "") -> str:
    return os.path.join(logs_folder, f"metrics_summary{special_str}.csv")


def hpo_file(state_folder: str) -> str:
    return os.path.join(state_folder, "hpo.csv")
