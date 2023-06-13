# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torchmetrics
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from syne_tune import Reporter
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset

from renate import defaults
from renate.updaters.peft_learner import PeftLearner
from renate.utils.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from renate.utils.distributed_strategies import create_strategy
from renate.utils.file import unlink_file_or_folder
from renate.utils.misc import int_or_str
from ..models import RenateModule
from renate.updaters.model_updater import SyneTuneCallback, ModelCheckpoint

logging_logger = logging.getLogger(__name__)


def loss_fn():
    return torch.nn.CrossEntropyLoss()


def main():
    trainable = PeftLearner()
