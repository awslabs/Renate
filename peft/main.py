# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import tempfile
from functools import partial
from pathlib import Path
from typing import Union, Optional, Dict

import torch
import torchmetrics
import transformers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModelForSequenceClassification
from pytorch_lightning.loggers.logger import Logger

from renate import defaults
from renate.benchmark.datasets.nlp_datasets import HuggingFaceTextDataModule
from renate.benchmark.models.transformer import HuggingFaceSequenceClassificationTransformer
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule
from renate.models.renate_module import RenateWrapper
from renate.updaters.model_updater import RenateModelCheckpoint
from renate.updaters.peft_learner import PeftLearner
from renate.utils.file import upload_folder_to_s3

logging_logger = logging.getLogger(__name__)


def get_data(dataset_name: str, data_path: Union[str, Path], seed: int = 123) -> RenateDataModule:
    tokenizer = transformers.AutoTokenizer.from_pretrained(dataset_name)
    data_module = HuggingFaceTextDataModule(
        data_path=data_path,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        val_size=0.2,
        seed=seed,
    )
    data_module.prepare_data()
    data_module.setup()
    return data_module


def get_trainer(
    checkpoint_path: Union[str, Path],
    metric: Optional[str] = None,
    mode: defaults.SUPPORTED_TUNING_MODE_TYPE = "min",
    max_epochs: int = defaults.MAX_EPOCHS,
    logger: Logger = defaults.LOGGER(**defaults.LOGGER_KWARGS),
    accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
    devices: Optional[int] = None,
    strategy: Optional[str] = defaults.DISTRIBUTED_STRATEGY,
    precision: str = defaults.PRECISION,
    deterministic_trainer: bool = defaults.DETERMINISTIC_TRAINER,
):
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor=metric,
        mode=mode,
    )
    callbacks = [model_checkpoint_callback]
    return Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        deterministic=deterministic_trainer,
        strategy=strategy,
        precision=precision,
    )


def main(pretrained_model_name: str, dataset_name: str, s3url: str, num_outputs: int):
    datapath = Path.home().joinpath("data")
    datapath.mkdir(exist_ok=True)

    checkpointpath = Path.home().joinpath("checkpoint")
    checkpointpath.mkdir(exist_ok=True)

    model = HuggingFaceSequenceClassificationTransformer(
        pretrained_model_name=pretrained_model_name, num_outputs=num_outputs
    )
    lossfunction = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = partial(torch.optim.AdamW, lr=1e-3)  # Specify your optimizer params here
    module = PeftLearner(
        model, loss_fn=lossfunction, optimizer=optimizer, batch_size=defaults.BATCH_SIZE
    )
    trainer = get_trainer(checkpoint_path=checkpointpath)

    # Setup data
    data_module = get_data(dataset_name, datapath)
    module.on_model_update_start(
        train_dataset=data_module.train_data(),
        val_dataset=data_module.val_data(),
    )

    # Fit the model
    trainer.fit(module)

    # Uplaod checkpoints to s3
    upload_folder_to_s3(checkpointpath, s3url)


if __name__ == "__main__":
    main(
        pretrained_model_name="distilbert-base-uncased",
        dataset_name="rotten_tomatoes",
        s3url="dummy",
        num_outputs=2,
    )
