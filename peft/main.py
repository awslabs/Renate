# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from functools import partial
from pathlib import Path
from typing import Union, Optional
import sys

import torch
import transformers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from transformers import default_data_collator

from renate import defaults
from renate.benchmark.datasets.nlp_datasets import (
    HuggingFaceTextDataModule,
    HuggingFaceExtractiveQADataModule,
)
from renate.benchmark.models.transformer import (
    HuggingFaceSequenceClassificationTransformer,
    HuggingFaceQuestionAnsweringTransformer,
)
from renate.data.data_module import RenateDataModule
from renate.updaters.peft_learner import PeftLearner, QAPeft
from renate.updaters.learner import Learner
from renate.utils.file import upload_folder_to_s3
from renate.utils.misc import int_or_str

logging_logger = logging.getLogger(__name__)


def get_data(
    dataset_name: str, pretrained_model_name: str, data_path: Union[str, Path], seed: int = 123
) -> RenateDataModule:
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)
    data_module = HuggingFaceExtractiveQADataModule(
        data_path=str(data_path),
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        val_size=0.2,
        seed=seed,
    )
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
        # enable_progress_bar=False,
        deterministic=deterministic_trainer,
        strategy=strategy,
        precision=int_or_str(precision),
        enable_model_summary=False,
    )


def main(pretrained_model_name: str, dataset_name: str, s3url: str, num_outputs: int):
    datapath = Path.home().joinpath("data")
    datapath.mkdir(exist_ok=True)

    checkpointpath = Path.home().joinpath("checkpoint")
    checkpointpath.mkdir(exist_ok=True)

    # model = HuggingFaceSequenceClassificationTransformer(
    #     pretrained_model_name=pretrained_model_name, num_outputs=num_outputs
    # )
    model = HuggingFaceQuestionAnsweringTransformer(pretrained_model_name=pretrained_model_name)
    lossfunction = torch.nn.CrossEntropyLoss()
    lossfunction = lambda x, y: x
    optimizer = partial(
        torch.optim.AdamW, lr=1e-5, weight_decay=1e-2
    )  # Specify your optimizer params here
    module = QAPeft(
        model,
        loss_fn=lossfunction,
        optimizer=optimizer,
        batch_size=12,
        logged_metrics={},
    )

    # Setup data
    data_module = get_data(dataset_name, pretrained_model_name, datapath)
    module.on_model_update_start(
        train_dataset=data_module.train_data(),
        val_dataset=data_module.val_data(),
        train_dataset_collate_fn=default_data_collator,
        val_dataset_collate_fn=default_data_collator,
    )
    trainer = get_trainer(checkpoint_path=checkpointpath, precision="16")

    # print(data_module.train_data()[20])
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
