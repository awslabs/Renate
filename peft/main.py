# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
from functools import partial
from pathlib import Path
from typing import Dict, Union, Optional

import torch
import transformers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from transformers import default_data_collator, DataCollatorForLanguageModeling

from renate import defaults
from renate.benchmark.datasets.nlp_datasets import (
    HuggingFaceLanguageModelingModule,
    HuggingFaceTextDataModule,
    HuggingFaceExtractiveQADataModule,
)
from renate.benchmark.models.transformer import (
    HuggingFaceSequenceClassificationTransformer,
    HuggingFaceQuestionAnsweringTransformer,
    HuggingFaceLanguageModelingTransformer,
)
from renate.data.data_module import RenateDataModule
from renate.updaters.peft_learner import PeftLearner, QAPeft
from renate.utils.file import upload_folder_to_s3
from renate.utils.misc import int_or_str


def get_data(
    dataset_name: str, pretrained_model_name: str, data_path: Union[str, Path], seed: int = 123
) -> RenateDataModule:
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)
    if dataset_name == "squad":
        data_module = HuggingFaceExtractiveQADataModule(
            data_path=str(data_path),
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            val_size=0.2,
            seed=seed,
        )
    elif dataset_name == "rotten_tomatoes":
        data_module = HuggingFaceTextDataModule(
            data_path=str(data_path),
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            val_size=0.2,
            seed=seed,
        )
    elif dataset_name == "eli5":
        data_module = HuggingFaceLanguageModelingModule(
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
        enable_progress_bar=True,
        deterministic=deterministic_trainer,
        strategy=strategy,
        precision=int_or_str(precision),
        enable_model_summary=False,
    )


def main(pretrained_model_name: str, dataset_name: str, s3url: str, config: Dict):
    datapath = Path.home().joinpath("data")
    datapath.mkdir(exist_ok=True)

    checkpointpath = Path.home().joinpath("checkpoint")
    checkpointpath.mkdir(exist_ok=True)

    # Setup data
    data_module = get_data(dataset_name, pretrained_model_name, datapath)

    if dataset_name == "rotten_tomatoes":
        num_outputs = 2
        model = HuggingFaceSequenceClassificationTransformer(
            pretrained_model_name=pretrained_model_name, num_outputs=num_outputs
        )
        lossfunction = torch.nn.CrossEntropyLoss()
        collator_fn = None
        module_cls = PeftLearner

    elif dataset_name == "squad":
        model = HuggingFaceQuestionAnsweringTransformer(pretrained_model_name=pretrained_model_name)
        lossfunction = lambda x, y: x
        collator_fn = default_data_collator
        module_cls = QAPeft

    elif dataset_name == "eli5":
        do_causal_lm = False
        model = HuggingFaceLanguageModelingTransformer(
            pretrained_model_name=pretrained_model_name, causal=do_causal_lm
        )
        lossfunction = lambda x, y: x
        data_module._tokenizer.pad_token = data_module._tokenizer.eos_token
        collator_fn = DataCollatorForLanguageModeling(
            mlm=not do_causal_lm, tokenizer=data_module._tokenizer
        )
        module_cls = QAPeft

    # Specify your optimizer params here
    optimizer = partial(torch.optim.AdamW, lr=1e-5, weight_decay=1e-2)
    module = module_cls(
        model,
        loss_fn=lossfunction,
        optimizer=optimizer,
        batch_size=config["batch_size"],
        logged_metrics={},
    )

    module.on_model_update_start(
        train_dataset=data_module.train_data(),
        val_dataset=data_module.val_data(),
        train_dataset_collate_fn=collator_fn,
        val_dataset_collate_fn=collator_fn,
    )
    trainer = get_trainer(
        checkpoint_path=checkpointpath,
        precision="16",
        max_epochs=config["max_epochs"],
    )

    # Fit the model
    trainer.fit(module)

    # Uplaod checkpoints to s3
    # upload_folder_to_s3(checkpointpath, s3url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument(
        "--dataset",
        type=str,
        default="rotten_tomatoes",
        choices=["rotten_tomatoes", "squad", "eli5"],
    )
    parser.add_argument("--checkpoint-folder", type=str, default="working_folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=5)

    args = parser.parse_args()
    main(
        pretrained_model_name=args.pretrained_model_name,
        dataset_name=args.dataset,
        s3url=args.checkpoint_folder,
        config=vars(args),
    )
