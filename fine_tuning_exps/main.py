# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import transformers
from model_utils import make_model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from training_args import get_arguments
from transformers import DataCollatorForLanguageModeling, default_data_collator

from renate import defaults
from renate.benchmark.datasets.nlp_datasets import (
    HuggingFaceExtractiveQADataModule,
    HuggingFaceLanguageModelingModule,
    HuggingFaceTextDataModule,
)
from renate.benchmark.models.transformer import (
    HuggingFaceQuestionAnsweringTransformer,
    HuggingFaceSequenceClassificationTransformer,
)
from renate.data.data_module import RenateDataModule
from renate.updaters.peft_learner import PeftLearner, QAPeft
from renate.utils.distributed_strategies import create_strategy
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
    args: Namespace, metric: Optional[str] = None, mode: defaults.SUPPORTED_TUNING_MODE_TYPE = "min"
):
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=args.default_root_dir,
        monitor=metric,
        mode=mode,
    )
    callbacks = [model_checkpoint_callback]
    return Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        precision=int_or_str(args.precision),
        strategy=create_strategy(strategy_name=args.strategy, devices=int_or_str(args.devices)),
        replace_sampler_ddp=False,
        # val_check_interval=1
    )


def main(
    args: Namespace,
):  #: pretrained_model_name: str, dataset_name: str, s3url: str, config: Dict):
    datapath = Path(args.default_root_dir or "./data")
    datapath.mkdir(exist_ok=True)

    checkpointpath = datapath / "checkpoint"
    checkpointpath.mkdir(exist_ok=True)

    ## load trainer first to spawn before instantiation.
    trainer = get_trainer(args)

    # Setup data
    data_module = get_data(args.dataset, args.pretrained_model_name, datapath)
    if args.dataset == "rotten_tomatoes":
        num_outputs = 2
        model = HuggingFaceSequenceClassificationTransformer(
            pretrained_model_name=args.pretrained_model_name, num_outputs=num_outputs
        )
        lossfunction = torch.nn.CrossEntropyLoss()
        collator_fn = None
        module_cls = PeftLearner

    elif args.dataset == "squad":
        model = HuggingFaceQuestionAnsweringTransformer(
            pretrained_model_name=args.pretrained_model_name
        )
        lossfunction = lambda x, y: x
        collator_fn = default_data_collator
        module_cls = QAPeft

    elif args.dataset == "eli5":
        model = make_model(
            pretrained_model_name=args.pretrained_model_name,
            causal=not args.mlm,
            quantize=args.quantize,
            alpha=args.alpha,
        )
        lossfunction = lambda x, y: x
        data_module._tokenizer.pad_token = data_module._tokenizer.eos_token
        collator_fn = DataCollatorForLanguageModeling(
            mlm=args.mlm, tokenizer=data_module._tokenizer
        )
        module_cls = QAPeft

    # Specify your optimizer params here
    optimizer = partial(torch.optim.AdamW, lr=args.lr, weight_decay=1e-2)
    module = module_cls(
        model,
        loss_fn=lossfunction,
        optimizer=optimizer,
        batch_size=args.batch_size,
        logged_metrics={},
    )

    module.on_model_update_start(
        train_dataset=data_module.train_data(),
        val_dataset=data_module.val_data(),
        train_dataset_collate_fn=collator_fn,
        val_dataset_collate_fn=collator_fn,
    )
    # Fit the model
    trainer.fit(module)

    # Uplaod checkpoints to s3
    # upload_folder_to_s3(checkpointpath, s3url)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
