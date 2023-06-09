from argparse import ArgumentParser
from typing import Any, Optional

import torch
import transformers
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from transformer import HFSequenceClassificationTransformerWithLora

from renate.benchmark.datasets.nlp_datasets import HuggingFaceTextDataModule


def loss_fn():
    return torch.nn.CrossEntropyLoss()


def dataset_fn(module_name):
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(module_name)
    data_module = HuggingFaceTextDataModule(
        "data", dataset_name="rotten_tomatoes", tokenizer=tokenizer, val_size=0.2
    )

    rank_zero_only(data_module.prepare_data)()
    data_module.setup()

    return data_module


class MyLightningModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = HFSequenceClassificationTransformerWithLora(args.module_name, 2, 4, 0)
        self.loss_fn = loss_fn()

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx: int):
        inputs, targets = batch
        outputs = self(**batch)
        loss = self._loss_fn(outputs, targets)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        inputs, targets = batch
        outputs = self(**batch)

        return outputs


def training_function(args):
    # we need module, data and loss
    model = HFSequenceClassificationTransformerWithLora(args.module_name, 2, 4, 0)
    data_module = dataset_fn(module_name=args.module_name)

    train_data, val_data = data_module.train_data(), data_module.val_data()

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model, train_loader, val_loader)


def main():
    parser = ArgumentParser()
    parser.add_argument("--module-name", default="distilbert-base-uncased", type=str)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    training_function(args)


if __name__ == "__main__":
    main()
