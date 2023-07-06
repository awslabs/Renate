from argparse import ArgumentParser
from pytorch_lightning import Trainer


def lightning_args(parser: ArgumentParser):
    return Trainer.add_argparse_args(parser)


def model_args(parser: ArgumentParser):
    parser.add_argument("--pretrained-model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--enable-gradient-checkpointing", action="store_true")
    parser.add_argument("--mlm", action="store_true")
    parser.add_argument("--quantize", action="store_true")


def lora_args(parser: ArgumentParser):
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--r", type=int, default=8)


def data_args(parser: ArgumentParser):
    parser.add_argument(
        "--dataset", type=str, default="eli5", choices=["rotten_tomatoes", "squad", "eli5"]
    )


def training_hp_args(parser: ArgumentParser):
    parser.add_argument("--checkpoint-folder", type=str, default="working_folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=10**-5)


def get_arguments():
    parser = ArgumentParser()
    lightning_args(parser)
    lora_args(parser)
    data_args(parser)
    model_args(parser)
    training_hp_args(parser)
    return parser.parse_args()
