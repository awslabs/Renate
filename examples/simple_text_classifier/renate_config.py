from pathlib import Path
from typing import Optional, Union

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from renate import defaults
from renate.benchmark.models.mlp import MultiLayerPerceptron
from renate.benchmark.scenarios import Scenario
from renate.data.data_module import RenateDataModule
from renate.models import RenateModule

# this is a hack but if it works we should generalize it and
# move it into the benchmark folder
class HFDataModule(RenateDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        src_bucket: Optional[str] = None,
        src_object_name: Optional[str] = None,
        dataset_name: str = "glue-mrpc",
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super(HFDataModule, self).__init__(
            data_path,
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            val_size=val_size,
            seed=seed,
        )

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        train = load_dataset("glue", "mrpc", split="train")
        test = load_dataset("glue", "mrpc", split="test")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        train = train.map(lambda e: tokenizer(e["sentence1"]), batched=True)
        test = test.map(lambda e: tokenizer(e["sentence1"]), batched=True)

        train.set_format(type="torch", columns=["input_ids", "label"])

        test.set_format(type="torch", columns=["input_ids", "label"])

        self._train_data, self._val_data = self._split_train_val_data(train)
        self._test_data = test


def data_module_fn(
    data_path: Union[Path, str], chunk_id: int, seed: int = defaults.SEED
) -> Scenario:
    """Returns a class-incremental scenario instance.
    The transformations passed to prepare the input data are required to convert the data to
    PyTorch tensors.
    """

    data_module = HFDataModule(data_path, val_size=0.1, seed=seed)

    # this is not the expected format -- yet!
    return data_module


def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    """Returns a model instance."""
    # TODO replace this with something more useful, eg pre-trained Roberta
    if model_state_url is None:
        model = MultiLayerPerceptron(
            num_inputs=256, num_outputs=4, num_hidden_layers=2, hidden_size=64
        )
    else:
        state_dict = torch.load(str(model_state_url))
        model = MultiLayerPerceptron.from_state_dict(state_dict)
    return model
