from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torchvision.transforms import transforms

from renate.benchmark.models.mlp import MultiLayerPerceptron

from renate.models import RenateModule

from renate import defaults
from renate.benchmark.datasets.vision_datasets import TorchVisionDataModule
from renate.benchmark.scenarios import ClassIncrementalScenario, Scenario


def data_module_fn(
    data_path: Union[Path, str], chunk_id: int, seed: int = defaults.SEED
) -> Scenario:
    """Returns a class-incremental scenario instance.

    The transformations passed to prepare the input data are required to convert the data to
    PyTorch tensors.
    """
    data_module = TorchVisionDataModule(
        str(data_path),
        dataset_name="MNIST",
        val_size=0.1,
        seed=seed,
    )

    class_incremental_scenario = ClassIncrementalScenario(
        data_module=data_module,
        class_groupings=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        chunk_id=chunk_id,
    )
    return class_incremental_scenario


def model_fn(model_state_url: Optional[Union[Path, str]] = None) -> RenateModule:
    """Returns a model instance."""
    if model_state_url is None:
        model = MultiLayerPerceptron(
            num_inputs=784, num_outputs=10, num_hidden_layers=2, hidden_size=128
        )
    else:
        state_dict = torch.load(str(model_state_url))
        model = MultiLayerPerceptron.from_state_dict(state_dict)
    return model


def train_transform() -> Callable:
    """Returns a transform function to be used in the training."""
    return transforms.Lambda(lambda x: torch.flatten(x))
