# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Tuple

import torch
from avalanche.benchmarks import dataset_benchmark
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from renate.data.datasets import _TransformedDataset


class AvalancheSubset(Dataset):
    """Helper class to use dataset subsets with Avalanche.

    Since Avalanche does not support the Subset class, this class is used to convert subsets.
    Furthermore, some Avalanche updaters (e.g. iCaRL) directly access the targets attribute.

    Args:
        subset: The subset dataset that is converted.
    """

    def __init__(self, subset: Subset):
        super().__init__()
        x_data, y_data = [], []
        data_loader = DataLoader(subset)
        for x, y in data_loader:
            x_data.append(x)
            y_data.append(y.item())
        self.x: Tensor = torch.cat(x_data)
        self.y: List[int] = y_data
        self._targets: Optional[Tensor] = None

    @property
    def targets(self) -> Tensor:
        """Access to the dataset targets.

        Since not every Avalanche updater requires this attribute, we only create it when
        requested.
        """
        if self._targets is not None:
            return self._targets
        self._targets = torch.tensor(self.y, dtype=torch.long)
        return self._targets

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.y)


class AvalancheBenchmarkWrapper:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        train_transform,
        train_target_transform,
        test_transform,
        test_target_transform,
    ):
        self._n_classes_per_exp = None
        self._classes_order = None
        self._n_classes = 0
        self._train_dataset = train_dataset
        self._train_target_transform = train_target_transform
        self._benchmark = dataset_benchmark(
            [train_dataset],
            [val_dataset],
            train_transform=train_transform,
            train_target_transform=train_target_transform,
            eval_transform=test_transform,
            eval_target_transform=test_target_transform,
        )
        self.train_stream = self._benchmark.train_stream
        self.test_stream = self._benchmark.test_stream

    def update_benchmark_properties(self):
        dataset = _TransformedDataset(
            dataset=self._train_dataset, target_transform=self._train_target_transform
        )
        dataloader = DataLoader(dataset)
        unique_classes = set()
        for batch in dataloader:
            unique_classes.add(batch[1].item())
        if self._n_classes_per_exp is None:
            self._n_classes_per_exp = [len(unique_classes)]
            self._classes_order = list(sorted(unique_classes))
        else:
            self._n_classes_per_exp.append(len(unique_classes))
            self._classes_order += list(sorted(unique_classes))
        self._n_classes = sum(self._classes_order)
        self._benchmark.n_classes_per_exp = self._n_classes_per_exp
        self._benchmark.classes_order = self._classes_order

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the benchmark."""
        state_dict = {
            "n_classes_per_exp": self._n_classes_per_exp,
            "classes_order": self._classes_order,
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores the state of the benchmark."""
        self._n_classes_per_exp = state_dict["n_classes_per_exp"]
        self._classes_order = state_dict["classes_order"]
