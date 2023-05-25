# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple, Union

import torch.nn as nn

from renate.benchmark.models.base import RenateBenchmarkingModule
from renate.models.prediction_strategies import PredictionStrategy


class MultiLayerPerceptron(RenateBenchmarkingModule):
    """A simple Multi Layer Perceptron with hidden layers, activation and Batch Normalization if
    enabled.

    Args:
        num_inputs: Number of input nodes.
        num_outputs: Number of output nodes.
        num_hidden_layers: Number of hidden layers.
        hidden_size: Uniform hidden size or the list or tuple of hidden sizes for individual hidden
            layers.
        activation: Activation name, matching activation name in `torch.nn` to be used between the
            hidden layers.
        batch_normalization: Whether to use Batch Normalization after the activation. By default the
            Batch Normalization tracks the running statistics.
        prediction_strategy: Continual learning strategies may alter the prediction at train or test
            time.
        add_icarl_class_means: If ``True``, additional parameters used only by the
            ``ICaRLModelUpdater`` are added. Only required when using that updater.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_hidden_layers: int,
        hidden_size: Union[int, List[int], Tuple[int]],
        activation: str = "ReLU",
        batch_normalization: bool = False,
        prediction_strategy: Optional[PredictionStrategy] = None,
        add_icarl_class_means: bool = True,
    ) -> None:
        embedding_size = hidden_size if type(hidden_size) == int else hidden_size[-1]
        super().__init__(
            embedding_size=embedding_size,
            num_outputs=num_outputs,
            constructor_arguments={
                "num_inputs": num_inputs,
                "num_hidden_layers": num_hidden_layers,
                "hidden_size": hidden_size,
                "activation": activation,
                "batch_normalization": batch_normalization,
            },
            prediction_strategy=prediction_strategy,
            add_icarl_class_means=add_icarl_class_means,
        )
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size for _ in range(num_hidden_layers + 1)]
        assert len(hidden_size) == num_hidden_layers + 1

        activation = getattr(nn, activation)

        hidden_size = [num_inputs] + hidden_size
        layers = [nn.Flatten()]
        for i in range(num_hidden_layers + 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            layers.append(activation())
            if batch_normalization:
                layers.append(nn.BatchNorm1d(hidden_size[i + 1]))

        self._backbone = nn.Sequential(*layers)
