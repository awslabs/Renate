# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torchmetrics
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader, Dataset

from renate import defaults
from renate.data.datasets import _TransformedDataset
from renate.evaluation.metrics.utils import create_metrics
from renate.models import RenateModule


class Evaluator(LightningModule, abc.ABC):
    """A general Evaluator module for collection of quantitative metrics on the test dataset.

    This is an abstract interface which can be called with respect to a PyTorch Lightning `Trainer`.
    and its `.test()` function. It collects quantitative observations with respect to a single
    dataset. The metrics that are being collected are defined in the `create_metrics` function.

    Args:
        model: A `RenateModule` to be evaluated.
        task: The machine learning problem considered.
        batch_size: The batch size to be used when creating the test data loader.
        transform: The transformation applied for evaluation.
        target_transform: The target transformation applied for evaluation.
        logged_metrics: Metrics logged additional to the default ones.
    """

    def __init__(
        self,
        model: RenateModule,
        task: defaults.SUPPORTED_TASKS_TYPE,
        batch_size: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._model.deregister_hooks()
        self._batch_size = batch_size
        self._transform = transform
        self._target_transform = target_transform
        self._metric_collection = create_metrics(task=task, additional_metrics=logged_metrics)

    def on_model_test_start(
        self, test_dataset: Dataset, task_id: Optional[str] = None
    ) -> DataLoader:
        """Called before a model test starts."""
        test_dataset = _TransformedDataset(
            test_dataset,
            transform=self._transform,
            target_transform=self._target_transform,
        )
        self._task_id = task_id
        return DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, pin_memory=True)

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """PyTorch Lightning function to perform the test step."""
        x, y = batch
        outputs = self(x)
        self._metric_collection(outputs, y)

    @abc.abstractmethod
    def forward(self, x, task_id: Optional[str] = None) -> torch.Tensor:
        """Forward pass of the model.

        Task ID can be used to specify, for example, the output head to perform the evaluation with
        a specific data Chunk ID. Here, the `task_id` is used only to compute the test metrics.
        """
        pass

    def on_test_epoch_end(self) -> None:
        """PyTorch Lightning function to perform at the end of test loop.

        Logs the metrics and resets the metric collection.
        """
        self.log_dict(self._metric_collection.compute(), on_step=False, on_epoch=True)
        self._metric_collection.reset()


class ClassificationEvaluator(Evaluator):
    """A classification Evaluator module for collection of quantitative metrics on the test
    dataset.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(task="classification", **kwargs)

    def forward(self, x, task_id: Optional[str] = None) -> torch.Tensor:
        """Forward pass of the model.

        Task ID can be used to specify, for example, the output head to perform the evaluation with
        a specific data Chunk ID. Here, the `task_id` is used only to compute the test metrics.
        """
        if task_id is None:
            task_id = self._task_id
        return self._model.get_logits(x, task_id=task_id)


def evaluate(
    model: RenateModule,
    test_dataset: Union[List[Dataset], Dataset],
    task_id: Union[List[str], str] = defaults.TASK_ID,
    batch_size: int = defaults.BATCH_SIZE,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    logged_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    logger: Logger = defaults.LOGGER(**defaults.LOGGER_KWARGS),
    accelerator: defaults.SUPPORTED_ACCELERATORS_TYPE = defaults.ACCELERATOR,
    devices: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Evaluate the model on the test dataset or a set of test datasets corresponding to distinct
    tasks.

    If the `test_dataset` are specified as a list of datasets, it is assumed to be ordered.
    Similarly, in a case the `task_id` are specified as a list, it is assumed to be ordered. A task
    ID list can be used to set specific model part to be used, for example, an output head with some
    specific test dataset in the input sequence.

    Args:
        model: A `RenateModule` to be evaluated.
        test_dataset: The test dataset(s) to be evaluated.
        task_id: The task id(s) of the test dataset(s).
        batch_size: The batch size to be used when creating the test data loader.
        transform: The transformation applied for evaluation.
        target_transform: The target transformation applied for evaluation.
        logged_metrics: Metrics logged additional to the default ones.
        logger: Logger used by PyTorch Lightning to log intermediate results.
        accelerator: Accelerator used by PyTorch Lightning to train the model.
        devices: Devices used by PyTorch Lightning to train the model. If the devices flag is not
            defined, it will assume devices to be "auto" and fetch the `auto_device_count` from the
            `accelerator`.
    """
    if isinstance(test_dataset, Dataset):
        test_dataset = [test_dataset]

    if isinstance(task_id, str):
        task_id = [task_id] * len(test_dataset)

    assert len(task_id) == len(test_dataset)

    evaluator = ClassificationEvaluator(
        model=model,
        batch_size=batch_size,
        transform=transform,
        target_transform=target_transform,
        logged_metrics=logged_metrics,
    )

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    results = {}
    for i in range(len(test_dataset)):
        test_loader = evaluator.on_model_test_start(test_dataset[i], task_id[i])
        trainer.test(
            evaluator,
            test_loader,
        )
        for metric_name, value in trainer.logged_metrics.items():
            if metric_name not in results:
                results[metric_name] = []
            results[metric_name].append(value.item())
    return results
