# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC
from pathlib import Path
from typing import Optional, Union

from renate import defaults
from renate.data.data_module import RenateDataModule


class TimeIncrementalDataModule(RenateDataModule, ABC):
    """Base class for all :py:class:`~renate.data.data_module.RenateDataModule` compatible with
    :py:class:`~renate.benchmark.scenarios.TimeIncrementalScenario`.

    Defines the API required by the :py:class:`~renate.benchmark.scenarios.TimeIncrementalScenario`.
    All classes extending this class must load the datasets corresponding to the value in
    ``time_step`` whenever ``setup()`` is called.

    Args:
        data_path: the path to the folder containing the dataset files.
        time_step: Time slice to be loaded.
        src_bucket: the name of the s3 bucket. If not provided, downloads the data from original
            source.
        src_object_name: the folder path in the s3 bucket.
        val_size: Fraction of the training data to be used for validation.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        time_step: Union[int, str],
        src_bucket: Optional[str] = None,
        src_object_name: Optional[str] = None,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super().__init__(
            data_path=data_path,
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            val_size=val_size,
            seed=seed,
        )
        self.time_step = time_step


class DomainIncrementalDataModule(RenateDataModule, ABC):
    """Base class for all :py:class:`~renate.data.data_module.RenateDataModule` compatible with
    :py:class:`~renate.benchmark.scenarios.DomainIncrementalScenario`.

    Defines the API required by the
    :py:class:`~renate.benchmark.scenarios.DomainIncrementalScenario`.
    All classes extending this class must load the datasets corresponding to the value in
    ``domain`` whenever ``setup()`` is called.

    Args:
        data_path: the path to the folder containing the dataset files.
        domain: Domain to be loaded.
        src_bucket: the name of the s3 bucket. If not provided, downloads the data from original
            source.
        src_object_name: the folder path in the s3 bucket.
        val_size: Fraction of the training data to be used for validation.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        domain: Union[int, str],
        src_bucket: Optional[str] = None,
        src_object_name: Optional[str] = None,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super().__init__(
            data_path=data_path,
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            val_size=val_size,
            seed=seed,
        )
        self.domain = domain
