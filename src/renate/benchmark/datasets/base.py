# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC
from pathlib import Path
from typing import Optional, Union

from renate import defaults
from renate.data.data_module import RenateDataModule


class DataIncrementalDataModule(RenateDataModule, ABC):
    """Base class for all :py:class:`~renate.data.data_module.RenateDataModule` compatible with
    :py:class:`~renate.benchmark.scenarios.DataIncrementalScenario`.

    Defines the API required by the :py:class:`~renate.benchmark.scenarios.DataIncrementalScenario`.
    All classes extending this class must load the datasets corresponding to the value in
    ``data_id`` whenever ``setup()`` is called.

    Args:
        data_path: the path to the folder containing the dataset files.
        data_id: Time slice to be loaded.
        src_bucket: the name of the s3 bucket. If not provided, downloads the data from original
            source.
        src_object_name: the folder path in the s3 bucket.
        val_size: Fraction of the training data to be used for validation.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        data_id: Union[int, str],
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
        self.data_id = data_id
