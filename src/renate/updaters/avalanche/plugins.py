# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from avalanche.training.plugins.checkpoint import CheckpointPlugin, FileSystemCheckpointStorage

from renate import defaults


class RenateFileSystemCheckpointStorage(FileSystemCheckpointStorage):
    def _make_checkpoint_dir(self, checkpoint_name: str) -> Path:
        return self.directory

    def _make_checkpoint_file_path(self, checkpoint_name: str) -> Path:
        return Path(defaults.learner_state_file(str(self._make_checkpoint_dir(checkpoint_name))))

    def checkpoint_exists(self, checkpoint_name: str) -> bool:
        return self._make_checkpoint_file_path(checkpoint_name).exists()

    def load_checkpoint(self, checkpoint_name: str, checkpoint_loader) -> Any:
        checkpoint_file = self._make_checkpoint_file_path(checkpoint_name)
        with open(checkpoint_file, "rb") as f:
            return checkpoint_loader(f)


class RenateCheckpointPlugin(CheckpointPlugin):
    def __init__(
        self,
        storage: RenateFileSystemCheckpointStorage,
        # metric: str,
        # mode: defaults.SUPPORTED_TUNING_MODE_TYPE,
        map_location: Optional[Union[str, torch.device, Dict[str, str]]] = None,
    ):
        super().__init__(storage=storage, map_location=map_location)

    def load_checkpoint_if_exists(self):
        if not self.storage.checkpoint_exists(defaults.LEARNER_CHECKPOINT_NAME):
            return None, 0
        loaded_checkpoint = self.storage.load_checkpoint(
            defaults.LEARNER_CHECKPOINT_NAME, self.load_checkpoint
        )
        return loaded_checkpoint["strategy"], loaded_checkpoint["exp_counter"]
