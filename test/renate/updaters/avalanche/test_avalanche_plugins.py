# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from renate import defaults
from renate.updaters.avalanche.plugins import RenateFileSystemCheckpointStorage


def test_renate_file_system_checkpoint_storage_plugin(tmpdir):
    expected_directory = tmpdir
    checkpoint_name = "checkpoint_name"
    plugin = RenateFileSystemCheckpointStorage(directory=expected_directory)
    assert plugin._make_checkpoint_dir(checkpoint_name) == expected_directory
    assert defaults.learner_state_file(expected_directory) == str(
        plugin._make_checkpoint_file_path(checkpoint_name)
    )
    assert not plugin.checkpoint_exists(checkpoint_name)
    Path(defaults.learner_state_file(expected_directory)).touch()
    assert plugin.checkpoint_exists(checkpoint_name)
