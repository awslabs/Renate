# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from renate.utils.file import copy_to_uri, move_to_uri


def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


@pytest.mark.parametrize(
    "file_content1_source_dir, file_content2_source_dir, file_content1_destination_dir,"
    "file_content2_destination_dir",
    [["test1_1", "test1_1", "test1_2", "test2_2"]],
)
def test_move_to_uri_locally_directory(
    tmpdir,
    file_content1_source_dir,
    file_content2_source_dir,
    file_content1_destination_dir,
    file_content2_destination_dir,
):
    """Test for moving files from a local directory to another local directory.

    The files should be moved from the source directory and the destination directory should be
    created if it does not exist.

    If there are files with the same name in the destination directory as in the source directory
    they should be overwritten.
    """

    create_file(os.path.join(tmpdir, "source_dir", "file1.txt"), file_content1_source_dir)
    create_file(os.path.join(tmpdir, "source_dir", "file2.txt"), file_content2_source_dir)

    create_file(os.path.join(tmpdir, "destination_dir", "file1.txt"), file_content1_destination_dir)
    create_file(os.path.join(tmpdir, "destination_dir", "file3.txt"), file_content2_destination_dir)

    move_to_uri(os.path.join(tmpdir, "source_dir"), os.path.join(tmpdir, "destination_dir"))

    with open(os.path.join(tmpdir, "destination_dir", "file1.txt"), "r") as f:
        assert f.read() == file_content1_source_dir
    with open(os.path.join(tmpdir, "destination_dir", "file2.txt"), "r") as f:
        assert f.read() == file_content2_source_dir
    with open(os.path.join(tmpdir, "destination_dir", "file3.txt"), "r") as f:
        assert f.read() == file_content2_destination_dir

    assert not os.path.exists(os.path.join(tmpdir, "source_dir", "file1.txt"))
    assert not os.path.exists(os.path.join(tmpdir, "source_dir", "file2.txt"))


def test_move_to_uri_locally_file(tmpdir):
    """Test for moving a file to another local directory.

    The file should be moved from the source directory and the destination directory should be
    created if it does not exist.

    If there are files with the same name in the destination directory as in the source directory
    they should be overwritten.
    """

    file_content = "content"
    create_file(os.path.join(tmpdir, "source_dir", "file.txt"), file_content)

    move_to_uri(
        os.path.join(tmpdir, "source_dir", "file.txt"), os.path.join(tmpdir, "destination_dir")
    )
    with open(os.path.join(tmpdir, "destination_dir", "file.txt"), "r") as f:
        assert f.read() == file_content
    assert not os.path.exists(os.path.join(tmpdir, "source_dir", "file.txt"))

    file_content = "content2"
    create_file(os.path.join(tmpdir, "source_dir", "file.txt"), file_content)
    move_to_uri(
        os.path.join(tmpdir, "source_dir", "file.txt"), os.path.join(tmpdir, "destination_dir")
    )
    with open(os.path.join(tmpdir, "destination_dir", "file.txt"), "r") as f:
        assert f.read() == file_content
    assert not os.path.exists(os.path.join(tmpdir, "source_dir", "file.txt"))


@pytest.mark.parametrize(
    "file_content1_source_dir, file_content2_source_dir, file_content1_destination_dir, "
    "file_content2_destination_dir",
    [["test1_1", "test1_1", "test1_2", "test2_2"]],
)
def test_copy_to_uri_locally_directory(
    tmpdir,
    file_content1_source_dir,
    file_content2_source_dir,
    file_content1_destination_dir,
    file_content2_destination_dir,
):
    """Test for copying files from a local directory to another local directory.

    The files should be copied from the source directory and the destination directory should be
    created if it does not exist.

    If there are files with the same name in the destination directory as in the source directory
    they should be overwritten.

    The source directory should not be changed.
    """

    create_file(os.path.join(tmpdir, "source_dir", "file1.txt"), file_content1_source_dir)
    create_file(os.path.join(tmpdir, "source_dir", "file2.txt"), file_content2_source_dir)

    create_file(os.path.join(tmpdir, "destination_dir", "file1.txt"), file_content1_destination_dir)
    create_file(os.path.join(tmpdir, "destination_dir", "file3.txt"), file_content2_destination_dir)

    copy_to_uri(os.path.join(tmpdir, "source_dir"), os.path.join(tmpdir, "destination_dir"))

    with open(os.path.join(tmpdir, "destination_dir", "file1.txt"), "r") as f:
        assert f.read() == file_content1_source_dir
    with open(os.path.join(tmpdir, "destination_dir", "file2.txt"), "r") as f:
        assert f.read() == file_content2_source_dir
    with open(os.path.join(tmpdir, "destination_dir", "file3.txt"), "r") as f:
        assert f.read() == file_content2_destination_dir

    with open(os.path.join(tmpdir, "source_dir", "file1.txt"), "r") as f:
        assert f.read() == file_content1_source_dir
    with open(os.path.join(tmpdir, "source_dir", "file2.txt"), "r") as f:
        assert f.read() == file_content2_source_dir


def test_copy_to_uri_locally_file(tmpdir):
    """Test for copying a file from a local directory to another local directory.

    The file should be copied from the source directory and the destination directory should be
    created if it does not exist.

    If there are files with the same name in the destination directory as in the source directory
    they should be overwritten.

    The source directory should not be changed.
    """
    file_content = "content"
    create_file(os.path.join(tmpdir, "source_dir", "file.txt"), file_content)

    copy_to_uri(
        os.path.join(tmpdir, "source_dir", "file.txt"), os.path.join(tmpdir, "destination_dir")
    )
    with open(os.path.join(tmpdir, "destination_dir", "file.txt"), "r") as f:
        assert f.read() == file_content
    with open(os.path.join(tmpdir, "source_dir", "file.txt"), "r") as f:
        assert f.read() == file_content
