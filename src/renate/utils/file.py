# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse
from zipfile import ZipFile

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)


def get_aws_region() -> str:
    """Returns the name of the AWS region used during the execution."""
    return boto3.Session().region_name or "us-west-2"


def get_bucket() -> str:
    """Returns the default S3 bucket."""
    aws_account = boto3.client("sts").get_caller_identity().get("Account")
    return f"sagemaker-{get_aws_region()}-{aws_account}"


def is_s3_uri(uri: str) -> bool:
    """Checks if the uri is an S3 uri."""
    return urlparse(uri).scheme == "s3"


def _parse_s3_url(s3_url: str) -> Tuple[str, str]:
    parsed_url = urlparse(s3_url)
    if parsed_url.scheme == "s3":
        return parsed_url.netloc, parsed_url.path[1:]
    raise ValueError(f"{s3_url} is not an S3 URL.")


def _move_locally(
    src: Union[str, Path],
    dst: Union[str, Path],
    ignore_extensions: List[str] = [".sagemaker-uploading", ".sagemaker-uploaded"],
    copy: bool = False,
) -> None:
    """Moves files in directory or file to directory. If the files exist they are overwritten.

    Args:
        src: Source directory or file.
        dst: Target directory or file.
        ignore_extensions: List of extensions to ignore.
        copy: If `True`, copy instead of move.
    """
    if os.path.isfile(src):
        os.makedirs(dst, exist_ok=True)
        dst_file = os.path.join(dst, os.path.basename(src))
        if os.path.exists(dst_file):
            os.remove(dst_file)
        if copy:
            shutil.copy(src, dst)
        else:
            shutil.move(src, dst)
    for src_dir, _, files in os.walk(src):
        dst_dir = src_dir.replace(src, dst, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            if f.endswith(tuple(ignore_extensions)):
                continue
            src_f = os.path.join(src_dir, f)
            dst_f = os.path.join(dst_dir, f)
            if os.path.exists(dst_f):
                os.remove(dst_f)
            if copy:
                shutil.copy(src_f, dst_f)
            else:
                shutil.move(src_f, dst_f)


def _move_to_s3(
    src: Union[str, Path],
    dst: Union[str, Path],
    ignore_extensions: List[str] = [".sagemaker-uploading", ".sagemaker-uploaded"],
    copy: bool = False,
) -> None:
    """Moves files in directory or file to directory or s3.

    If the files exist they are overwritten. The files in the local directory are deleted.

    Args:
        src: Local file or directory to move.
        dst: Target directory or s3 uri.
        ignore_extensions: List of extensions to ignore.
        copy: If `True`, copy instead of move.
    """
    if os.path.isfile(src):
        dst_file = os.path.join(dst, os.path.basename(src))
        upload_file_to_s3(src, dst_file)
        if not copy:
            os.remove(src)
    else:
        upload_folder_to_s3(src, dst, ignore_extensions=ignore_extensions)
        if not copy:
            shutil.rmtree(src)


def _move_to_uri(
    src: Union[Path, str],
    dst: str,
    ignore_extensions: List[str] = [".sagemaker-uploading", ".sagemaker-uploaded"],
    copy: bool = False,
) -> None:
    """Moves files in directory or file to directory or s3.

    If the files exist they are overwritten. The files in the local directory are deleted.

    Args:
        src: Local file or directory to move.
        dst: Target directory or s3 uri.
        ignore_extensions: List of extensions to ignore.
        copy: If `True`, copy instead of move.
    """
    if is_s3_uri(dst):
        _move_to_s3(src, dst, ignore_extensions=ignore_extensions, copy=copy)
    elif src != dst:
        _move_locally(src, dst, ignore_extensions=ignore_extensions, copy=copy)
    else:
        logging.warning(f"Source and destination are the same: {src}")


def move_to_uri(
    src: Union[Path, str],
    dst: str,
    ignore_extensions: List[str] = [".sagemaker-uploading", ".sagemaker-uploaded"],
) -> None:
    """Moves files in directory or file to directory or s3.

    If the files exist they are overwritten. The files in the local directory are deleted.

    Args:
        src: Local file or directory to move.
        dst: Target directory or s3 uri.
        ignore_extensions: List of extensions to ignore.
    """
    _move_to_uri(src=src, dst=dst, ignore_extensions=ignore_extensions, copy=False)


def copy_to_uri(
    src: Union[Path, str],
    dst: str,
    ignore_extensions: List[str] = [".sagemaker-uploading", ".sagemaker-uploaded"],
) -> None:
    """Copies files in directory or file to directory or s3.

    If the files exist they are overwritten. The files in the local directory are preserved.

    Args:
        src: Local directory to copy.
        dst: Target directory or s3 uri.
        ignore_extensions: List of extensions to ignore.
    """
    _move_to_uri(src=src, dst=dst, ignore_extensions=ignore_extensions, copy=True)


def maybe_download_from_s3(url: str, local_dir: Union[Path, str]) -> str:
    """Tries to download a file from S3."""
    try:
        src_bucket, src_object_name = _parse_s3_url(url)
        url = str(local_dir)
        download_folder_from_s3(
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            dst_dir=url,
        )
    except ValueError:
        pass
    return url


def download_folder_from_s3(
    src_bucket: str, src_object_name: Union[Path, str], dst_dir: Union[Path, str]
) -> None:
    """Downloads folder from S3 to local disk."""
    src_object_name = str(Path(src_object_name))
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(src_bucket)
    for obj in bucket.objects.filter(Prefix=src_object_name):
        dst_file = os.path.join(dst_dir, obj.key[len(src_object_name) + 1 :])
        if obj.key[-1] == "/":
            continue
        download_file_from_s3(src_bucket, obj.key, dst_file)


def upload_folder_to_s3(
    local_dir: Union[Path, str],
    s3_url: Optional[Union[Path, str]] = None,
    dst_bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    ignore_extensions: List[str] = [".sagemaker-uploading", ".sagemaker-uploaded"],
) -> None:
    """Uploads all files within a local folder to s3.

    Args:
        local_dir: Folder containing files to be uploaded.
        s3_url: Full path to s3 location.
        dst_bucket: s3 bucket.
        prefix: Prefix for all s3 object names.
        ignore_extensions: List of extensions to ignore.
    """
    assert (
        s3_url is not None or dst_bucket is not None and prefix is not None
    ), "Either pass s3_url or both dst_bucket and prefix."
    if s3_url is not None:
        dst_bucket, prefix = _parse_s3_url(s3_url)
    local_dir = str(Path(local_dir))
    for current_folder, folders, files in os.walk(local_dir):
        for file_name in files:
            if file_name.endswith(tuple(ignore_extensions)):
                continue
            file_path = os.path.join(current_folder, file_name)
            object_name = os.path.join(prefix, current_folder[len(local_dir) + 1 :], file_name)
            upload_file_to_s3(file_path, dst_bucket=dst_bucket, dst_object_name=object_name)


def download_file_from_s3(
    src_bucket: str, src_object_name: Union[Path, str], dst: Union[Path, str]
) -> None:
    """Downloads file from S3 to local disk

    Args:
        src_bucket: Source S3 bucket
        src_object_name: Source S3 object
        dst: local destination
    """
    if isinstance(dst, str):
        dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client("s3")
    logger.info(f"Download file from s3://{src_bucket}/{src_object_name} to {dst}")
    with open(dst, "wb") as f:
        s3_client.download_fileobj(src_bucket, str(src_object_name), f)


def upload_file_to_s3(
    src: Union[Path, str],
    s3_url: Optional[Union[Path, str]] = None,
    dst_bucket: Optional[str] = None,
    dst_object_name: Optional[Union[Path, str]] = None,
) -> bool:
    """Upload a file to an S3 bucket

    Args:
        src: File to upload.
        s3_url: Full path to s3 location.
        dst_bucket: Destination S3 bucket
        dst_object_name: Destination S3 object

    Return:
        True if file was uploaded, else False
    """
    assert (
        s3_url is not None or dst_bucket is not None and dst_object_name is not None
    ), "Either pass s3_url or both dst_bucket and dst_object_name."
    if s3_url is not None:
        dst_bucket, dst_object_name = _parse_s3_url(s3_url)
    s3_client = boto3.client("s3")
    logger.info(f"Upload file from {src} to s3://{dst_bucket}/{dst_object_name}")
    try:
        s3_client.upload_file(str(src), dst_bucket, str(dst_object_name))
    except ClientError as e:
        logging.error(e)
        return False
    return True


def delete_file_from_s3(bucket: str, object_name: str) -> None:
    """Delete file from the S3 bucket

    Args:
        bucket: bucket in which the object (file) is stored
        object_name: object to be deleted
    """
    s3_client = boto3.client("s3")
    s3_client.delete_object(Bucket=bucket, Key=str(object_name))


def unzip_file(dataset_name: str, data_path: Union[str, Path], file_name: str) -> None:
    """Extract .zip files into folder named with dataset name."""
    with ZipFile(os.path.join(data_path, dataset_name, file_name)) as f:
        f.extractall(os.path.join(data_path, dataset_name))


def download_file(
    dataset_name: str,
    data_path: Union[str, Path],
    src_bucket: str,
    src_object_name: str,
    url: str,
    file_name: str,
) -> None:
    """A helper function to download data from URL or s3."""
    if src_bucket is None:
        if not os.path.exists(os.path.join(data_path, dataset_name)):
            os.makedirs(os.path.join(data_path, dataset_name))
        with requests.get(os.path.join(url, file_name), allow_redirects=True, stream=True) as r:
            r.raise_for_status()
            with open(os.path.join(data_path, dataset_name, file_name), "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        download_file_from_s3(
            src_bucket,
            os.path.join(src_object_name, file_name),
            os.path.join(data_path, dataset_name, file_name),
        )


def download_and_unzip_file(
    dataset_name: str,
    data_path: Union[str, Path],
    src_bucket: str,
    src_object_name: str,
    url: str,
    file_name: str,
) -> None:
    """A helper function to download data .zips and uncompress them."""
    download_file(dataset_name, data_path, src_bucket, src_object_name, url, file_name)
    unzip_file(dataset_name, data_path, file_name)


def save_pandas_df_to_csv(df: pd.DataFrame, file_path: Union[str, Path]) -> pd.DataFrame:
    """A helper function to save pandas dataframe to a .csv.

    It guarantees that the saved dataframes across Renate are consistent.
    """
    df.to_csv(file_path, index=False)
    return df


@rank_zero_only
def unlink_file_or_folder(path: Path) -> None:
    """Function to remove files and folders.

    Unlink works for files, rmdir for empty folders, but not for non-empty ones. Hence a
    recursive solution.
    """
    if path.exists():
        if path.is_file():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path)
