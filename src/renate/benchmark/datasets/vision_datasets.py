# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import torchvision
from torchvision import transforms

from renate import defaults
from renate.data import ImageDataset
from renate.data.data_module import RenateDataModule
from renate.utils.file import download_and_unzip_file, download_folder_from_s3


class TinyImageNetDataModule(RenateDataModule):
    """Datamodule that process TinyImageNet dataset.

    Source: http://cs231n.stanford.edu/

    The TinyImageNet dataset is a subset of the ImageNet dataset. It contains 200 classes, each with
    500 training images, 50 validation images with labels, and 50 test images without labels. The images are 3x64x64 colored images.
    Note that, in our setting the validation set is used as the test set. The original test set is unused.

    Args:
        data_path: Path to the directory where the dataset should be stored.
        src_bucket: Name of the bucket where the dataset is stored.
        src_object_name: Name of the object in the bucket where the dataset is stored.
        transform: Transform to be applied to the dataset.
        target_transform: Transform to be applied to the target.
        val_size: Proportion of the training set to be used for validation.
        seed: Seed to be used for splitting the dataset.
    """

    md5s = {"tiny-imagenet-200.zip": "90528d7ca1a48142e341f4ef8d21d0de"}

    def __init__(
        self,
        data_path: Union[Path, str],
        src_bucket: Optional[str] = None,
        src_object_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ) -> None:
        super(TinyImageNetDataModule, self).__init__(
            data_path=data_path,
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            val_size=val_size,
            seed=seed,
        )
        self._transform = transform
        self._target_transform = target_transform
        self._dataset_name = "tiny_imagenet"

    def prepare_data(self) -> None:
        """Download the TinyImageNet dataset."""
        if not self._verify_file("tiny-imagenet-200.zip"):
            download_and_unzip_file(
                self._dataset_name,
                self._data_path,
                self._src_bucket,
                self._src_object_name,
                "http://cs231n.stanford.edu/",
                "tiny-imagenet-200.zip",
            )

    def setup(self, stage: Optional[Literal["start", "val", "test"]] = None) -> None:
        """Make assignments: train/val/test splits.

        The validation split in this setting is taken as the test split.
        """
        # Assign train dataset
        if stage in ["train", "val"] or stage is None:
            X, y = self._preprocess_tiny_imagenet(train=True)
            train_data = ImageDataset(X, y, self._transform, self._target_transform)
            self._train_data, self._val_data = self._split_train_val_data(train_data)

        # Assign test dataset
        if stage == "test" or stage is None:
            X, y = self._preprocess_tiny_imagenet(train=False)
            self._test_data = ImageDataset(
                X, y, transform=self._transform, target_transform=self._target_transform
            )

    def _preprocess_tiny_imagenet(self, train: bool) -> Tuple[List[str], List[int]]:
        """A helper function to preprocess the TinyImageNet dataset."""
        if train:
            data_path = os.path.join(self._data_path, self._dataset_name, "tiny-imagenet-200")
        else:
            data_path = os.path.join(self._data_path, self._dataset_name, "tiny-imagenet-200")
        label_encoding = {}
        with open(os.path.join(data_path, "wnids.txt"), "r") as f:
            label_encoding = {line.strip(): i for i, line in enumerate(f.readlines())}
        X = []
        y = []
        if train:
            for label in label_encoding.keys():
                class_path = os.path.join(data_path, "train", label)
                for file_name in os.listdir(os.path.join(class_path, "images")):
                    X.append(os.path.join(class_path, "images", file_name))
                    y.append(label_encoding[label])
        else:
            val_annotations_file = os.path.join(data_path, "val", "val_annotations.txt")
            with open(val_annotations_file, "r") as f:
                for line in f:
                    file_name, label = line.split("\t")[:2]
                    X.append(os.path.join(data_path, "val", "images", file_name))
                    y.append(label_encoding[label])
        return X, y


class TorchVisionDataModule(RenateDataModule):
    """Dataset with data from torchvision.

    Args:
        data_path: the path to the folder containing the dataset files.
        src_bucket: the name of the s3 bucket. If not provided, downloads the data from original source.
        src_object_name: the folder path in the s3 bucket.
        dataset_name: Name of the torchvision dataset.
        download: Set True if data is needed to be downloaded.
        val_size: If `val_size` is provided split the train data into train and validation according to `val_size`.
        seed: Seed used to fix random number generation.
    """

    dataset_dict = {
        "CIFAR10": (torchvision.datasets.CIFAR10, "cifar-10-batches-py"),
        "CIFAR100": (torchvision.datasets.CIFAR100, "cifar-100-python"),
        "FashionMNIST": (torchvision.datasets.FashionMNIST, "FashionMNIST"),
        "MNIST": (torchvision.datasets.MNIST, "MNIST"),
    }

    def __init__(
        self,
        data_path: Union[Path, str],
        src_bucket: Optional[str] = None,
        src_object_name: Optional[str] = None,
        dataset_name: str = "MNIST",
        download: bool = False,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super(TorchVisionDataModule, self).__init__(
            data_path,
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            val_size=val_size,
            seed=seed,
        )
        self._download = download
        self._dataset_name = dataset_name
        assert (
            self._dataset_name in TorchVisionDataModule.dataset_dict
        ), f"Dataset {self._dataset_name} currently not supported."

    def prepare_data(self) -> None:
        """Download torchvision dataset with given dataset_name. If the data is not available in local data_path,
        it is downloaded. If s3 bucket is provided, the data is downloaded from s3, otherwise from the
        original source.
        """
        if not self._download:
            return
        cls, dataset_pathname = TorchVisionDataModule.dataset_dict[self._dataset_name]
        if self._src_bucket is None:
            cls(self._data_path, train=True, download=self._download)
            cls(self._data_path, train=False, download=self._download)
        else:
            download_folder_from_s3(
                src_bucket=self._src_bucket,
                src_object_name=self._src_object_name,
                dst_dir=os.path.join(self._data_path, dataset_pathname),
            )

    def setup(self, stage: Optional[Literal["train", "val", "test"]] = None) -> None:
        """Make assignments: train/valid/test splits (Torchvision datasets only have train and test splits)."""
        if stage in ["train", "val"] or stage is None:
            train_data = TorchVisionDataModule.dataset_dict[self._dataset_name][0](
                self._data_path,
                train=True,
                transform=transforms.ToTensor(),
                target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)),
            )
            self._train_data, self._val_data = self._split_train_val_data(train_data)

        if stage == "test" or stage is None:
            self._test_data = TorchVisionDataModule.dataset_dict[self._dataset_name][0](
                self._data_path,
                train=False,
                transform=transforms.ToTensor(),
                target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)),
            )


class CLEARDataModule(RenateDataModule):
    """Datamodule that process CLEAR datasets: CLEAR10 and CLEAR100.

    Source: https://clear-benchmark.github.io/.

    Args:
        data_path: the path to the folder containing the dataset files.
        src_bucket: the name of the s3 bucket. If not provided, downloads the data from original source.
        src_object_name: the folder path in the s3 bucket.
        transform: Transformation or augmentation to perform on the sample.
        target_transform: Transformation or augmentation to perform on the target.
        dataset_name: CLEAR dataset name, options are clear10 and clear100.
        chunk_id: Used to define the CLEAR dataset splits. There are 10 splits in total with ids from 0 to 9.
        val_size: If `val_size` is provided split the train data into train and validation according to `val_size`.
        seed: Seed used to fix random number generation.
    """

    md5s = {
        "clear10-train-image-only.zip": "5171f720810d60b471c308dee595d430",
        "clear100-train-image-only.zip": "ea85cdba9efcb3abf77eaab5554052c8",
        "clear10-test.zip": "bf9a85bfb78fe742c7ed32648c9a3275",
        "clear100-test.zip": "e160815fb5fd4bc71dacd339ff41e6a9",
    }

    def __init__(
        self,
        data_path: Union[Path, str],
        src_bucket: Optional[str] = None,
        src_object_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dataset_name: str = "CLEAR10",
        chunk_id: int = 0,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super(CLEARDataModule, self).__init__(
            data_path=data_path,
            src_bucket=src_bucket,
            src_object_name=src_object_name,
            val_size=val_size,
            seed=seed,
        )
        self._transform = transform
        self._target_transform = target_transform
        assert dataset_name == "CLEAR10" or dataset_name == "CLEAR100"
        self._dataset_name = dataset_name.lower()
        self._verify_chunk_id(chunk_id)
        self._chunk_id = chunk_id

    def _verify_chunk_id(self, chunk_id: int) -> None:
        """Verify that the chunk_id is valid."""
        assert 0 <= chunk_id <= 9

    def prepare_data(self) -> None:
        """Download CLEAR dataset with given dataset_name (clear10/clear100)."""
        for file_name in [
            f"{self._dataset_name}-train-image-only.zip",
            f"{self._dataset_name}-test.zip",
        ]:
            if not self._verify_file(file_name):
                download_and_unzip_file(
                    self._dataset_name,
                    self._data_path,
                    self._src_bucket,
                    self._src_object_name,
                    "https://clear-challenge.s3.us-east-2.amazonaws.com/",
                    file_name,
                )

    def setup(
        self,
        stage: Optional[Literal["train", "val", "test"]] = None,
        chunk_id: Optional[int] = None,
    ) -> None:
        """Make assignments: train/test splits (CLEAR datasets only have train and test splits)."""
        if chunk_id is None:
            chunk_id = self._chunk_id
        self._verify_chunk_id(chunk_id)
        # Assign train dataset
        if stage in ["train", "val"] or stage is None:
            X, y = self._process_clear_data(train=True, chunk_id=chunk_id)
            train_data = ImageDataset(
                X, y, transform=self._transform, target_transform=self._target_transform
            )
            self._train_data, self._val_data = self._split_train_val_data(train_data)

        # Assign test dataset
        if stage == "test" or stage is None:
            X, y = self._process_clear_data(train=False, chunk_id=chunk_id)
            self._test_data = ImageDataset(
                X, y, transform=self._transform, target_transform=self._target_transform
            )

    def _process_clear_data(self, train: bool, chunk_id: int) -> Tuple[List[str], List[int]]:
        """CLEAR data specific function to read the data from .jpg and return features and labels"""
        data = []
        labels = []
        path = os.path.join(self._data_path, self._dataset_name)
        path = os.path.join(path, "train_image_only" if train else "test")

        # Load the class names and create a class mapping. The class names are in `class_names.txt`
        label_encoding = {}
        with open(os.path.join(path, "class_names.txt"), "r") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip() != "BACKGROUND"]
            label_encoding = {name: cnt for cnt, name in enumerate(class_names)}

        path = os.path.join(path, "labeled_images", str(chunk_id + 1))

        # Go through all the subfolders in the path folder and search for all .jpg images
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".jpg"):
                    folder = root.split("/")[-1]
                    if folder == "BACKGROUND":
                        continue
                    data.append(os.path.join(root, file))
                    labels.append(label_encoding[folder])

        return data, labels
