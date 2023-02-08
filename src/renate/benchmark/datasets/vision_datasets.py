# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from renate import defaults
from renate.data import ImageDataset
from renate.data.data_module import RenateDataModule
from renate.utils.file import download_and_unzip_file, download_folder_from_s3


class TinyImageNetDataModule(RenateDataModule):
    """Datamodule that process TinyImageNet dataset.

    Source: http://cs231n.stanford.edu/

    The TinyImageNet dataset is a subset of the ImageNet dataset. It contains 200 classes, each with
    500 training images, 50 validation images with labels. There are also 50 unlabeled test images
    per class, which we are not using here. We use the validation split as the test set.

    Args:
        data_path: Path to the directory where the dataset should be stored.
        src_bucket: Name of the bucket where the dataset is stored.
        src_object_name: Name of the object in the bucket where the dataset is stored.
        val_size: Fraction of the training data to be used for validation.
        seed: Seed to be used for splitting the dataset.
    """

    md5s = {"tiny-imagenet-200.zip": "90528d7ca1a48142e341f4ef8d21d0de"}

    def __init__(
        self,
        data_path: Union[Path, str],
        src_bucket: Optional[str] = None,
        src_object_name: Optional[str] = None,
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

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        X, y = self._preprocess_tiny_imagenet(train=True)
        train_data = ImageDataset(X, y, transform=transforms.ToTensor())
        self._train_data, self._val_data = self._split_train_val_data(train_data)
        X, y = self._preprocess_tiny_imagenet(train=False)
        self._test_data = ImageDataset(X, y, transform=transforms.ToTensor())

    def _preprocess_tiny_imagenet(self, train: bool) -> Tuple[List[str], List[int]]:
        """A helper function to preprocess the TinyImageNet dataset."""
        if train:
            data_path = os.path.join(self._data_path, self._dataset_name, "tiny-imagenet-200")
        else:
            data_path = os.path.join(self._data_path, self._dataset_name, "tiny-imagenet-200")
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
    """Data module wrapping torchvision datasets.

    Args:
        data_path: the path to the folder containing the dataset files.
        src_bucket: the name of the s3 bucket. If not provided, downloads the data from original
            source.
        src_object_name: the folder path in the s3 bucket.
        dataset_name: Name of the torchvision dataset.
        val_size: Fraction of the training data to be used for validation.
        seed: Seed used to fix random number generation.
    """

    dataset_dict = {
        "CIFAR10": (torchvision.datasets.CIFAR10, "cifar-10-batches-py"),
        "CIFAR100": (torchvision.datasets.CIFAR100, "cifar-100-python"),
        "FashionMNIST": (torchvision.datasets.FashionMNIST, "FashionMNIST"),
        "MNIST": (torchvision.datasets.MNIST, "MNIST"),
    }
    dataset_stats = {
        "CIFAR10": {
            "mean": (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
            "std": (0.24703223246174102, 0.24348512800151828, 0.26158784172803257),
        },
        "CIFAR100": {
            "mean": (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            "std": (0.26733428587941854, 0.25643846292120615, 0.2761504713263903),
        },
        "FashionMNIST": {"mean": 0.2860405969887955, "std": 0.3530242445149223},
        "MNIST": {"mean": 0.1306604762738429, "std": 0.30810780385646264},
    }

    def __init__(
        self,
        data_path: Union[Path, str],
        src_bucket: Optional[str] = None,
        src_object_name: Optional[str] = None,
        dataset_name: str = "MNIST",
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
        self._dataset_name = dataset_name
        if self._dataset_name not in TorchVisionDataModule.dataset_dict:
            raise ValueError(f"Dataset {self._dataset_name} currently not supported.")

    def prepare_data(self) -> None:
        """Download data.

        If s3 bucket is given, the data is downloaded from s3, otherwise from the original source.
        """
        cls, dataset_pathname = TorchVisionDataModule.dataset_dict[self._dataset_name]
        if self._src_bucket is None:
            cls(self._data_path, train=True, download=True)
            cls(self._data_path, train=False, download=True)
        else:
            download_folder_from_s3(
                src_bucket=self._src_bucket,
                src_object_name=self._src_object_name,
                dst_dir=os.path.join(self._data_path, dataset_pathname),
            )

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        cls, _ = TorchVisionDataModule.dataset_dict[self._dataset_name]
        train_data = cls(
            self._data_path,
            train=True,
            transform=transforms.ToTensor(),
            target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)),
        )
        self._train_data, self._val_data = self._split_train_val_data(train_data)
        self._test_data = cls(
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
        src_bucket: the name of the s3 bucket. If not provided, downloads the data from original
            source.
        src_object_name: the folder path in the s3 bucket.
        dataset_name: CLEAR dataset name, options are clear10 and clear100.
        chunk_id: Used to define the CLEAR dataset splits. There are 10 splits in total with ids
            from 0 to 9.
        val_size: Fraction of the training data to be used for validation.
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
        self._dataset_name = dataset_name.lower()
        assert self._dataset_name in ["clear10", "clear100"]
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

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        X, y = self._get_filepaths_and_labels(train=True, chunk_id=self._chunk_id)
        train_data = ImageDataset(X, y, transform=transforms.ToTensor())
        self._train_data, self._val_data = self._split_train_val_data(train_data)
        self._test_data = []
        for i in range(10):
            X, y = self._get_filepaths_and_labels(train=False, chunk_id=i)
            self._test_data.append(ImageDataset(X, y, transform=transforms.ToTensor()))

    # TODO: This is a work-around to make CLEAR available as a data module (single test dataset)
    # as well as a scenario (all test datasets) via BenchmarkScenario. Clean up!
    def test_data(self) -> Dataset:
        return self._test_data[self._chunk_id]

    def _get_filepaths_and_labels(self, train: bool, chunk_id: int) -> Tuple[List[str], List[int]]:
        """Extracts all the filepaths and labels for a given chunk id and split."""
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
