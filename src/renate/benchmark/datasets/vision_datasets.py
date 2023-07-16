# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
import torchvision
from avalanche.benchmarks import CLEAR
from torch.utils.data import Dataset
from torchvision import transforms

from renate import defaults
from renate.data import ImageDataset
from renate.data.data_module import RenateDataModule
from renate.utils.file import download_and_unzip_file, download_file, download_folder_from_s3
from renate.utils.pytorch import randomly_split_data


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
            target_transform=transforms.Lambda(to_long),
        )
        self._train_data, self._val_data = self._split_train_val_data(train_data)
        self._test_data = cls(
            self._data_path,
            train=False,
            transform=transforms.ToTensor(),
            target_transform=transforms.Lambda(to_long),
        )


def to_long(x):
    return torch.tensor(x, dtype=torch.long)


class DataWrapper(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int):
        l = self._dataset[idx]
        return l[0], l[1]


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

    md5s = {"clear10-public.zip": "04d3b228599bdd5a874c261907ebe217"}
    dataset_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

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
        CLEAR(
            data_name=self._dataset_name,
            evaluation_protocol="iid",
            feature_type=None,
            seed=self._seed,
            train_transform=None,
            eval_transform=None,
            dataset_root=self._data_path,
        )
        """
        file_name = f"{self._dataset_name}-public.zip"
        if not self._verify_file(file_name):
            download_and_unzip_file(
                self._dataset_name,
                self._data_path,
                self._src_bucket,
                self._src_object_name,
                "https://clear-challenge.s3.us-east-2.amazonaws.com/",
                file_name,
            )
        """

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        benchmark = CLEAR(
            data_name=self._dataset_name,
            evaluation_protocol="iid",
            feature_type=None,
            seed=self._seed,
            train_transform=None,
            eval_transform=None,
            dataset_root=self._data_path,
        )
        self._train_data = DataWrapper(benchmark.train_stream[self._chunk_id].dataset)
        # self._val_data = DataWrapper(benchmark.test_stream[self._chunk_id].dataset)
        self._test_data = DataWrapper(benchmark.test_stream[self._chunk_id].dataset)
        """
        file_paths, labels = self._get_filepaths_and_labels(chunk_id=self._chunk_id)
        dataset = ImageDataset(file_paths, labels, transform=transforms.ToTensor())
        self._train_data, self._test_data = randomly_split_data(dataset, [0.7, 0.3], self._seed)
        self._train_data, self._val_data = self._split_train_val_data(self._train_data)
        """

    def _get_filepaths_and_labels(self, chunk_id: int) -> Tuple[List[str], List[int]]:
        """Extracts all the filepaths and labels for a given chunk id and split."""
        path = os.path.join(self._data_path, self._dataset_name)

        file_paths, labels = [], []
        with open(
            os.path.join(path, "training_folder", "filelists", str(chunk_id + 1), "all.txt"), "r"
        ) as f:
            for line in f.readlines():
                file_path, label = line.split(" ")
                label = int(label)
                file_paths.append(os.path.join(path, file_path))
                labels.append(label)

        return file_paths, labels


class DomainNetDataModule(RenateDataModule):
    """Datamodule that provides access to DomainNet.

    Args:
        data_path: the path to the folder containing the dataset files.
        src_bucket: the name of the s3 bucket. If not provided, downloads the data from original
            source.
        src_object_name: the folder path in the s3 bucket.
        domain: DomainNet domain name, options are clipart, infograph, painting, quickdraw, real, and sketch.
        val_size: Fraction of the training data to be used for validation.
        seed: Seed used to fix random number generation.
    """

    md5s = {
        "clipart.zip": "cd0d8f2d77a4e181449b78ed62bccf1e",
        "clipart_train.txt": "b4349693a7f9c05c53955725c47ed6cb",
        "clipart_test.txt": "f5ddbcfd657a3acf9d0f7da10db22565",
        "infograph.zip": "720380b86f9e6ab4805bb38b6bd135f8",
        "infograph_train.txt": "379b50054f4ac2018dca4f89421b92d9",
        "infograph_test.txt": "779626b50869edffe8ea6941c3755c71",
        "painting.zip": "1ae32cdb4f98fe7ab5eb0a351768abfd",
        "painting_train.txt": "b732ced3939ac8efdd8c0a889dca56cc",
        "painting_test.txt": "c1a828fdfe216fb109f1c0083a252c6f",
        "quickdraw.zip": "bdc1b6f09f277da1a263389efe0c7a66",
        "quickdraw_train.txt": "b4349693a7f9c05c53955725c47ed6cb",
        "quickdraw_test.txt": "f5ddbcfd657a3acf9d0f7da10db22565",
        "real.zip": "dcc47055e8935767784b7162e7c7cca6",
        "real_train.txt": "8ebf02c2075fadd564705f0dc7cd6291",
        "real_test.txt": "6098816791c3ebed543c71ffa11b9054",
        "sketch.zip": "658d8009644040ff7ce30bb2e820850f",
        "sketch_train.txt": "1233bd18aa9a8a200bf4cecf1c34ef3e",
        "sketch_test.txt": "d8a222e4672cfd585298aa14d02ea441",
    }

    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    dataset_stats = {
        "clipart": {"mean": [0.7395, 0.7195, 0.6865], "std": [0.3621, 0.3640, 0.3873]},
        "infograph": {"mean": [0.6882, 0.6962, 0.6644], "std": [0.3328, 0.3095, 0.3277]},
        "painting": {"mean": [0.5737, 0.5456, 0.5067], "std": [0.3079, 0.3003, 0.3161]},
        "quickdraw": {"mean": [0.9525, 0.9525, 0.9525], "std": [0.2127, 0.2127, 0.2127]},
        "real": {"mean": [0.6066, 0.5897, 0.5564], "std": [0.3335, 0.3270, 0.3485]},
        "sketch": {"mean": [0.8325, 0.8269, 0.8180], "std": [0.2723, 0.2747, 0.2801]},
        "all": {"mean": [0.7491, 0.7391, 0.7179], "std": [0.3318, 0.3314, 0.3512]},
    }

    def __init__(
        self,
        data_path: Union[Path, str],
        src_bucket: Optional[str] = None,
        src_object_name: Optional[str] = None,
        domain: str = "clipart",
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
        self._dataset_name = domain.lower()
        assert self._dataset_name in self.domains, f"Unknown domain {self._dataset_name}."

    def prepare_data(self) -> None:
        """Download DomainNet dataset for given domain."""
        file_name = f"{self._dataset_name}.zip"
        url = "http://csr.bu.edu/ftp/visda/2019/multi-source/"
        if self._dataset_name in ["clipart", "painting"]:
            url = os.path.join(url, "groundtruth")
        if not self._verify_file(file_name):
            download_and_unzip_file(
                self._dataset_name,
                self._data_path,
                self._src_bucket,
                self._src_object_name,
                url,
                file_name,
            )
        for file_name in [f"{self._dataset_name}_train.txt", f"{self._dataset_name}_test.txt"]:
            if not self._verify_file(file_name):
                download_file(
                    self._dataset_name,
                    self._data_path,
                    self._src_bucket,
                    self._src_object_name,
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/",
                    file_name,
                )

    def setup(self) -> None:
        """Set up train, test and val datasets."""
        X, y = self._get_filepaths_and_labels("train")
        train_data = ImageDataset(X, y, transform=transforms.ToTensor())
        self._train_data, self._val_data = self._split_train_val_data(train_data)
        X, y = self._get_filepaths_and_labels("test")
        self._test_data = ImageDataset(X, y, transform=transforms.ToTensor())

    def _get_filepaths_and_labels(self, split: str) -> Tuple[List[str], List[int]]:
        """Extracts all the filepaths and labels for a given split."""
        path = os.path.join(self._data_path, self._dataset_name)
        df = pd.read_csv(
            os.path.join(path, f"{self._dataset_name}_{split}.txt"),
            sep=" ",
            header=None,
            names=["path", "label"],
        )
        data = list(df.path.apply(lambda x: os.path.join(path, x)))
        labels = list(df.label)
        return data, labels
