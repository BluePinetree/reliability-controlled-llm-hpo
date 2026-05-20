"""Dataset loaders used by the paper experiments.

This module intentionally contains only the datasets reported in the
manuscript artifact: CIFAR-100, AG News, and IMDb. Custom tabular/image/text
recipes are handled separately in ``custom_data.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import Subset


class DatasetLoader:
    """Load datasets used in the paper experiments."""

    @staticmethod
    def load_cifar100(
        data_dir: str = "./data",
        train_subset: Optional[int] = None,
        val_subset: Optional[int] = 5000,
    ):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        if train_subset:
            train_dataset = Subset(train_dataset, range(min(train_subset, len(train_dataset))))
        if val_subset:
            val_dataset = Subset(val_dataset, range(min(val_subset, len(val_dataset))))
        return train_dataset, val_dataset

    @staticmethod
    def load_ag_news(data_dir: str = "./data", train_subset: Optional[int] = 5000, val_subset: Optional[int] = 1000):
        dataset = load_dataset("ag_news", cache_dir=data_dir)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        if train_subset:
            train_dataset = train_dataset.select(range(min(train_subset, len(train_dataset))))
        if val_subset:
            val_dataset = val_dataset.select(range(min(val_subset, len(val_dataset))))
        return train_dataset, val_dataset, 4

    @staticmethod
    def load_imdb(data_dir: str = "./data", train_subset: Optional[int] = 5000, val_subset: Optional[int] = 1000):
        dataset = load_dataset("imdb", cache_dir=data_dir)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        if train_subset:
            train_dataset = train_dataset.select(range(min(train_subset, len(train_dataset))))
        if val_subset:
            val_dataset = val_dataset.select(range(min(val_subset, len(val_dataset))))
        return train_dataset, val_dataset, 2

    @staticmethod
    def get_dataset_info(dataset_name: str) -> dict:
        info = {
            "cifar100": {"domain": "CV", "n_classes": 100, "task": "image_classification"},
            "ag_news": {"domain": "NLP", "n_classes": 4, "task": "text_classification"},
            "imdb": {"domain": "NLP", "n_classes": 2, "task": "sentiment_classification"},
        }
        key = dataset_name.lower()
        if key not in info:
            raise ValueError(f"Dataset is not part of the paper artifact: {dataset_name}")
        return info[key]
