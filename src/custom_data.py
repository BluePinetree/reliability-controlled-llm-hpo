"""Custom dataset loaders for release examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class LoadedDataset:
    dataset_type: str
    train: Any
    valid: Any
    metadata: dict[str, Any]


def load_custom_dataset(config: dict[str, Any]) -> LoadedDataset:
    dataset = config.get("dataset")
    if dataset == "custom_tabular":
        return load_tabular_csv(config)
    if dataset == "custom_text":
        return load_text_csv(config)
    if dataset == "custom_image_folder":
        return load_image_folder(config)
    raise ValueError(f"Unsupported custom dataset type: {dataset}")


def load_tabular_csv(config: dict[str, Any]) -> LoadedDataset:
    data_cfg = config["data"]
    train_df = pd.read_csv(data_cfg["train_path"])
    valid_path = data_cfg.get("valid_path")
    label_col = data_cfg["label_column"]
    id_cols = set(data_cfg.get("id_columns", []))

    if valid_path:
        valid_df = pd.read_csv(valid_path)
    else:
        train_df, valid_df = train_test_split(
            train_df,
            test_size=float(data_cfg.get("valid_fraction", 0.2)),
            random_state=int(config.get("seed", 42)),
            stratify=train_df[label_col] if data_cfg.get("stratify", True) else None,
        )

    feature_cols = [c for c in train_df.columns if c != label_col and c not in id_cols]
    y_encoder = LabelEncoder()
    y_train = y_encoder.fit_transform(train_df[label_col])
    y_valid = y_encoder.transform(valid_df[label_col])

    X_train = train_df[feature_cols].copy()
    X_valid = valid_df[feature_cols].copy()
    for col in feature_cols:
        if X_train[col].dtype == "object":
            encoder = LabelEncoder()
            combined = pd.concat([X_train[col], X_valid[col]], axis=0).astype(str)
            encoder.fit(combined)
            X_train[col] = encoder.transform(X_train[col].astype(str))
            X_valid[col] = encoder.transform(X_valid[col].astype(str))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.fillna(0))
    X_valid = scaler.transform(X_valid.fillna(0))
    return LoadedDataset(
        "custom_tabular",
        (X_train, y_train),
        (X_valid, y_valid),
        {"n_features": len(feature_cols), "n_classes": len(y_encoder.classes_), "feature_columns": feature_cols},
    )


def load_text_csv(config: dict[str, Any]) -> LoadedDataset:
    data_cfg = config["data"]
    train_df = pd.read_csv(data_cfg["train_path"])
    valid_path = data_cfg.get("valid_path")
    if valid_path:
        valid_df = pd.read_csv(valid_path)
    else:
        valid_df = train_df.sample(frac=float(data_cfg.get("valid_fraction", 0.2)), random_state=int(config.get("seed", 42)))
        train_df = train_df.drop(valid_df.index)
    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]
    labels = sorted(set(train_df[label_col]).union(set(valid_df[label_col])))
    return LoadedDataset(
        "custom_text",
        train_df[[text_col, label_col]].reset_index(drop=True),
        valid_df[[text_col, label_col]].reset_index(drop=True),
        {"text_column": text_col, "label_column": label_col, "n_classes": len(labels)},
    )


def load_image_folder(config: dict[str, Any]) -> LoadedDataset:
    from torchvision import datasets, transforms

    data_cfg = config["data"]
    image_size = int(data_cfg.get("image_size", 224))
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    train = datasets.ImageFolder(root=Path(data_cfg["train_dir"]), transform=transform)
    valid = datasets.ImageFolder(root=Path(data_cfg["valid_dir"]), transform=transform)
    return LoadedDataset(
        "custom_image_folder",
        train,
        valid,
        {"n_classes": len(train.classes), "classes": train.classes, "image_size": image_size},
    )
