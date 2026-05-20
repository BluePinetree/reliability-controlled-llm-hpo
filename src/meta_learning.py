"""Lightweight meta-learning context for paper prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from paper_datasets import DatasetLoader


@dataclass
class MetaTask:
    dataset_name: str
    domain: str
    task_type: str
    description: str
    best_hyperparameters: Dict[str, float]
    best_accuracy: float

    @property
    def embedding_text(self) -> str:
        return f"{self.domain} {self.task_type} {self.dataset_name} {self.description}"


class MetaLearner:
    """Small static knowledge base used for prompt context."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.tasks: List[MetaTask] = []
        self.embedding_model = SentenceTransformer(embedding_model)
        self.task_embeddings: Optional[np.ndarray] = None
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> None:
        self.add_task(
            MetaTask(
                dataset_name="ImageNet",
                domain="Computer Vision",
                task_type="Image Classification",
                description="Large scale image classification with 1000 classes.",
                best_hyperparameters={"learning_rate": 0.1, "batch_size": 256, "weight_decay": 1e-4, "momentum": 0.9},
                best_accuracy=0.76,
            )
        )
        self.add_task(
            MetaTask(
                dataset_name="SST-2",
                domain="Natural Language Processing",
                task_type="Sentiment Analysis",
                description="Binary sentiment classification of movie reviews.",
                best_hyperparameters={"learning_rate": 2e-5, "batch_size": 32, "weight_decay": 0.01},
                best_accuracy=0.92,
            )
        )
        self.add_task(
            MetaTask(
                dataset_name="CIFAR-10",
                domain="Computer Vision",
                task_type="Image Classification",
                description="10 classes of natural images.",
                best_hyperparameters={"learning_rate": 0.05, "batch_size": 128, "weight_decay": 5e-4},
                best_accuracy=0.95,
            )
        )

    def add_task(self, task: MetaTask) -> None:
        self.tasks.append(task)
        self._update_embeddings()

    def _update_embeddings(self) -> None:
        texts = [task.embedding_text for task in self.tasks]
        if texts:
            self.task_embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

    def retrieve_similar_tasks(self, current_dataset_name: str, k: int = 3) -> List[Tuple[MetaTask, float]]:
        info = DatasetLoader.get_dataset_info(current_dataset_name)
        query_text = f"{info.get('domain', '')} {info.get('task', '')} {current_dataset_name}"
        query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True)
        if self.task_embeddings is None or not self.tasks:
            return []
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.task_embeddings)[0]
        scored = []
        for idx, score in enumerate(similarities):
            if self.tasks[idx].dataset_name.lower() == current_dataset_name.lower():
                continue
            scored.append((self.tasks[idx], float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def format_for_prompt(self, similar_tasks: List[Tuple[MetaTask, float]]) -> str:
        if not similar_tasks:
            return "No relevant meta-learning tasks found."
        lines = [
            "## Meta-Learning Summary",
            "| dataset | domain | task_type | similarity | best_hyperparameters | best_accuracy |",
            "|---|---|---|---|---|---|",
        ]
        for task, score in similar_tasks:
            hp = json.dumps(task.best_hyperparameters, separators=(",", ":"))
            lines.append(f"| {task.dataset_name} | {task.domain} | {task.task_type} | {score:.2f} | {hp} | {task.best_accuracy:.4f} |")
        return "\n".join(lines)
