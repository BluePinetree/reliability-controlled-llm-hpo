"""Episodic memory for retrieving prior HPO trials as prompt context."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


@dataclass
class Episode:
    """Stored HPO episode with optional text embedding."""
    episode_id: int
    hyperparameters: Dict[str, float]
    accuracy: float
    hypothesis: str
    timestamp: str
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Serialize the episode for JSON storage."""
        return {
            'episode_id': self.episode_id,
            'hyperparameters': self.hyperparameters,
            'accuracy': self.accuracy,
            'hypothesis': self.hypothesis,
            'timestamp': self.timestamp,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Episode':
        """Deserialize an episode from JSON-compatible data."""
        embedding = np.array(data['embedding']) if data.get('embedding') is not None else None
        return cls(
            episode_id=data['episode_id'],
            hyperparameters=data['hyperparameters'],
            accuracy=data['accuracy'],
            hypothesis=data['hypothesis'],
            timestamp=data['timestamp'],
            embedding=embedding
        )


class EpisodicMemory:
    """Store and retrieve HPO episodes using sentence-embedding similarity."""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.episodes: List[Episode] = []
        self.embedding_model = SentenceTransformer(embedding_model)
        
    def add_episode(self, 
                   hyperparameters: Dict[str, float],
                   accuracy: float,
                   hypothesis: str,
                   timestamp: str) -> Episode:
        """Add one completed trial to memory."""
        episode_id = len(self.episodes)
        
        embedding = self._encode_hypothesis(hypothesis)
        
        episode = Episode(
            episode_id=episode_id,
            hyperparameters=hyperparameters,
            accuracy=accuracy,
            hypothesis=hypothesis,
            timestamp=timestamp,
            embedding=embedding
        )
        
        self.episodes.append(episode)
        return episode
    
    def retrieve_similar_episodes(self, 
                                  current_hypothesis: str,
                                  k: int = 5,
                                  min_similarity: float = 0.3) -> List[Tuple[Episode, float]]:
        """Retrieve episodes with embedding similarity above the given threshold."""
        if not self.episodes:
            return []
        
        current_embedding = self._encode_hypothesis(current_hypothesis)
        
        similarities = []
        for episode in self.episodes:
            if episode.embedding is None:
                continue
            
            sim = cosine_similarity(
                current_embedding.reshape(1, -1),
                episode.embedding.reshape(1, -1)
            )[0, 0]
            
            if sim >= min_similarity:
                similarities.append((episode, float(sim)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def retrieve_top_performing_episodes(self, k: int = 5) -> List[Episode]:
        """Return the top-k episodes by accuracy."""
        sorted_episodes = sorted(self.episodes, key=lambda ep: ep.accuracy, reverse=True)
        return sorted_episodes[:k]

    def retrieve_worst_episodes(self, k: int = 5) -> List[Episode]:
        """Return the bottom-k episodes by accuracy."""
        sorted_episodes = sorted(self.episodes, key=lambda ep: ep.accuracy)
        return sorted_episodes[:k]
    
    def retrieve_diverse_episodes(self, k: int = 5) -> List[Episode]:
        """Select a diverse subset by greedy embedding distance."""
        if len(self.episodes) <= k:
            return self.episodes.copy()
        
        selected = [self.episodes[0]]
        remaining = self.episodes[1:]
        
        for _ in range(k - 1):
            max_min_dist = -1
            best_episode = None
            
            for episode in remaining:
                if episode.embedding is None:
                    continue
                
                min_dist = min([
                    1 - cosine_similarity(
                        episode.embedding.reshape(1, -1),
                        selected_ep.embedding.reshape(1, -1)
                    )[0, 0]
                    for selected_ep in selected
                    if selected_ep.embedding is not None
                ])
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_episode = episode
            
            if best_episode:
                selected.append(best_episode)
                remaining.remove(best_episode)
        
        return selected
    
    def get_statistics(self) -> Dict:
        """Return count and accuracy statistics for stored episodes."""
        if not self.episodes:
            return {
                'total_episodes': 0,
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'best_accuracy': 0.0,
                'worst_accuracy': 0.0
            }
        
        accuracies = [ep.accuracy for ep in self.episodes]
        
        return {
            'total_episodes': len(self.episodes),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'best_accuracy': np.max(accuracies),
            'worst_accuracy': np.min(accuracies)
        }
    
    def save(self, filepath: str):
        """Save stored episodes to JSON."""
        data = {
            'episodes': [ep.to_dict() for ep in self.episodes],
            'embedding_model': self.embedding_model.get_sentence_embedding_dimension()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load stored episodes from JSON."""
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        self.episodes = [Episode.from_dict(ep_data) for ep_data in data['episodes']]
    
    def _encode_hypothesis(self, hypothesis: str) -> np.ndarray:
        """Encode an episode rationale or current hypothesis."""
        return self.embedding_model.encode(hypothesis, convert_to_numpy=True)
    
    def format_episodes_for_prompt(self,
                                   top_episodes: List,
                                   worst_episodes: List,
                                   diverse_episodes: List) -> str:
        """
        Format episodic memory as structured tables for prompt context.
        Each list can contain Episode or (Episode, similarity).
        """
        def normalize(items: List):
            normalized = []
            for item in items:
                if isinstance(item, tuple):
                    episode, similarity = item
                else:
                    episode, similarity = item, None
                normalized.append((episode, similarity))
            return normalized

        def table(title: str, items: List[Tuple[Episode, Optional[float]]]) -> str:
            if not items:
                return f"### {title}\nNo episodes.\n"
            lines = [
                f"### {title}",
                "| rank | episode_id | accuracy | similarity | hyperparameters |",
                "|---|---|---|---|---|"
            ]
            for idx, (episode, similarity) in enumerate(items, 1):
                sim_str = "" if similarity is None else f"{similarity:.3f}"
                hp_str = json.dumps(episode.hyperparameters, separators=(",", ":"))
                lines.append(
                    f"| {idx} | {episode.episode_id} | {episode.accuracy:.4f} | {sim_str} | {hp_str} |"
                )
            return "\n".join(lines) + "\n"

        top_norm = normalize(top_episodes)
        worst_norm = normalize(worst_episodes)
        diverse_norm = normalize(diverse_episodes)

        sections = [
            "## Episodic Memory Summary",
            table("Top-K Episodes", top_norm),
            table("Worst-K Episodes", worst_norm),
            table("Diverse-K Episodes", diverse_norm)
        ]
        return "\n".join(sections)
