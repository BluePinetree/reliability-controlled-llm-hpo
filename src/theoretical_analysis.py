"""Convergence, regret, and sample-complexity utilities for HPO results."""

from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TheoreticalBound:
    """Summary of bound-related quantities for one method."""

    method_name: str
    convergence_rate: float
    regret_bound: float
    sample_complexity: int
    confidence: float = 0.95


class ConvergenceAnalyzer:
    """Stores per-trial accuracy sequences and computes convergence summaries."""

    def __init__(self) -> None:
        self.convergence_data = {}

    def add_experiment_results(self, method_name: str, accuracies: List[float]) -> None:
        """Register the accuracy trajectory for one method."""

        self.convergence_data[method_name] = list(accuracies)

    def compute_convergence_rate(self, method_name: str) -> int:
        """Return the first trial reaching 95% of the method's final best value."""

        accuracies = self._get_accuracies(method_name)
        best_acc = np.max(accuracies)
        target_acc = 0.95 * best_acc
        converged_indices = np.where(accuracies >= target_acc)[0]
        if len(converged_indices) == 0:
            return len(accuracies)
        return int(converged_indices[0] + 1)

    def compute_cumulative_regret(self, method_name: str, optimal_value: float) -> List[float]:
        """Return cumulative regret sum_{i=1}^{t} (f_star - f(x_i))."""

        accuracies = self._get_accuracies(method_name)
        regrets = float(optimal_value) - accuracies
        return np.cumsum(regrets).tolist()

    def compute_simple_regret(self, method_name: str, optimal_value: float) -> List[float]:
        """Return simple regret f_star - max_{i<=t} f(x_i)."""

        accuracies = self._get_accuracies(method_name)
        best_so_far = np.maximum.accumulate(accuracies)
        return (float(optimal_value) - best_so_far).tolist()

    def estimate_sample_complexity(
        self,
        method_name: str,
        target_accuracy: float,
        confidence: float = 0.95,
    ) -> int:
        """Return the first trial index that reaches the target accuracy."""

        del confidence
        accuracies = self._get_accuracies(method_name)
        reached_indices = np.where(accuracies >= target_accuracy)[0]
        if len(reached_indices) == 0:
            return len(accuracies)
        return int(reached_indices[0] + 1)

    def plot_convergence_curves(self, save_path: Optional[str] = None) -> None:
        """Plot best-so-far convergence curves for registered methods."""

        plt.figure(figsize=(10, 6))
        for method_name, accuracies in self.convergence_data.items():
            best_so_far = np.maximum.accumulate(accuracies)
            plt.plot(range(1, len(best_so_far) + 1), best_so_far,
                     marker="o", label=method_name, linewidth=2)

        plt.xlabel("Number of Trials", fontsize=12)
        plt.ylabel("Best Accuracy So Far", fontsize=12)
        plt.title("Convergence Curves", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._finish_plot(save_path)

    def plot_regret_curves(self, optimal_value: float, save_path: Optional[str] = None) -> None:
        """Plot simple and cumulative regret curves for registered methods."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for method_name in self.convergence_data.keys():
            simple_regret = self.compute_simple_regret(method_name, optimal_value)
            ax1.plot(range(1, len(simple_regret) + 1), simple_regret,
                     marker="o", label=method_name, linewidth=2)

        ax1.set_xlabel("Number of Trials", fontsize=12)
        ax1.set_ylabel("Simple Regret", fontsize=12)
        ax1.set_title("Simple Regret", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        for method_name in self.convergence_data.keys():
            cumulative_regret = self.compute_cumulative_regret(method_name, optimal_value)
            ax2.plot(range(1, len(cumulative_regret) + 1), cumulative_regret,
                     marker="o", label=method_name, linewidth=2)

        ax2.set_xlabel("Number of Trials", fontsize=12)
        ax2.set_ylabel("Cumulative Regret", fontsize=12)
        ax2.set_title("Cumulative Regret", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        self._finish_plot(save_path)

    def _get_accuracies(self, method_name: str) -> np.ndarray:
        if method_name not in self.convergence_data:
            raise ValueError(f"Method {method_name} not found")
        return np.array(self.convergence_data[method_name])

    @staticmethod
    def _finish_plot(save_path: Optional[str]) -> None:
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()


class TheoreticalBoundEstimator:
    """Closed-form reference bounds used for analysis and plotting."""

    @staticmethod
    def gp_ucb_regret_bound(T: int, d: int, delta: float = 0.1) -> float:
        """Return a GP-UCB-style O(sqrt(T d log(T) log(1/delta))) bound."""

        return float(np.sqrt(T * d * np.log(max(T, 2)) * np.log(1 / delta)))

    @staticmethod
    def random_search_regret_bound(T: int, d: int) -> float:
        """Return a simple dimension-dependent random-search regret proxy."""

        return float(T ** (1 - 1 / (2 * d)))

    @staticmethod
    def llm_episodic_regret_bound(T: int, d: int, k: int, delta: float = 0.1) -> float:
        """Return the manuscript's effective-dimension episodic-memory proxy bound."""

        d_eff = d / (1 + np.log(max(k, 1)))
        return float(np.sqrt(T * d_eff * np.log(max(T, 2)) * np.log(1 / delta)))

    @staticmethod
    def compute_sample_complexity_bound(epsilon: float, delta: float, d: int) -> int:
        """Return ceil((d / epsilon^2) log(1 / delta))."""

        if epsilon <= 0 or delta <= 0:
            raise ValueError("epsilon and delta must be positive")
        return int(np.ceil((d / (epsilon ** 2)) * np.log(1 / delta)))

    @staticmethod
    def plot_theoretical_bounds(
        T_max: int = 100,
        d: int = 5,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot reference regret-bound curves."""

        T_values = np.arange(1, T_max + 1)
        gp_ucb_bounds = [
            TheoreticalBoundEstimator.gp_ucb_regret_bound(T, d) for T in T_values
        ]
        random_bounds = [
            TheoreticalBoundEstimator.random_search_regret_bound(T, d) for T in T_values
        ]
        llm_episodic_bounds = [
            TheoreticalBoundEstimator.llm_episodic_regret_bound(T, d, k=5)
            for T in T_values
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(T_values, gp_ucb_bounds, label="GP-UCB", linewidth=2)
        plt.plot(T_values, random_bounds, label="Random Search", linewidth=2)
        plt.plot(T_values, llm_episodic_bounds, label="LLM+Episodic", linewidth=2, linestyle="--")
        plt.xlabel("Number of Trials (T)", fontsize=12)
        plt.ylabel("Regret Bound", fontsize=12)
        plt.title(f"Reference Regret Bounds (d={d})", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
