"""Statistical tests and summary tables for HPO experiment results."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ExperimentResult:
    """Single seed-level result for one optimization method."""

    method_name: str
    accuracy: float
    seed: int
    config_id: Optional[int] = None


class StatisticalSignificance:
    """Container for method-wise accuracy samples and common significance tests."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.results: Dict[str, List[float]] = {}

    def add_results(self, method_name: str, accuracies: List[float]) -> None:
        """Append accuracy samples for a method."""

        if method_name not in self.results:
            self.results[method_name] = []
        self.results[method_name].extend(float(v) for v in accuracies)

    def t_test(
        self,
        method1: str,
        method2: str,
        alternative: str = "two-sided",
    ) -> Dict:
        """Run an independent two-sample t-test between two methods."""

        data1, data2 = self._method_arrays(method1, method2)
        t_statistic, p_value = stats.ttest_ind(data1, data2, alternative=alternative)
        cohens_d = self._cohens_d(data1, data2)

        return {
            "method1": method1,
            "method2": method2,
            "mean1": np.mean(data1),
            "mean2": np.mean(data2),
            "std1": np.std(data1, ddof=1),
            "std2": np.std(data2, ddof=1),
            "n1": len(data1),
            "n2": len(data2),
            "t_statistic": t_statistic,
            "p_value": p_value,
            "is_significant": p_value < self.alpha,
            "cohens_d": cohens_d,
            "effect_size_interpretation": self._interpret_cohens_d(cohens_d),
        }

    def paired_t_test(
        self,
        method1: str,
        method2: str,
        alternative: str = "two-sided",
    ) -> Dict:
        """Run a paired t-test between two methods with matched samples."""

        data1, data2 = self._method_arrays(method1, method2)
        if len(data1) != len(data2):
            raise ValueError("Paired t-test requires equal number of samples")

        t_statistic, p_value = stats.ttest_rel(data1, data2, alternative=alternative)
        diff = data1 - data2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        cohens_d = 0.0 if std_diff == 0 else mean_diff / std_diff

        return {
            "method1": method1,
            "method2": method2,
            "mean1": np.mean(data1),
            "mean2": np.mean(data2),
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "n": len(data1),
            "t_statistic": t_statistic,
            "p_value": p_value,
            "is_significant": p_value < self.alpha,
            "cohens_d": cohens_d,
            "effect_size_interpretation": self._interpret_cohens_d(cohens_d),
        }

    def anova(self) -> Dict:
        """Run one-way ANOVA across all registered methods."""

        if len(self.results) < 2:
            raise ValueError("ANOVA requires at least 2 methods")

        data_groups = [np.array(accuracies) for accuracies in self.results.values()]
        f_statistic, p_value = stats.f_oneway(*data_groups)

        method_stats = {}
        for method_name, accuracies in self.results.items():
            method_stats[method_name] = {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies, ddof=1),
                "n": len(accuracies),
            }

        return {
            "f_statistic": f_statistic,
            "p_value": p_value,
            "is_significant": p_value < self.alpha,
            "method_stats": method_stats,
            "interpretation": (
                "At least one method is significantly different"
                if p_value < self.alpha
                else "No significant difference found"
            ),
        }

    def confidence_interval(
        self,
        method_name: str,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Return mean and t-based confidence interval for one method."""

        if method_name not in self.results:
            raise ValueError(f"Method {method_name} not found in results")

        data = np.array(self.results[method_name])
        n = len(data)
        mean = np.mean(data)
        if n < 2:
            return mean, mean, mean

        std_err = stats.sem(data)
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin_of_error = t_critical * std_err
        return mean, mean - margin_of_error, mean + margin_of_error

    def all_pairwise_comparisons(self, correction: str = "bonferroni") -> pd.DataFrame:
        """Return pairwise t-tests with optional Bonferroni or Holm correction."""

        methods = list(self.results.keys())
        n_comparisons = len(methods) * (len(methods) - 1) // 2
        comparisons = []
        p_values = []

        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1:]:
                result = self.t_test(method1, method2)
                comparisons.append(result)
                p_values.append(result["p_value"])

        if n_comparisons == 0:
            return pd.DataFrame(comparisons)

        if correction == "bonferroni":
            adjusted_alpha = self.alpha / n_comparisons
            for comp, p_val in zip(comparisons, p_values):
                comp["adjusted_p_value"] = p_val
                comp["is_significant_corrected"] = p_val < adjusted_alpha
        elif correction == "holm":
            sorted_indices = np.argsort(p_values)
            for rank, idx in enumerate(sorted_indices, 1):
                adjusted_alpha = self.alpha / (n_comparisons - rank + 1)
                comparisons[idx]["adjusted_p_value"] = p_values[idx]
                comparisons[idx]["is_significant_corrected"] = p_values[idx] < adjusted_alpha
        else:
            raise ValueError("correction must be one of: bonferroni, holm")

        return pd.DataFrame(comparisons)

    def summary_table(self, confidence: float = 0.95) -> pd.DataFrame:
        """Return method-level best, mean, standard deviation, and confidence interval."""

        summary = []
        for method_name in self.results.keys():
            data = np.array(self.results[method_name])
            mean, lower, upper = self.confidence_interval(method_name, confidence)
            summary.append({
                "Method": method_name,
                "Best Acc (%)": np.max(data) * 100,
                "Mean Acc (%)": mean * 100,
                "Std Dev (%)": np.std(data, ddof=1) * 100 if len(data) > 1 else 0.0,
                f"{int(confidence * 100)}% CI Lower": lower * 100,
                f"{int(confidence * 100)}% CI Upper": upper * 100,
                "N": len(data),
            })

        return pd.DataFrame(summary).sort_values("Mean Acc (%)", ascending=False)

    def _method_arrays(self, method1: str, method2: str) -> Tuple[np.ndarray, np.ndarray]:
        if method1 not in self.results or method2 not in self.results:
            raise ValueError(f"Method {method1} or {method2} not found in results")
        return np.array(self.results[method1]), np.array(self.results[method2])

    @staticmethod
    def _cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
        n1, n2 = len(data1), len(data2)
        if n1 < 2 or n2 < 2:
            return 0.0
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return 0.0 if pooled_std == 0 else (np.mean(data1) - np.mean(data2)) / pooled_std

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        if abs_d < 0.5:
            return "small"
        if abs_d < 0.8:
            return "medium"
        return "large"
