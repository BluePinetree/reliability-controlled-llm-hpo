import json
import math
from typing import Any, Dict, List, Sequence


class PromptBuilder:
    def __init__(self, top_k: int = 3) -> None:
        self.top_k = int(top_k)

    def build(
        self,
        trials: Sequence[Dict[str, Any]],
        search_space: Dict[str, Any],
        objective: Dict[str, Any],
        maximize: bool,
    ) -> str:
        completed = [t for t in trials if t.get("status") == "completed"]
        best = self._top_k(completed, maximize, self.top_k)
        worst = self._top_k(completed, not maximize, self.top_k)
        diverse = self._diverse_k(completed, search_space, self.top_k)

        sections = []
        sections.append("Objective")
        sections.append(self._format_objective(objective, maximize))
        sections.append("Search Space")
        sections.append(json.dumps(search_space, indent=2, sort_keys=True))
        sections.append("Top-K Best")
        sections.append(self._format_trials(best))
        sections.append("Top-K Worst")
        sections.append(self._format_trials(worst))
        sections.append("Top-K Diverse")
        sections.append(self._format_trials(diverse))
        sections.append("Return ONLY JSON with fields: candidates[], global_notes.")

        return "\n\n".join(sections)

    @staticmethod
    def _format_objective(objective: Dict[str, Any], maximize: bool) -> str:
        metric = objective.get("metric", "metric")
        direction = "maximize" if maximize else "minimize"
        return f"Metric: {metric} | Direction: {direction}"

    @staticmethod
    def _format_trials(trials: Sequence[Dict[str, Any]]) -> str:
        lines = []
        for t in trials:
            metric = t.get("metric")
            params = t.get("params")
            trial_id = t.get("trial_id")
            lines.append(f"- id={trial_id} metric={metric} params={params}")
        return "\n".join(lines) if lines else "- (none)"

    @staticmethod
    def _top_k(trials: Sequence[Dict[str, Any]], maximize: bool, k: int) -> List[Dict[str, Any]]:
        if not trials:
            return []
        sorted_trials = sorted(
            trials,
            key=lambda t: t.get("metric", float("-inf" if maximize else "inf")),
            reverse=maximize,
        )
        return sorted_trials[:k]

    def _diverse_k(
        self,
        trials: Sequence[Dict[str, Any]],
        search_space: Dict[str, Any],
        k: int,
    ) -> List[Dict[str, Any]]:
        if not trials or k <= 0:
            return []
        vectors = []
        for t in trials:
            params = t.get("params", {})
            vectors.append(self._vectorize_params(params, search_space))

        selected = [0]
        while len(selected) < min(k, len(trials)):
            best_idx = None
            best_dist = -1.0
            for i in range(len(trials)):
                if i in selected:
                    continue
                min_dist = min(self._l2(vectors[i], vectors[j]) for j in selected)
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = i
            if best_idx is None:
                break
            selected.append(best_idx)
        return [trials[i] for i in selected]

    @staticmethod
    def _vectorize_params(params: Dict[str, Any], search_space: Dict[str, Any]) -> List[float]:
        vector = []
        for name, spec in search_space.items():
            value = params.get(name)
            param_type = spec.get("type")
            if param_type == "categorical":
                choices = spec.get("choices", [])
                idx = choices.index(value) if value in choices else 0
                denom = max(len(choices) - 1, 1)
                vector.append(idx / denom)
                continue
            bounds = spec.get("bounds", [0.0, 1.0])
            low, high = bounds[0], bounds[1]
            if value is None:
                vector.append(0.0)
                continue
            if spec.get("scale") == "log" and value > 0 and low > 0 and high > 0:
                value = math.log10(float(value))
                low = math.log10(float(low))
                high = math.log10(float(high))
            denom = (high - low) if (high - low) != 0 else 1.0
            vector.append((float(value) - float(low)) / denom)
        return vector

    @staticmethod
    def _l2(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
