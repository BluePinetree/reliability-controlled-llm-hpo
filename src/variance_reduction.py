"""Sampling utilities for variance-controlled HPO experiment design."""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.number))


def _coerce_param_spec(param_spec: Dict) -> Dict:
    if isinstance(param_spec, dict) and "type" in param_spec:
        return param_spec
    if isinstance(param_spec, (list, tuple)):
        if len(param_spec) == 2 and all(_is_number(v) for v in param_spec):
            return {"type": "float", "scale": "linear", "bounds": [param_spec[0], param_spec[1]]}
        return {"type": "categorical", "choices": list(param_spec)}
    raise ValueError("Invalid parameter spec format")


@dataclass
class HyperparameterConfig:
    """Sampled hyperparameter configuration with optional variance-design metadata."""

    config_id: int
    params: Dict[str, float]
    seed: Optional[int] = None
    stratum: Optional[int] = None
    is_antithetic_pair: bool = False
    pair_id: Optional[int] = None


class VarianceReduction:
    """Generate stratified, antithetic, mixed, and random HPO samples."""

    def __init__(self, search_space: Dict[str, Tuple[float, float]]) -> None:
        self.search_space = search_space
        self.param_names = list(search_space.keys())

    def stratified_sampling(
        self,
        n_samples: int,
        n_strata_per_dim: int = 3,
    ) -> List[HyperparameterConfig]:
        """Sample configurations from a regular stratum grid."""

        configs = []
        n_dims = len(self.param_names)
        total_strata = n_strata_per_dim ** n_dims
        samples_per_stratum = max(1, n_samples // total_strata)
        stratum_indices = list(itertools.product(range(n_strata_per_dim), repeat=n_dims))

        config_id = 0
        for stratum_id, stratum_idx in enumerate(stratum_indices):
            for _ in range(samples_per_stratum):
                params = {}
                for dim, param_name in enumerate(self.param_names):
                    params[param_name] = self._sample_from_stratum(
                        self.search_space[param_name],
                        stratum_idx[dim],
                        n_strata_per_dim,
                    )

                configs.append(HyperparameterConfig(
                    config_id=config_id,
                    params=params,
                    stratum=stratum_id,
                ))
                config_id += 1

                if len(configs) >= n_samples:
                    break
            if len(configs) >= n_samples:
                break

        return configs[:n_samples]

    def antithetic_sampling(self, n_pairs: int) -> List[HyperparameterConfig]:
        """Sample paired configurations reflected around each numeric range midpoint."""

        configs = []
        config_id = 0

        for pair_id in range(n_pairs):
            original_params = {}
            antithetic_params = {}

            for param_name in self.param_names:
                original, antithetic = self._sample_antithetic_pair(self.search_space[param_name])
                original_params[param_name] = original
                antithetic_params[param_name] = antithetic

            configs.append(HyperparameterConfig(
                config_id=config_id,
                params=original_params,
                is_antithetic_pair=True,
                pair_id=pair_id,
            ))
            config_id += 1

            configs.append(HyperparameterConfig(
                config_id=config_id,
                params=antithetic_params,
                is_antithetic_pair=True,
                pair_id=pair_id,
            ))
            config_id += 1

        return configs

    def combined_sampling(
        self,
        n_samples: int,
        n_strata_per_dim: int = 3,
        antithetic_ratio: float = 0.5,
    ) -> List[HyperparameterConfig]:
        """Combine stratified and antithetic samples."""

        n_antithetic = int(n_samples * antithetic_ratio)
        n_antithetic = (n_antithetic // 2) * 2
        n_stratified = n_samples - n_antithetic

        all_configs = (
            self.stratified_sampling(n_stratified, n_strata_per_dim)
            + self.antithetic_sampling(n_antithetic // 2)
        )
        return self._renumber(all_configs)

    def random_sampling(self, n_samples: int) -> List[HyperparameterConfig]:
        """Sample configurations uniformly from the declared search space."""

        configs = []
        for i in range(n_samples):
            params = {
                param_name: self._sample_random(self.search_space[param_name])
                for param_name in self.param_names
            }
            configs.append(HyperparameterConfig(config_id=i, params=params))
        return configs

    def mix_sampling(
        self,
        n_samples: int,
        mix_ratio: float = 0.0,
        n_strata_per_dim: int = 3,
        antithetic_ratio: float = 0.5,
    ) -> List[HyperparameterConfig]:
        """Mix structured samples with random samples."""

        mix_ratio = max(min(float(mix_ratio), 1.0), 0.0)
        n_random = int(round(n_samples * mix_ratio))
        n_structured = max(n_samples - n_random, 0)

        structured = self.combined_sampling(
            n_structured,
            n_strata_per_dim=n_strata_per_dim,
            antithetic_ratio=antithetic_ratio,
        )
        return self._renumber(structured + self.random_sampling(n_random))

    def add_multiple_seeds(
        self,
        configs: List[HyperparameterConfig],
        n_seeds: int = 3,
        seed_start: int = 42,
    ) -> List[HyperparameterConfig]:
        """Expand configurations across a fixed sequence of random seeds."""

        expanded_configs = []
        config_id = 0
        for config in configs:
            for seed_idx in range(n_seeds):
                expanded_configs.append(HyperparameterConfig(
                    config_id=config_id,
                    params=config.params.copy(),
                    seed=seed_start + seed_idx,
                    stratum=config.stratum,
                    is_antithetic_pair=config.is_antithetic_pair,
                    pair_id=config.pair_id,
                ))
                config_id += 1
        return expanded_configs

    def _sample_from_stratum(self, param_spec: Dict, stratum_idx: int, n_strata: int):
        spec = _coerce_param_spec(param_spec)
        param_type = spec.get("type")
        if param_type == "categorical":
            return np.random.choice(spec.get("choices", []))

        low, high = spec.get("bounds", [0, 0])
        scale = spec.get("scale", "linear")
        if scale == "log" and low > 0:
            log_low = np.log10(low)
            log_high = np.log10(high)
            low_t = log_low + (log_high - log_low) * stratum_idx / n_strata
            high_t = log_low + (log_high - log_low) * (stratum_idx + 1) / n_strata
            value = 10 ** np.random.uniform(low_t, high_t)
        else:
            low_t = low + (high - low) * stratum_idx / n_strata
            high_t = low + (high - low) * (stratum_idx + 1) / n_strata
            value = np.random.uniform(low_t, high_t)
        return self._coerce_numeric_value(value, spec)

    def _sample_antithetic_pair(self, param_spec: Dict):
        spec = _coerce_param_spec(param_spec)
        param_type = spec.get("type")
        if param_type == "categorical":
            choices = spec.get("choices", [])
            return np.random.choice(choices), np.random.choice(choices)

        low, high = spec.get("bounds", [0, 0])
        scale = spec.get("scale", "linear")
        if scale == "log" and low > 0:
            log_low = np.log10(low)
            log_high = np.log10(high)
            log_mid = (log_low + log_high) / 2
            log_value = np.random.uniform(log_low, log_high)
            original = 10 ** log_value
            antithetic = 10 ** (2 * log_mid - log_value)
        else:
            mid = (low + high) / 2
            original = np.random.uniform(low, high)
            antithetic = 2 * mid - original
        return self._coerce_numeric_value(original, spec), self._coerce_numeric_value(antithetic, spec)

    def _sample_random(self, param_spec: Dict):
        spec = _coerce_param_spec(param_spec)
        param_type = spec.get("type")
        if param_type == "categorical":
            return np.random.choice(spec.get("choices", []))

        low, high = spec.get("bounds", [0, 0])
        scale = spec.get("scale", "linear")
        if scale == "log" and low > 0:
            value = 10 ** np.random.uniform(np.log10(low), np.log10(high))
        else:
            value = np.random.uniform(low, high)
        return self._coerce_numeric_value(value, spec)

    @staticmethod
    def _coerce_numeric_value(value, spec: Dict):
        if spec.get("type") == "int":
            low, high = spec.get("bounds", [0, 0])
            value = int(round(value))
            return max(min(value, int(high)), int(low))
        return value

    @staticmethod
    def _renumber(configs: List[HyperparameterConfig]) -> List[HyperparameterConfig]:
        for i, config in enumerate(configs):
            config.config_id = i
        return configs


class ControlVariates:
    """Apply a scalar baseline correction to measured accuracies."""

    def __init__(self, baseline_accuracy: float) -> None:
        self.baseline_accuracy = baseline_accuracy
        self.results = []

    def add_result(
        self,
        config: HyperparameterConfig,
        accuracy: float,
        baseline_accuracy_sample: Optional[float] = None,
    ) -> None:
        """Store raw and baseline-corrected accuracy for a configuration."""

        if baseline_accuracy_sample is None:
            baseline_accuracy_sample = self.baseline_accuracy
        controlled_accuracy = accuracy - (baseline_accuracy_sample - self.baseline_accuracy)
        self.results.append({
            "config": config,
            "raw_accuracy": accuracy,
            "controlled_accuracy": controlled_accuracy,
            "baseline_sample": baseline_accuracy_sample,
        })

    def get_statistics(self) -> Dict:
        """Return mean, standard deviation, and variance-reduction ratio."""

        if not self.results:
            return {}

        raw_accs = [r["raw_accuracy"] for r in self.results]
        controlled_accs = [r["controlled_accuracy"] for r in self.results]
        raw_var = np.var(raw_accs)

        return {
            "raw_mean": np.mean(raw_accs),
            "raw_std": np.std(raw_accs),
            "controlled_mean": np.mean(controlled_accs),
            "controlled_std": np.std(controlled_accs),
            "variance_reduction_ratio": 0.0 if raw_var == 0 else 1 - (np.var(controlled_accs) / raw_var),
        }
