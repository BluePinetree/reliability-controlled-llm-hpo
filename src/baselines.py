import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from acquisition import expected_improvement, upper_confidence_bound
from gp_surrogate import GPSurrogate


class RandomSearch:
    def __init__(self, search_space: Dict[str, Any], seed: Optional[int] = None) -> None:
        self.search_space = search_space
        self.rng = np.random.default_rng(seed)

    def suggest(self, n: int = 1) -> List[Dict[str, Any]]:
        return [self._sample_one() for _ in range(n)]

    def _sample_one(self) -> Dict[str, Any]:
        params = {}
        for name, spec in self.search_space.items():
            param_type = spec.get("type")
            if param_type == "categorical":
                choices = spec.get("choices", [])
                params[name] = choices[int(self.rng.integers(0, len(choices)))] if choices else None
                continue
            bounds = spec.get("bounds", [0.0, 1.0])
            low, high = bounds[0], bounds[1]
            if param_type == "int":
                if spec.get("scale") == "log" and low > 0:
                    value = int(round(10 ** self.rng.uniform(math.log10(low), math.log10(high))))
                else:
                    value = int(self.rng.integers(int(low), int(high) + 1))
                params[name] = max(min(value, int(high)), int(low))
            else:
                if spec.get("scale") == "log" and low > 0:
                    value = 10 ** self.rng.uniform(math.log10(low), math.log10(high))
                else:
                    value = float(self.rng.uniform(low, high))
                params[name] = float(value)
        return params


def lhs_sample(search_space: Dict[str, Any], n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    names = list(search_space.keys())
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    sample = sampler.random(n=n)
    params_list = []
    for row in sample:
        params = {}
        for i, name in enumerate(names):
            spec = search_space[name]
            param_type = spec.get("type")
            if param_type == "categorical":
                choices = spec.get("choices", [])
                idx = int(round(row[i] * max(len(choices) - 1, 0))) if choices else 0
                params[name] = choices[idx] if choices else None
                continue
            bounds = spec.get("bounds", [0.0, 1.0])
            low, high = bounds[0], bounds[1]
            if spec.get("scale") == "log" and low > 0:
                low_t = math.log10(low)
                high_t = math.log10(high)
                value = 10 ** (low_t + row[i] * (high_t - low_t))
            else:
                value = low + row[i] * (high - low)
            if param_type == "int":
                value = int(round(value))
            params[name] = value
        params_list.append(params)
    return params_list


class GPBOSearch:
    def __init__(
        self,
        search_space: Dict[str, Any],
        acq_type: str = "ucb",
        kappa: float = 2.0,
        xi: float = 0.0,
    ) -> None:
        self.search_space = search_space
        self.acq_type = acq_type
        self.kappa = kappa
        self.xi = xi
        self.gp = GPSurrogate(search_space)

    def suggest(
        self,
        trials: List[Dict[str, Any]],
        metric_key: str,
        maximize: bool,
        n_candidates: int = 128,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        gp_ok = self.gp.fit(trials, metric_key)
        if not gp_ok:
            return RandomSearch(self.search_space, seed=seed)._sample_one()
        candidates = lhs_sample(self.search_space, n_candidates, seed=seed)
        mu, sigma = self.gp.predict(candidates)
        best = self._best_so_far(trials, metric_key, maximize)
        scores = []
        for m, s in zip(mu, sigma):
            if self.acq_type == "ei":
                score = expected_improvement(m, s, best, maximize=maximize, xi=self.xi)
            else:
                score = upper_confidence_bound(m, s, self.kappa, maximize=maximize)
            scores.append(score)
        best_idx = int(np.argmax(scores))
        return candidates[best_idx]

    @staticmethod
    def _best_so_far(trials: List[Dict[str, Any]], metric_key: str, maximize: bool) -> Optional[float]:
        values = [t.get(metric_key) for t in trials if t.get("status") == "completed"]
        values = [v for v in values if v is not None]
        if not values:
            return None
        return max(values) if maximize else min(values)


class TPESearch:
    def __init__(
        self,
        search_space: Dict[str, Any],
        seed: Optional[int],
        maximize: bool,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 64,
        multivariate: bool = False,
        group: bool = False,
        constant_liar: bool = False,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
    ) -> None:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("optuna is required for TPE baseline") from exc
        self.search_space = search_space
        self.direction = "maximize" if maximize else "minimize"
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=max(int(n_startup_trials), 0),
            n_ei_candidates=max(int(n_ei_candidates), 1),
            multivariate=bool(multivariate),
            group=bool(group),
            constant_liar=bool(constant_liar),
            consider_prior=bool(consider_prior),
            prior_weight=float(prior_weight),
        )
        self.study = optuna.create_study(direction=self.direction, sampler=sampler)
        self._pending = {}

    def suggest(self) -> Tuple[Dict[str, Any], int]:
        trial = self.study.ask()
        params = {}
        for name, spec in self.search_space.items():
            param_type = spec.get("type")
            if param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec.get("choices", []))
                continue
            bounds = spec.get("bounds", [0.0, 1.0])
            low, high = bounds[0], bounds[1]
            if param_type == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                log = spec.get("scale") == "log"
                params[name] = trial.suggest_float(name, float(low), float(high), log=log)
        self._pending[trial.number] = trial
        return params, trial.number

    def update(self, trial_number: int, value: float) -> None:
        trial = self._pending.pop(trial_number, None)
        if trial is not None:
            self.study.tell(trial, value)


class TPEPrunedSearch:
    """
    Optuna TPE baseline with a configurable pruner.
    Note: pruning decisions are made from reported metric steps in update().
    """

    def __init__(
        self,
        search_space: Dict[str, Any],
        seed: Optional[int],
        maximize: bool,
        pruner_type: str = "median",
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        interval_steps: int = 1,
        n_ei_candidates: int = 64,
        multivariate: bool = False,
        group: bool = False,
        constant_liar: bool = False,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
    ) -> None:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("optuna is required for TPE+Pruner baseline") from exc

        self._optuna = optuna
        self.search_space = search_space
        self.direction = "maximize" if maximize else "minimize"
        self.pruner_type = (pruner_type or "median").lower()

        if self.pruner_type == "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=max(int(n_startup_trials), 0),
                n_warmup_steps=max(int(n_warmup_steps), 0),
                interval_steps=max(int(interval_steps), 1),
            )
        elif self.pruner_type == "percentile":
            pruner = optuna.pruners.PercentilePruner(
                percentile=25.0,
                n_startup_trials=max(int(n_startup_trials), 0),
                n_warmup_steps=max(int(n_warmup_steps), 0),
                interval_steps=max(int(interval_steps), 1),
            )
        elif self.pruner_type == "none":
            pruner = optuna.pruners.NopPruner()
        else:
            raise ValueError(
                f"Unsupported pruner_type: {pruner_type}. "
                "Use one of: median, percentile, none."
            )

        self.study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(
                seed=seed,
                n_startup_trials=max(int(n_startup_trials), 0),
                n_ei_candidates=max(int(n_ei_candidates), 1),
                multivariate=bool(multivariate),
                group=bool(group),
                constant_liar=bool(constant_liar),
                consider_prior=bool(consider_prior),
                prior_weight=float(prior_weight),
            ),
            pruner=pruner,
        )
        self._pending = {}

    def suggest(self) -> Tuple[Dict[str, Any], int]:
        trial = self.study.ask()
        params = {}
        for name, spec in self.search_space.items():
            param_type = spec.get("type")
            if param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec.get("choices", []))
                continue
            bounds = spec.get("bounds", [0.0, 1.0])
            low, high = bounds[0], bounds[1]
            if param_type == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                log = spec.get("scale") == "log"
                params[name] = trial.suggest_float(name, float(low), float(high), log=log)
        self._pending[trial.number] = trial
        return params, trial.number

    def update(self, trial_number: int, value: float, step: int = 0) -> bool:
        trial = self._pending.pop(trial_number, None)
        if trial is None:
            return False

        step = max(int(step), 0)
        trial.report(float(value), step=step)
        if trial.should_prune():
            self.study.tell(trial, state=self._optuna.trial.TrialState.PRUNED)
            return True

        self.study.tell(trial, float(value))
        return False


class SMACSearch:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("SMAC is outside the scope of this release artifact.")


class TPEMultivariateSearch(TPESearch):
    def __init__(
        self,
        search_space: Dict[str, Any],
        seed: Optional[int],
        maximize: bool,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 64,
        constant_liar: bool = True,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
    ) -> None:
        super().__init__(
            search_space=search_space,
            seed=seed,
            maximize=maximize,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            multivariate=True,
            group=True,
            constant_liar=constant_liar,
            consider_prior=consider_prior,
            prior_weight=prior_weight,
        )
