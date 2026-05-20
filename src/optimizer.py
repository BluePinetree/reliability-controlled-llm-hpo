import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from acquisition import expected_improvement, upper_confidence_bound
from baselines import GPBOSearch, RandomSearch, TPESearch, lhs_sample
from calibration import GatingRule, UncertaintyCalibrator
from gp_surrogate import GPSurrogate
from llm_client import LLMClient
from prompt_builder import PromptBuilder


@dataclass
class TrialRecord:
    trial_id: int
    params: Dict[str, Any]
    metric: float
    status: str
    seed: int
    wall_time: float
    timestamp: str
    notes: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    acq_type: str = "ei"
    kappa: float = 2.0
    xi: float = 0.0
    gp_rerank: bool = True
    gating: bool = True
    fusion_mode: bool = False
    fusion_weight_llm: float = 0.3
    fusion_weight_gp: float = 0.7
    fallback_type: str = "random"  # random | lhs | gp_bo | tpe
    fallback_candidates: int = 1
    gating_rule: GatingRule = field(default_factory=GatingRule)


class CalGatedOptimizer:
    def __init__(
        self,
        search_space: Dict[str, Any],
        objective: Dict[str, Any],
        llm_client: LLMClient,
        config: OptimizerConfig,
        maximize: bool,
    ) -> None:
        self.search_space = search_space
        self.objective = objective
        self.llm_client = llm_client
        self.config = config
        self.maximize = maximize
        self.prompt_builder = PromptBuilder()
        self.calibrator = UncertaintyCalibrator()
        self.gp = GPSurrogate(search_space)
        self.gp_bo = GPBOSearch(search_space, acq_type=config.acq_type, kappa=config.kappa, xi=config.xi)
        self._tpe = None

    def suggest(self, trials: List[Dict[str, Any]], trial_id: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        system_prompt = (
            "You are a hyperparameter proposer. "
            "Return JSON that matches the schema."
        )
        prompt = self.prompt_builder.build(trials, self.search_space, self.objective, self.maximize)
        llm_result = self.llm_client.generate_candidates(system_prompt, prompt, self.search_space)

        gating_pass, gating_reason, cal_report = self.calibrator.gate(self.config.gating_rule)
        if not self.config.gating:
            gating_pass = True
            gating_reason = "gating_disabled"

        context = {
            "llm_raw_response": llm_result.raw_response,
            "llm_global_notes": llm_result.global_notes,
            "llm_candidates": llm_result.candidates,
            "llm_stats": dict(llm_result.stats.__dict__),
            "calibration_report": cal_report.__dict__,
            "gating_pass": gating_pass,
            "gating_reason": gating_reason,
        }

        if not llm_result.candidates:
            params = self._fallback(trials, trial_id, context, reason="llm_empty")
            return params, context
        if self.config.gating and not gating_pass:
            params = self._fallback(trials, trial_id, context, reason="gating_fail")
            return params, context

        if not self.config.gp_rerank:
            top_candidate = llm_result.candidates[0]
            context["selection_mode"] = "llm_top1"
            context["selected_candidate"] = top_candidate
            return top_candidate["params"], context

        candidates = self._attach_llm_calibration(llm_result.candidates, context)
        gp_ok = self.gp.fit(trials, self.objective.get("metric", "metric"))
        if not gp_ok:
            params = self._fallback(trials, trial_id, context, reason="gp_fit_fail")
            return params, context

        params_list = [c["params"] for c in candidates]
        try:
            mu_gp, sigma_gp = self.gp.predict(params_list)
        except Exception:
            params = self._fallback(trials, trial_id, context, reason="gp_predict_fail")
            return params, context

        best = self._best_so_far(trials)
        scored = []
        for cand, mu, sigma in zip(candidates, mu_gp, sigma_gp):
            mu_use = mu
            sigma_use = sigma
            if self.config.fusion_mode and gating_pass:
                mu_use = (self.config.fusion_weight_gp * mu) + (self.config.fusion_weight_llm * cand["mu_llm_cal"])
                sigma_use = max(sigma, cand["sigma_llm_cal"])
            if self.config.acq_type == "ei":
                score = expected_improvement(mu_use, sigma_use, best, maximize=self.maximize, xi=self.config.xi)
            else:
                score = upper_confidence_bound(mu_use, sigma_use, self.config.kappa, maximize=self.maximize)
            scored.append({
                "params": cand["params"],
                "mu_gp": mu,
                "sigma_gp": sigma,
                "mu_llm_raw": cand["mu_llm_raw"],
                "sigma_llm_raw": cand["sigma_llm_raw"],
                "mu_llm_cal": cand["mu_llm_cal"],
                "sigma_llm_cal": cand["sigma_llm_cal"],
                "score": score,
            })

        if not scored:
            params = self._fallback(trials, trial_id, context, reason="no_scored_candidates")
            return params, context

        selected = max(scored, key=lambda s: s["score"])
        context["selection_mode"] = "gp_rerank"
        context["candidate_scores"] = scored
        context["selected_candidate"] = selected
        return selected["params"], context

    def update(self, trial_record: Dict[str, Any]) -> None:
        context = trial_record.get("context", {})
        selected = context.get("selected_candidate") or {}
        mu_raw = selected.get("mu_llm_raw", selected.get("mu"))
        sigma_raw = selected.get("sigma_llm_raw", selected.get("sigma"))
        metric = trial_record.get("metric")
        if mu_raw is not None and sigma_raw is not None and metric is not None:
            self.calibrator.update(float(mu_raw), float(sigma_raw), float(metric))
            self.calibrator.fit()

    def _attach_llm_calibration(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        enriched = []
        for cand in candidates:
            mu_raw = float(cand["mu"])
            sigma_raw = float(cand["sigma"])
            mu_cal, sigma_cal = self.calibrator.calibrate(mu_raw, sigma_raw)
            enriched.append({
                "params": cand["params"],
                "mu_llm_raw": mu_raw,
                "sigma_llm_raw": sigma_raw,
                "mu_llm_cal": mu_cal,
                "sigma_llm_cal": sigma_cal,
                "reason": cand.get("reason", ""),
            })
        context["llm_candidates_calibrated"] = enriched
        return enriched

    def _best_so_far(self, trials: List[Dict[str, Any]]) -> Optional[float]:
        values = [t.get("metric") for t in trials if t.get("status") == "completed"]
        values = [v for v in values if v is not None]
        if not values:
            return None
        return max(values) if self.maximize else min(values)

    def _fallback(self, trials: List[Dict[str, Any]], trial_id: int, context: Dict[str, Any], reason: str) -> Dict[str, Any]:
        context["selection_mode"] = "fallback"
        context["fallback_reason"] = reason
        if self.config.fallback_type == "lhs":
            params = lhs_sample(self.search_space, self.config.fallback_candidates, seed=trial_id)[0]
        elif self.config.fallback_type == "gp_bo":
            params = self.gp_bo.suggest(trials, self.objective.get("metric", "metric"), self.maximize, seed=trial_id)
        elif self.config.fallback_type == "tpe":
            if self._tpe is None:
                self._tpe = TPESearch(self.search_space, seed=trial_id, maximize=self.maximize)
            params, _ = self._tpe.suggest()
        else:
            params = RandomSearch(self.search_space, seed=trial_id).suggest(1)[0]
        context["fallback_params"] = params
        return params
