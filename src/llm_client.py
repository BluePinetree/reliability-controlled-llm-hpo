import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from unified_llm_client import UnifiedLLMClient


@dataclass
class LLMStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_sec: float = 0.0
    parse_failures: int = 0
    invalid_params: int = 0
    total_candidates: int = 0


@dataclass
class LLMResult:
    candidates: List[Dict[str, Any]]
    raw_response: str
    global_notes: str
    stats: LLMStats


class LLMClient:
    def __init__(
        self,
        model_name: str,
        num_candidates: int,
        temperature: float,
        seed: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.num_candidates = int(num_candidates)
        self.temperature = float(temperature)
        self.seed = seed
        self.client = UnifiedLLMClient(model_name=model_name)
        self.stats = LLMStats()

    def build_response_schema(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        params_props = {}
        required_params = []
        for name, spec in search_space.items():
            param_type = spec.get("type")
            schema = {}
            if param_type == "int":
                schema = {"type": "integer"}
                bounds = spec.get("bounds")
                if isinstance(bounds, list) and len(bounds) == 2:
                    schema["minimum"] = int(bounds[0])
                    schema["maximum"] = int(bounds[1])
            elif param_type == "float":
                schema = {"type": "number"}
                bounds = spec.get("bounds")
                if isinstance(bounds, list) and len(bounds) == 2:
                    schema["minimum"] = float(bounds[0])
                    schema["maximum"] = float(bounds[1])
            elif param_type == "categorical":
                schema = {"type": "string"}
            else:
                schema = {"type": ["string", "number", "integer", "boolean"]}
            params_props[name] = schema
            required_params.append(name)

        candidate_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "params": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": params_props,
                    "required": required_params,
                },
                "mu": {"type": "number"},
                "sigma": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["params", "mu", "sigma", "reason"],
        }

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": self.num_candidates,
                    "maxItems": self.num_candidates,
                    "items": candidate_schema,
                },
                "global_notes": {"type": "string"},
            },
            "required": ["candidates", "global_notes"],
        }

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "hpo_candidates_v2",
                "schema": schema,
                "strict": True,
            },
        }

    def generate_candidates(
        self,
        system_prompt: str,
        user_prompt: str,
        search_space: Dict[str, Any],
    ) -> LLMResult:
        start = time.time()
        response_format = self.build_response_schema(search_space)
        response = self.client.generate(
            system_prompt,
            user_prompt,
            temperature=self.temperature,
            response_format=response_format,
            seed=self.seed,
        )
        elapsed = time.time() - start

        self.stats.calls += 1
        self.stats.wall_time_sec += elapsed
        self.stats.prompt_tokens += self._estimate_tokens(system_prompt + "\n" + user_prompt)
        self.stats.completion_tokens += self._estimate_tokens(response)

        payload = self._extract_json(response)
        if payload is None:
            self.stats.parse_failures += 1
            return LLMResult([], response, "", self.stats)

        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        global_notes = payload.get("global_notes") if isinstance(payload, dict) else ""
        if not isinstance(candidates, list):
            self.stats.parse_failures += 1
            return LLMResult([], response, global_notes or "", self.stats)

        valid = []
        invalid = 0
        for cand in candidates:
            cleaned, ok = self._validate_and_clip_candidate(cand, search_space)
            if ok:
                valid.append(cleaned)
            else:
                invalid += 1

        self.stats.invalid_params += invalid
        self.stats.total_candidates += len(candidates)

        return LLMResult(valid, response, global_notes or "", self.stats)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def _validate_and_clip_candidate(
        candidate: Dict[str, Any],
        search_space: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        if not isinstance(candidate, dict):
            return {}, False
        params = candidate.get("params")
        mu = candidate.get("mu")
        sigma = candidate.get("sigma")
        reason = candidate.get("reason", "")
        if not isinstance(params, dict) or not isinstance(mu, (int, float)) or not isinstance(sigma, (int, float)):
            return {}, False
        if sigma <= 0:
            return {}, False

        cleaned = {}
        valid = True
        for name, spec in search_space.items():
            if name not in params:
                valid = False
                continue
            value = params[name]
            param_type = spec.get("type")
            if param_type == "categorical":
                choices = spec.get("choices", [])
                if value not in choices:
                    valid = False
                    value = choices[0] if choices else value
                cleaned[name] = value
                continue

            if not isinstance(value, (int, float)):
                valid = False
                continue

            bounds = spec.get("bounds", [None, None])
            low, high = bounds[0], bounds[1]
            if param_type == "int":
                value = int(round(value))
                if low is not None and value < int(low):
                    value = int(low)
                    valid = False
                if high is not None and value > int(high):
                    value = int(high)
                    valid = False
            else:
                value = float(value)
                if low is not None and value < float(low):
                    value = float(low)
                    valid = False
                if high is not None and value > float(high):
                    value = float(high)
                    valid = False
            cleaned[name] = value

        cleaned_candidate = {
            "params": cleaned,
            "mu": float(mu),
            "sigma": float(sigma),
            "reason": str(reason),
        }
        return cleaned_candidate, valid
