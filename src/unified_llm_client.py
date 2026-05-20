"""OpenAI-backed LLM client used by the paper artifact."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI


load_dotenv(find_dotenv(), override=False)


class UnifiedLLMClient:
    """Small OpenAI-only client for HPO candidate proposal.

    The original research workspace supported multiple providers. The public
    paper artifact keeps one provider path to reduce dependency and API-key
    ambiguity.
    """

    def __init__(self, model_name: str = "gpt-5.2", api_model: Optional[str] = None):
        self.model_name = model_name.lower()
        self.api_model = api_model or self._resolve_model(self.model_name)
        self.display_name = self.api_model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @staticmethod
    def _resolve_model(model_name: str) -> str:
        if "mini" in model_name:
            return "gpt-5-mini"
        if model_name.startswith("gpt-"):
            return model_name
        raise ValueError(f"Unsupported model for this artifact: {model_name}. Use an OpenAI GPT model.")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.api_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if "gpt-5" not in self.api_model:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format
        if seed is not None:
            kwargs["seed"] = int(seed)

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def generate_hyperparameters(
        self,
        task_description: str,
        search_space: Dict,
        num_candidates: int = 5,
        temperature: float = 0.7,
    ) -> list[Dict[str, Any]]:
        system_prompt = "You are an expert hyperparameter optimization assistant. Return JSON only."
        user_prompt = f"""
Task: {task_description}
Search space:
{json.dumps(search_space, indent=2)}

Return exactly {num_candidates} candidates as:
{{"candidates": [{{"params": {{}}, "mu": 0.0, "sigma": 0.0, "reason": "..."}}]}}
"""
        content = self.generate(system_prompt, user_prompt, temperature=temperature, max_tokens=1200)
        parsed = self._extract_json(content)
        candidates = parsed.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError("LLM response JSON has no list-valued 'candidates' field.")
        return candidates[:num_candidates]

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))
