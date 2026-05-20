import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class GPSurrogate:
    def __init__(self, search_space: Dict[str, Any]) -> None:
        self.search_space = search_space
        self.param_names = list(search_space.keys())
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.fitted = False

    def fit(self, trials: Sequence[Dict[str, Any]], metric_key: str) -> bool:
        X = []
        y = []
        for t in trials:
            if t.get("status") != "completed":
                continue
            params = t.get("params")
            metric = t.get(metric_key)
            if params is None or metric is None:
                continue
            X.append(self._encode_params(params))
            y.append(float(metric))
        if len(X) < 2:
            self.fitted = False
            return False
        try:
            self.model.fit(np.array(X), np.array(y))
            self.fitted = True
            return True
        except Exception:
            self.fitted = False
            return False

    def predict(self, candidates: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        if not self.fitted:
            raise RuntimeError("GP not fitted")
        X = [self._encode_params(c) for c in candidates]
        mu, std = self.model.predict(np.array(X), return_std=True)
        return list(mu), list(std)

    def _encode_params(self, params: Dict[str, Any]) -> List[float]:
        vector = []
        for name in self.param_names:
            spec = self.search_space[name]
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
