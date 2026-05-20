import json
import math
from typing import Dict, List, Optional


class UncertaintyCalibrator:
    """
    Simple uncertainty calibrator using global bias and scale.
    """

    def __init__(self, sigma_scale: float = 1.0, mu_bias: float = 0.0):
        self.sigma_scale = float(sigma_scale)
        self.mu_bias = float(mu_bias)
        self.records: List[Dict] = []

    def update(self, mu: float, sigma: float, y: float) -> None:
        self.records.append({"mu": float(mu), "sigma": float(sigma), "y": float(y)})

    def fit(self) -> None:
        if len(self.records) < 2:
            return
        residuals = [r["y"] - r["mu"] for r in self.records]
        mean_res = sum(residuals) / len(residuals)
        self.mu_bias = mean_res

        sigma_pred = [max(r["sigma"], 1e-12) for r in self.records]
        mean_sigma = sum(sigma_pred) / len(sigma_pred)
        mean_res_sq = sum([(r - mean_res) ** 2 for r in residuals]) / max(len(residuals) - 1, 1)
        emp_std = mean_res_sq ** 0.5
        if mean_sigma > 0:
            scale = emp_std / mean_sigma
            self.sigma_scale = max(min(scale, 10.0), 0.1)

    def calibrate(self, mu: float, sigma: float) -> (float, float):
        mu_cal = float(mu) + self.mu_bias
        sigma_cal = float(sigma) * self.sigma_scale
        return mu_cal, sigma_cal

    def coverage(self, z: float) -> Optional[float]:
        if not self.records:
            return None
        count = 0
        for r in self.records:
            mu_cal, sigma_cal = self.calibrate(r["mu"], r["sigma"])
            sigma_cal = max(float(sigma_cal), 1e-12)
            if abs(r["y"] - mu_cal) <= z * sigma_cal:
                count += 1
        return count / len(self.records)

    def nll(self) -> Optional[float]:
        if not self.records:
            return None
        total = 0.0
        for r in self.records:
            mu_cal, sigma_cal = self.calibrate(r["mu"], r["sigma"])
            sigma_cal = max(float(sigma_cal), 1e-12)
            resid = float(r["y"]) - float(mu_cal)
            total += 0.5 * math.log(2.0 * math.pi * (sigma_cal ** 2)) + 0.5 * (resid ** 2) / (sigma_cal ** 2)
        return total / len(self.records)

    def calibration_error(self, levels: Optional[List[float]] = None) -> Optional[float]:
        if not self.records:
            return None
        if levels is None:
            levels = [0.50, 0.68, 0.80, 0.90, 0.95]
        errors = []
        for level in levels:
            z = self._z_from_level(float(level))
            empirical = self.coverage(z)
            if empirical is None:
                continue
            errors.append(abs(float(empirical) - float(level)))
        if not errors:
            return None
        return sum(errors) / len(errors)

    def report(self) -> Dict:
        return {
            "n": len(self.records),
            "mu_bias": self.mu_bias,
            "sigma_scale": self.sigma_scale,
            "coverage_1sigma": self.coverage(1.0),
            "coverage_2sigma": self.coverage(2.0),
            "nll": self.nll(),
            "calibration_error": self.calibration_error(),
        }

    @staticmethod
    def _z_from_level(level: float) -> float:
        if level <= 0.0 or level >= 1.0:
            return 0.0
        return math.sqrt(2.0) * _erf_inv(level)

    def save(self, path: str) -> None:
        data = {
            "sigma_scale": self.sigma_scale,
            "mu_bias": self.mu_bias,
            "records": self.records
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        self.sigma_scale = float(data.get("sigma_scale", 1.0))
        self.mu_bias = float(data.get("mu_bias", 0.0))
        self.records = data.get("records", [])


def _erf_inv(x: float) -> float:
    a = 0.147
    sign = 1.0 if x >= 0 else -1.0
    ln = math.log(1.0 - x ** 2)
    first = 2.0 / (math.pi * a) + ln / 2.0
    second = ln / a
    return sign * math.sqrt(math.sqrt(first ** 2 - second) - first)
