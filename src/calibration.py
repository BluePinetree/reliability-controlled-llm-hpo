import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CalibrationReport:
    n: int
    mu_bias: float
    sigma_scale: float
    coverage_68: Optional[float]
    coverage_95: Optional[float]
    nll: Optional[float]
    calibration_error: Optional[float]


@dataclass
class GatingRule:
    min_coverage_68: float = 0.55
    max_coverage_68: float = 0.80
    max_nll: float = 2.0
    min_records: int = 8


class UncertaintyCalibrator:
    def __init__(self, sigma_scale: float = 1.0, mu_bias: float = 0.0) -> None:
        self.sigma_scale = float(sigma_scale)
        self.mu_bias = float(mu_bias)
        self.records: List[Dict[str, float]] = []

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

    def calibrate(self, mu: float, sigma: float) -> Tuple[float, float]:
        mu_cal = float(mu) + self.mu_bias
        sigma_cal = float(sigma) * self.sigma_scale
        return mu_cal, sigma_cal

    def coverage(self, z: float) -> Optional[float]:
        if not self.records:
            return None
        count = 0
        for r in self.records:
            mu_cal, sigma_cal = self.calibrate(r["mu"], r["sigma"])
            sigma_cal = max(sigma_cal, 1e-12)
            if abs(r["y"] - mu_cal) <= z * sigma_cal:
                count += 1
        return count / len(self.records)

    def nll(self) -> Optional[float]:
        if not self.records:
            return None
        total = 0.0
        for r in self.records:
            mu_cal, sigma_cal = self.calibrate(r["mu"], r["sigma"])
            sigma_cal = max(sigma_cal, 1e-12)
            resid = r["y"] - mu_cal
            total += 0.5 * math.log(2 * math.pi * sigma_cal ** 2) + 0.5 * (resid ** 2) / (sigma_cal ** 2)
        return total / len(self.records)

    def calibration_error(self, levels: Optional[List[float]] = None) -> Optional[float]:
        if not self.records:
            return None
        if levels is None:
            levels = [i / 10 for i in range(1, 10)]
        errors = []
        for level in levels:
            z = self._z_from_level(level)
            emp = self.coverage(z)
            if emp is None:
                continue
            errors.append(abs(emp - level))
        if not errors:
            return None
        return sum(errors) / len(errors)

    def report(self) -> CalibrationReport:
        return CalibrationReport(
            n=len(self.records),
            mu_bias=self.mu_bias,
            sigma_scale=self.sigma_scale,
            coverage_68=self.coverage(1.0),
            coverage_95=self.coverage(1.96),
            nll=self.nll(),
            calibration_error=self.calibration_error(),
        )

    def gate(self, rule: Optional[GatingRule] = None) -> Tuple[bool, str, CalibrationReport]:
        if rule is None:
            rule = GatingRule()
        report = self.report()
        if report.n < rule.min_records:
            return False, "insufficient_records", report
        if report.coverage_68 is None or report.nll is None:
            return False, "missing_metrics", report
        if report.coverage_68 < rule.min_coverage_68 or report.coverage_68 > rule.max_coverage_68:
            return False, "coverage_out_of_range", report
        if report.nll > rule.max_nll:
            return False, "nll_too_high", report
        return True, "pass", report

    @staticmethod
    def _z_from_level(level: float) -> float:
        if level <= 0 or level >= 1:
            return 0.0
        return math.sqrt(2) * _erf_inv(level)


# Inverse error function approximation (AS241 style)

def _erf_inv(x: float) -> float:
    a = 0.147
    sign = 1.0 if x >= 0 else -1.0
    ln = math.log(1 - x ** 2)
    first = 2 / (math.pi * a) + ln / 2
    second = ln / a
    return sign * math.sqrt(math.sqrt(first ** 2 - second) - first)
