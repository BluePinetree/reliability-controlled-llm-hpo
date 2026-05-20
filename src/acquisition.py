from typing import Optional

from scipy.stats import norm


def expected_improvement(
    mu: float,
    sigma: float,
    best: Optional[float],
    maximize: bool = True,
    xi: float = 0.0,
) -> float:
    if best is None:
        return float(mu)
    sigma = float(sigma)
    mu = float(mu)
    if sigma <= 0:
        if maximize:
            return max(0.0, mu - best - xi)
        return max(0.0, best - mu - xi)
    improvement = (mu - best - xi) if maximize else (best - mu - xi)
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def upper_confidence_bound(mu: float, sigma: float, kappa: float, maximize: bool = True) -> float:
    mu = float(mu)
    sigma = float(sigma)
    score = mu + float(kappa) * sigma
    return score if maximize else -score
