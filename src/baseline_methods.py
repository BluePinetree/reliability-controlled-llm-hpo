"""Baseline optimizers used by the release artifact."""

import numpy as np
from typing import Dict, List, Tuple, Callable
from scipy.stats import norm
from scipy.optimize import minimize
import itertools

def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.number))

def _coerce_param_spec(param_spec: Dict) -> Dict:
    if isinstance(param_spec, dict) and 'type' in param_spec:
        return param_spec
    if isinstance(param_spec, (list, tuple)):
        if len(param_spec) == 2 and all(_is_number(v) for v in param_spec):
            return {'type': 'float', 'scale': 'linear', 'bounds': [param_spec[0], param_spec[1]]}
        return {'type': 'categorical', 'choices': list(param_spec)}
    raise ValueError("Invalid parameter spec format")

def _default_grid_values(spec: Dict) -> List:
    param_type = spec.get('type')
    if param_type == 'categorical':
        return list(spec.get('choices', []))
    bounds = spec.get('bounds', [0, 0])
    low, high = bounds[0], bounds[1]
    scale = spec.get('scale', 'linear')
    if param_type == 'int':
        low = int(low)
        high = int(high)
        if high - low <= 10:
            return list(range(low, high + 1))
        mid = int(round((low + high) / 2))
        return [low, mid, high]
    if scale == 'log' and low > 0:
        log_low = np.log10(low)
        log_high = np.log10(high)
        mid = 10 ** ((log_low + log_high) / 2)
        return [low, mid, high]
    mid = (low + high) / 2
    return [low, mid, high]

def llm_ucb_score(mu: float, sigma: float, kappa: float) -> float:
    return float(mu) + float(kappa) * float(sigma)

def llm_ei_score(mu: float, sigma: float, best: float, maximize: bool = True, xi: float = 0.0) -> float:
    if best is None:
        return float(mu) if maximize else -float(mu)
    sigma = float(sigma)
    mu = float(mu)
    if sigma <= 0:
        if maximize:
            return max(0.0, mu - best - xi)
        return max(0.0, best - mu - xi)
    if maximize:
        improvement = mu - best - xi
    else:
        improvement = best - mu - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


class RandomSearch:
    """Uniform random search over the declared search-space schema."""
    
    def __init__(self, search_space: Dict[str, Tuple[float, float]]):
        self.search_space = search_space
        self.param_names = list(search_space.keys())
    
    def suggest(self, n_suggestions: int = 1) -> List[Dict[str, float]]:
        """
        Randomly suggest hyperparameters based on schema.
        """
        suggestions = []

        for _ in range(n_suggestions):
            params = {}
            for param_name in self.param_names:
                spec = _coerce_param_spec(self.search_space[param_name])
                param_type = spec.get("type")
                if param_type == "categorical":
                    params[param_name] = np.random.choice(spec.get("choices", []))
                    continue

                bounds = spec.get("bounds", [0, 0])
                low, high = bounds[0], bounds[1]
                scale = spec.get("scale", "linear")
                if param_type == "int":
                    low = int(low)
                    high = int(high)
                    if scale == "log" and low > 0:
                        log_value = np.random.uniform(np.log10(low), np.log10(high))
                        value = int(round(10 ** log_value))
                    else:
                        value = int(np.random.randint(low, high + 1))
                    value = max(min(value, high), low)
                else:
                    if scale == "log" and low > 0:
                        log_value = np.random.uniform(np.log10(low), np.log10(high))
                        value = 10 ** log_value
                    else:
                        value = np.random.uniform(low, high)
                params[param_name] = value

            suggestions.append(params)

        return suggestions


class GridSearch:
    """Deterministic grid search over categorical or default numeric grids."""
    
    def __init__(self, search_space: Dict[str, List[float]]):
        self.search_space = search_space
        self.param_names = list(search_space.keys())
        
        param_values = []
        for name in self.param_names:
            spec = _coerce_param_spec(self.search_space[name])
            values = spec.get('choices')
            if values is None:
                values = spec.get('grid')
            if values is None:
                values = _default_grid_values(spec)
            param_values.append(values)
        self.grid = list(itertools.product(*param_values))
        self.current_index = 0
    
    def suggest(self, n_suggestions: int = 1) -> List[Dict[str, float]]:
        """Return the next configurations from the finite grid."""
        suggestions = []
        
        for _ in range(n_suggestions):
            if self.current_index >= len(self.grid):
                break
            
            values = self.grid[self.current_index]
            params = dict(zip(self.param_names, values))
            suggestions.append(params)
            
            self.current_index += 1
        
        return suggestions
    
    def total_combinations(self) -> int:
        """Return the number of configurations in the grid."""
        return len(self.grid)


class BayesianOptimization:
    """Lightweight Gaussian-process UCB baseline."""
    
    def __init__(self, search_space: Dict[str, Tuple[float, float]], 
                 kappa: float = 2.576):
        self.search_space = search_space
        self.param_names = list(search_space.keys())
        self.kappa = kappa
        
        self.X_observed = []
        self.y_observed = []
    
    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize params to [0, 1] based on schema."""
        normalized = []
        for param_name in self.param_names:
            spec = _coerce_param_spec(self.search_space[param_name])
            value = params[param_name]
            param_type = spec.get('type')
            if param_type == 'categorical':
                choices = spec.get('choices', [])
                idx = choices.index(value) if value in choices else 0
                denom = max(len(choices) - 1, 1)
                norm_value = idx / denom
            else:
                bounds = spec.get('bounds', [0, 1])
                low, high = bounds[0], bounds[1]
                scale = spec.get('scale', 'linear')
                if scale == 'log' and low > 0 and value > 0:
                    log_low, log_high = np.log10(low), np.log10(high)
                    denom = (log_high - log_low) if (log_high - log_low) != 0 else 1.0
                    norm_value = (np.log10(value) - log_low) / denom
                else:
                    denom = (high - low) if (high - low) != 0 else 1.0
                    norm_value = (value - low) / denom
            normalized.append(float(norm_value))
        return np.array(normalized)
    
    def _denormalize_params(self, normalized: np.ndarray) -> Dict[str, float]:
        """Denormalize params from [0, 1] based on schema."""
        params = {}
        for i, param_name in enumerate(self.param_names):
            spec = _coerce_param_spec(self.search_space[param_name])
            param_type = spec.get('type')
            norm_value = float(normalized[i])
            if param_type == 'categorical':
                choices = spec.get('choices', [])
                if not choices:
                    value = None
                else:
                    idx = int(round(norm_value * (len(choices) - 1)))
                    idx = max(min(idx, len(choices) - 1), 0)
                    value = choices[idx]
            else:
                bounds = spec.get('bounds', [0, 1])
                low, high = bounds[0], bounds[1]
                scale = spec.get('scale', 'linear')
                if scale == 'log' and low > 0:
                    log_low, log_high = np.log10(low), np.log10(high)
                    log_value = log_low + norm_value * (log_high - log_low)
                    value = 10 ** log_value
                else:
                    value = low + norm_value * (high - low)
                if param_type == 'int':
                    value = int(round(value))
                    value = max(min(value, int(high)), int(low))
            params[param_name] = value
        return params
    
    def _gaussian_process(self, X_new: np.ndarray) -> Tuple[float, float]:
        """Predict mean and standard deviation at a normalized candidate."""
        if len(self.X_observed) == 0:
            return 0.0, 1.0
        
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        
        def rbf_kernel(x1, x2, length_scale=0.1):
            return np.exp(-np.sum((x1 - x2) ** 2) / (2 * length_scale ** 2))
        
        # K(X*, X)
        k_star = np.array([rbf_kernel(X_new, x) for x in X_obs])
        
        # K(X, X) + noise
        K = np.array([[rbf_kernel(x1, x2) for x2 in X_obs] for x1 in X_obs])
        K += 1e-6 * np.eye(len(X_obs))  # noise
        
        # Mean and variance
        try:
            K_inv = np.linalg.inv(K)
            mean = k_star @ K_inv @ y_obs
            variance = 1 - k_star @ K_inv @ k_star
            std = np.sqrt(max(variance, 0))
        except:
            mean = np.mean(y_obs)
            std = np.std(y_obs)
        
        return mean, std
    
    def _ucb(self, X_new: np.ndarray) -> float:
        """Return the UCB acquisition value for one normalized candidate."""
        mean, std = self._gaussian_process(X_new)
        return mean + self.kappa * std
    
    def suggest(self, n_suggestions: int = 1) -> List[Dict[str, float]]:
        """Suggest configurations by maximizing UCB in normalized space."""
        suggestions = []
        
        for _ in range(n_suggestions):
            def neg_ucb(x):
                return -self._ucb(x)
            
            x0 = np.random.uniform(0, 1, len(self.param_names))
            
            result = minimize(neg_ucb, x0, bounds=[(0, 1)] * len(self.param_names), method='L-BFGS-B')
            
            params = self._denormalize_params(result.x)
            suggestions.append(params)
        
        return suggestions
    
    def update(self, params: Dict[str, float], accuracy: float):
        """Add one observed configuration and score."""
        X_new = self._normalize_params(params)
        self.X_observed.append(X_new)
        self.y_observed.append(accuracy)


class Hyperband:
    """Hyperband bracket scheduler."""
    
    def __init__(self, search_space: Dict[str, Tuple[float, float]], 
                 max_iter: int = 81, eta: int = 3):
        self.search_space = search_space
        self.max_iter = max_iter
        self.eta = eta
        
        self.s_max = int(np.log(max_iter) / np.log(eta))
        self.B = (self.s_max + 1) * max_iter
    
    def get_bracket_schedule(self, s: int) -> List[Tuple[int, int]]:
        """Return the (number of configs, resource) schedule for one bracket."""
        n = int(np.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
        r = self.max_iter * self.eta ** (-s)
        
        schedule = []
        for i in range(s + 1):
            n_i = int(n * self.eta ** (-i))
            r_i = int(r * self.eta ** i)
            schedule.append((n_i, r_i))
        
        return schedule
    
    def suggest_initial_configs(self, s: int) -> List[Dict[str, float]]:
        """Sample the initial configurations for a Hyperband bracket."""
        schedule = self.get_bracket_schedule(s)
        n_configs = schedule[0][0]
        
        # Random sampling
        random_search = RandomSearch(self.search_space)
        return random_search.suggest(n_configs)

