"""
Unified Experiment Runner

Main experiment runner for the release artifact.
"""

import os
import json
import argparse
import hashlib
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from episodic_memory import EpisodicMemory
from variance_reduction import VarianceReduction
from uncertainty_calibration import UncertaintyCalibrator
from statistical_significance import StatisticalSignificance
from paper_datasets import DatasetLoader
from baseline_methods import RandomSearch, GridSearch, BayesianOptimization, llm_ucb_score, llm_ei_score
from baselines import TPESearch, TPEPrunedSearch, TPEMultivariateSearch
from unified_llm_client import UnifiedLLMClient
from theoretical_analysis import ConvergenceAnalyzer, TheoreticalBoundEstimator
from model_trainer import CVModelTrainer, NLPModelTrainer, TabularModelTrainer


class ExperimentConfig:
    """Main experiment runner for the release artifact."""
    
    def __init__(self, config_dict: Dict):
        self.experiment_name = config_dict.get('experiment_name', 'default_experiment')
        self.dataset_name = config_dict.get('dataset', 'cifar100')
        self.domain = config_dict.get('domain', 'CV')
        self.method = config_dict.get('method', 'LLM+Episodic')
        self.llm_model = config_dict.get('llm_model', 'gpt-5.2-mini')
        self.llm_num_candidates = config_dict.get('llm_num_candidates', 5)
        self.llm_temperature = config_dict.get('llm_temperature', 0.7)
        self.llm_output_schema_version = config_dict.get('llm_output_schema_version', 'v1')
        self.llm_seed = config_dict.get('llm_seed', None)
        self.acq_type = config_dict.get('acq_type', 'ucb')
        self.kappa = config_dict.get('kappa', 2.0)
        self.exploration_schedule = config_dict.get('exploration_schedule', {'type': 'constant'})
        self.mix_random_ratio = config_dict.get('mix_random_ratio', 0.0)
        self.warm_start_trials = config_dict.get('warm_start_trials', 0)
        self.objective_metric = config_dict.get('objective_metric', 'accuracy')
        self.maximize = config_dict.get('maximize', True)
        self.noise_model = config_dict.get('noise_model', 'sub_gaussian')
        
        self.n_trials = config_dict.get('n_trials', 30)
        self.n_seeds = config_dict.get('n_seeds', 3)
        self.use_episodic_search = config_dict.get('use_episodic_search', True)
        self.use_meta_learning = config_dict.get('use_meta_learning', True)
        self.use_variance_reduction = config_dict.get('use_variance_reduction', True)
        self.k_similar_episodes = config_dict.get('k_similar_episodes', 5)
        self.budget_T = config_dict.get('budget_T', self.n_trials)
        if self.budget_T < self.n_trials:
            raise ValueError("budget_T must be >= n_trials")
        
        self.output_dir = config_dict.get('output_dir', './results')
        
        self.search_space = self._validate_search_space(config_dict.get('search_space', {
            'learning_rate': {'type': 'float', 'scale': 'log', 'bounds': [1e-5, 1e-1]},
            'batch_size': {'type': 'int', 'scale': 'linear', 'bounds': [16, 256]},
            'weight_decay': {'type': 'float', 'scale': 'linear', 'bounds': [0.0, 1e-3]}
        }))

        # Stability constraints and fail-safe controls.
        self.max_safe_learning_rate = config_dict.get('max_safe_learning_rate', None)
        self.dataset_max_safe_learning_rates = config_dict.get('dataset_max_safe_learning_rates', {})
        self.domain_max_safe_learning_rates = config_dict.get('domain_max_safe_learning_rates', {})
        self.dataset_risk_high_lr_thresholds = config_dict.get('dataset_risk_high_lr_thresholds', {})
        self.domain_risk_high_lr_thresholds = config_dict.get('domain_risk_high_lr_thresholds', {})
        self.dataset_fail_safe_low_accuracy = config_dict.get('dataset_fail_safe_low_accuracy', {})
        self.domain_fail_safe_low_accuracy = config_dict.get('domain_fail_safe_low_accuracy', {})
        self.enable_fail_safe = config_dict.get('enable_fail_safe', True)
        self.fail_safe_low_accuracy = float(config_dict.get('fail_safe_low_accuracy', 0.1))
        self.fail_safe_patience = int(config_dict.get('fail_safe_patience', 3))
        self.fail_safe_cooldown = int(config_dict.get('fail_safe_cooldown', 3))
        self.conservative_lr_max = float(config_dict.get('conservative_lr_max', 1e-2))
        self.enable_confidence_gated_episodic = bool(config_dict.get('enable_confidence_gated_episodic', True))
        self.episodic_gate_min_episodes = int(config_dict.get('episodic_gate_min_episodes', 8))
        self.episodic_gate_min_calibration_records = int(config_dict.get('episodic_gate_min_calibration_records', 8))
        self.episodic_gate_min_coverage = float(config_dict.get('episodic_gate_min_coverage', 0.55))
        self.episodic_gate_max_coverage = float(config_dict.get('episodic_gate_max_coverage', 0.85))
        self.episodic_gate_max_calibration_error = float(config_dict.get('episodic_gate_max_calibration_error', 0.20))
        self.episodic_gate_max_nll = float(config_dict.get('episodic_gate_max_nll', 2.0))
        self.enable_risk_aware_selection = bool(config_dict.get('enable_risk_aware_selection', True))
        self.risk_penalty_lambda = float(config_dict.get('risk_penalty_lambda', 0.50))
        self.param_risk_penalty_weight = float(config_dict.get('param_risk_penalty_weight', 0.15))
        self.risk_high_lr_threshold = float(config_dict.get('risk_high_lr_threshold', 1e-2))
        self.risk_high_dropout_threshold = float(config_dict.get('risk_high_dropout_threshold', 0.35))
        self.risk_high_mix_threshold = float(config_dict.get('risk_high_mix_threshold', 0.80))
        self.risk_high_cutmix_threshold = float(config_dict.get('risk_high_cutmix_threshold', 0.80))
        self.risk_high_warmup_threshold = int(config_dict.get('risk_high_warmup_threshold', 800))
        self.risk_small_batch_threshold = int(config_dict.get('risk_small_batch_threshold', 32))

        # Confidence-coupled reliability control options. Defaults preserve legacy behavior.
        self.reliability_control_mode = str(config_dict.get('reliability_control_mode', 'legacy')).lower()
        self.episodic_gate_mode = str(config_dict.get('episodic_gate_mode', 'binary')).lower()
        self.episodic_gate_partial_memory_ratio = float(config_dict.get('episodic_gate_partial_memory_ratio', 0.5))
        self.episodic_gate_partial_use_worst_diverse = bool(config_dict.get('episodic_gate_partial_use_worst_diverse', False))
        self.risk_sigma_penalty_mode = str(config_dict.get('risk_sigma_penalty_mode', 'legacy')).lower()
        self.risk_sigma_free_band = float(config_dict.get('risk_sigma_free_band', 0.03))
        self.hard_risk_weight = float(config_dict.get('hard_risk_weight', 1.0))
        self.soft_risk_weight = float(config_dict.get('soft_risk_weight', self.param_risk_penalty_weight))
        self.gate_coupled_soft_risk = bool(config_dict.get('gate_coupled_soft_risk', False))
        self.enable_near_optimal_rerank = bool(config_dict.get('enable_near_optimal_rerank', False))
        self.near_optimal_rerank_top_k = int(config_dict.get('near_optimal_rerank_top_k', 3))
        self.near_optimal_rerank_epsilon = float(config_dict.get('near_optimal_rerank_epsilon', 0.01))
        self.near_optimal_rerank_key = str(config_dict.get('near_optimal_rerank_key', 'soft_risk')).lower()

        # Optuna TPE baseline options.
        self.tpe_seed = config_dict.get('tpe_seed', 42)
        if self.tpe_seed is not None:
            self.tpe_seed = int(self.tpe_seed)
        self.tpe_n_startup_trials = int(config_dict.get('tpe_n_startup_trials', 10))
        self.tpe_n_ei_candidates = int(config_dict.get('tpe_n_ei_candidates', 64))
        self.tpe_consider_prior = bool(config_dict.get('tpe_consider_prior', True))
        self.tpe_prior_weight = float(config_dict.get('tpe_prior_weight', 1.0))
        self.tpe_constant_liar = bool(config_dict.get('tpe_constant_liar', False))
        self.tpe_mv_constant_liar = bool(config_dict.get('tpe_mv_constant_liar', True))
        self.tpe_pruner_type = str(config_dict.get('tpe_pruner_type', 'median'))
        self.tpe_pruner_n_startup_trials = int(config_dict.get('tpe_pruner_n_startup_trials', 5))
        self.tpe_pruner_n_warmup_steps = int(config_dict.get('tpe_pruner_n_warmup_steps', 0))
        self.tpe_pruner_interval_steps = int(config_dict.get('tpe_pruner_interval_steps', 1))
        self.tpe_pruner_n_ei_candidates = int(config_dict.get('tpe_pruner_n_ei_candidates', self.tpe_n_ei_candidates))
        self.report_best_at = [int(v) for v in config_dict.get('report_best_at', [1, 3, 5, 10, 20, 30])]
        self.report_accuracy_thresholds = [float(v) for v in config_dict.get('report_accuracy_thresholds', [0.50, 0.55, 0.58, 0.60])]
        self.state_dir = config_dict.get('state_dir', None)
        self.reset_calibration_state = bool(config_dict.get('reset_calibration_state', False))
        self.reset_episodic_memory = bool(config_dict.get('reset_episodic_memory', False))
        
        self._apply_domain_overrides()

    @staticmethod
    def _validate_search_space(search_space: Dict) -> Dict:
        if not isinstance(search_space, dict) or not search_space:
            raise ValueError("search_space must be a non-empty dict")
        normalized = {}
        for name, spec in search_space.items():
            if not isinstance(spec, dict):
                raise ValueError(f"search_space[{name}] must be a dict")
            param_type = spec.get('type')
            if param_type in ('float', 'int'):
                bounds = spec.get('bounds')
                if not isinstance(bounds, list) or len(bounds) != 2:
                    raise ValueError(f"search_space[{name}].bounds must be a list of length 2")
                low, high = bounds[0], bounds[1]
                if low > high:
                    raise ValueError(f"search_space[{name}].bounds low must be <= high")
                scale = spec.get('scale', 'linear')
                if scale not in ('linear', 'log'):
                    raise ValueError(f"search_space[{name}].scale must be 'linear' or 'log'")
                if scale == 'log' and low <= 0:
                    raise ValueError(f"search_space[{name}] log scale requires low > 0")
                if param_type == 'int':
                    low = int(low)
                    high = int(high)
                normalized[name] = {
                    'type': param_type,
                    'scale': scale,
                    'bounds': [low, high]
                }
            elif param_type == 'categorical':
                choices = spec.get('choices')
                if not isinstance(choices, list) or len(choices) == 0:
                    raise ValueError(f"search_space[{name}].choices must be a non-empty list")
                normalized[name] = {
                    'type': 'categorical',
                    'choices': choices
                }
            else:
                raise ValueError(f"search_space[{name}].type must be float, int, or categorical")
        return normalized

    def _apply_domain_overrides(self) -> None:
        dataset_key = str(self.dataset_name).lower()
        domain_key = str(self.domain).lower()

        def lookup(mapping):
            if not isinstance(mapping, dict):
                return None
            for key in (self.dataset_name, dataset_key, self.domain, domain_key):
                if key in mapping:
                    return mapping[key]
            return None

        safe_lr = lookup(self.dataset_max_safe_learning_rates)
        if safe_lr is None:
            safe_lr = lookup(self.domain_max_safe_learning_rates)
        if safe_lr is not None:
            safe_lr = float(safe_lr)
            self.max_safe_learning_rate = (
                safe_lr if self.max_safe_learning_rate is None
                else min(float(self.max_safe_learning_rate), safe_lr)
            )
            if 'learning_rate' in self.search_space:
                spec = dict(self.search_space['learning_rate'])
                low, high = spec.get('bounds', [1e-5, safe_lr])
                spec['bounds'] = [low, min(float(high), safe_lr)]
                self.search_space['learning_rate'] = spec

        high_lr = lookup(self.dataset_risk_high_lr_thresholds)
        if high_lr is None:
            high_lr = lookup(self.domain_risk_high_lr_thresholds)
        if high_lr is not None:
            self.risk_high_lr_threshold = min(self.risk_high_lr_threshold, float(high_lr))

        fail_safe_low = lookup(self.dataset_fail_safe_low_accuracy)
        if fail_safe_low is None:
            fail_safe_low = lookup(self.domain_fail_safe_low_accuracy)
        if fail_safe_low is not None:
            self.fail_safe_low_accuracy = float(fail_safe_low)


from meta_learning import MetaLearner

class ExperimentRunner:
    """Main experiment runner for the release artifact."""

    @staticmethod
    def set_trial_seed(seed: int) -> None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def __init__(self, config: ExperimentConfig, output_file: Optional[str] = None):
        self.config = config
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if output_file:
            self.results_file = output_file
            if os.path.dirname(self.results_file):
                os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.results_file = os.path.join(
                self.config.output_dir,
                f"{self.config.experiment_name}_{timestamp}.json"
            )

        self.state_dir = self.config.state_dir or os.path.dirname(self.results_file) or self.config.output_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self.episodic_memory_file = os.path.join(self.state_dir, "episodic_memory.json")

        self.run_metadata = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "results_file": self.results_file,
            "state_dir": self.state_dir
        }
        self.run_metadata["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            config_json = json.dumps(self.config.__dict__, sort_keys=True, default=str)
            self.run_metadata["config_hash"] = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        except Exception as e:
            self.run_metadata["config_hash_error"] = str(e)
        
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize optimizers, memory, calibration, and analysis helpers."""
        self.episodic_memory = None
        self.optimizer = None
        self.meta_learner = None
        self.completed_results = []

        if 'LLM' in self.config.method:
            self.llm_client = UnifiedLLMClient(model_name=self.config.llm_model)
            
            if self.config.use_episodic_search:
                self.episodic_memory = EpisodicMemory()
                memory_file = self.episodic_memory_file
                if os.path.exists(memory_file) and not self.config.reset_episodic_memory:
                    try:
                        self.episodic_memory.load(memory_file)
                        print(f"  [Episodic Memory] Loaded {len(self.episodic_memory.episodes)} episodes from {memory_file}")
                    except Exception as e:
                        print(f"  [Episodic Memory] Failed to load memory: {e}")
            
            if self.config.use_meta_learning:
                self.meta_learner = MetaLearner()
            else:
                self.meta_learner = None
        
        elif self.config.method == 'Random':
            self.optimizer = RandomSearch(self.config.search_space)
        
        elif self.config.method == 'Bayesian':
            self.optimizer = BayesianOptimization(self.config.search_space)

        elif self.config.method == 'TPE':
            self.optimizer = TPESearch(
                self.config.search_space,
                seed=self.config.tpe_seed,
                maximize=self.config.maximize,
                n_startup_trials=self.config.tpe_n_startup_trials,
                n_ei_candidates=self.config.tpe_n_ei_candidates,
                consider_prior=self.config.tpe_consider_prior,
                prior_weight=self.config.tpe_prior_weight,
                constant_liar=self.config.tpe_constant_liar
            )

        elif self.config.method in ('TPE+MV', 'TPE_MV'):
            self.optimizer = TPEMultivariateSearch(
                self.config.search_space,
                seed=self.config.tpe_seed,
                maximize=self.config.maximize,
                n_startup_trials=self.config.tpe_n_startup_trials,
                n_ei_candidates=self.config.tpe_n_ei_candidates,
                constant_liar=self.config.tpe_mv_constant_liar,
                consider_prior=self.config.tpe_consider_prior,
                prior_weight=self.config.tpe_prior_weight
            )

        elif self.config.method in ('TPE+Pruner', 'TPE_Pruner'):
            self.optimizer = TPEPrunedSearch(
                self.config.search_space,
                seed=self.config.tpe_seed,
                maximize=self.config.maximize,
                pruner_type=self.config.tpe_pruner_type,
                n_startup_trials=self.config.tpe_pruner_n_startup_trials,
                n_warmup_steps=self.config.tpe_pruner_n_warmup_steps,
                interval_steps=self.config.tpe_pruner_interval_steps,
                n_ei_candidates=self.config.tpe_pruner_n_ei_candidates,
                multivariate=True,
                group=True,
                constant_liar=self.config.tpe_mv_constant_liar,
                consider_prior=self.config.tpe_consider_prior,
                prior_weight=self.config.tpe_prior_weight
            )
        
        elif self.config.method == 'Grid':
            pass
        
        # Variance Reduction
        if self.config.use_variance_reduction:
            self.variance_reducer = VarianceReduction(self.config.search_space)
        else:
            self.variance_reducer = None
        
        self.stat_analyzer = StatisticalSignificance()
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.calibration_state_path = os.path.join(self.state_dir, "llm_calibration_state.json")
        self.calibrator = UncertaintyCalibrator()
        if os.path.exists(self.calibration_state_path) and not self.config.reset_calibration_state:
            try:
                self.calibrator.load(self.calibration_state_path)
            except Exception as e:
                print(f"  [Calibration] Failed to load state: {e}")
    
    def run_single_trial(self, trial_id: int, hyperparameters: Dict, seed: int, hypothesis: str = "") -> Dict:
        """Evaluate one configuration and return a trial result dictionary."""
        print(f"\n[Trial {trial_id}] Seed {seed}")
        if hypothesis:
             print(f"  Hypothesis/Reasoning: {hypothesis}")
        print(f"  Hyperparameters: {hyperparameters}")
        self.set_trial_seed(seed)
        
        int_params = ['batch_size', 'epochs', 'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'warmup_steps']
        for param in int_params:
            if param in hyperparameters:
                hyperparameters[param] = int(hyperparameters[param])
        
        print(f"  Training model for {self.config.dataset_name} ({self.config.domain})...")
        
        try:
            if self.config.domain == 'CV':
                if self.config.dataset_name == 'cifar100':
                    train_dataset, val_dataset, _ = DatasetLoader.load_cifar100(split_seed=42)
                else:
                    train_dataset, val_dataset, _ = DatasetLoader.load_cifar100(split_seed=42)
                
                batch_size = int(hyperparameters.get('batch_size', 64))
                train_generator = torch.Generator().manual_seed(int(seed))
                val_generator = torch.Generator().manual_seed(int(seed))
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=train_generator)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=val_generator)
                
                model = CVModelTrainer.create_resnet50(num_classes=100)
                train_result = CVModelTrainer.train(model, train_loader, val_loader, hyperparameters)
                
            elif self.config.domain == 'NLP':
                if self.config.dataset_name == 'ag_news':
                    train_dataset, val_dataset, _ = DatasetLoader.load_ag_news()
                    num_labels = 4
                elif self.config.dataset_name == 'imdb':
                    train_dataset, val_dataset, _ = DatasetLoader.load_imdb()
                    num_labels = 2
                else:
                    train_dataset, val_dataset, _ = DatasetLoader.load_ag_news()
                    num_labels = 4
                
                train_result = NLPModelTrainer.train(
                    train_dataset, val_dataset, hyperparameters, 
                    model_name='distilbert-base-uncased',
                    num_labels=num_labels
                )
                
            elif self.config.domain == 'Tabular':
                raise ValueError("Tabular paper experiments are not included in this release artifact. Use src/custom_runner.py for custom tabular HPO recipes.")
            
            else:
                print(f"Unknown domain: {self.config.domain}. Using simulation.")
                import numpy as np
                train_result = {
                    'accuracy': np.random.uniform(0.70, 0.95),
                    'loss': np.random.uniform(0.1, 0.5),
                    'training_time': np.random.uniform(10, 30)
                }

            accuracy = train_result['accuracy']
            loss = train_result['loss']
            training_time = train_result['training_time']
            history = train_result.get('history', {})
            model_state_dict = train_result.get('model_state_dict', None)
            trial_status = 'success'
            error_message = None
            
            if model_state_dict and os.getenv("LLM_HPO_SAVE_CHECKPOINTS", "0") == "1":
                model_dir = os.path.join(os.path.dirname(self.results_file), 'models')
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"trial_{trial_id}_model.pth")
                torch.save(model_state_dict, model_path)
            
        except Exception as e:
            print(f"  Error during training: {e}")
            accuracy = 0.0
            loss = 999.0
            training_time = 0.0
            history = {}
            trial_status = 'failed'
            error_message = str(e)
        
        print(f"  Accuracy: {accuracy:.4f}")
        
        result = {
            'trial_id': trial_id,
            'seed': seed,
            'hyperparameters': hyperparameters,
            'accuracy': accuracy,
            'loss': loss,
            'training_time': training_time,
            'history': history,
            'status': trial_status,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat()
        }
        
        return result

    def apply_stability_constraints(self, params: Dict) -> Dict:
        """
        Apply simple safety constraints to avoid unstable regions.
        """
        if not isinstance(params, dict):
            return params

        constrained = dict(params)
        if 'learning_rate' in constrained:
            try:
                lr = float(constrained['learning_rate'])
                if self.config.max_safe_learning_rate is not None:
                    lr = min(lr, float(self.config.max_safe_learning_rate))
                if lr <= 0:
                    lr = 1e-5
                constrained['learning_rate'] = lr
            except Exception:
                pass

        # Avoid very small batch sizes when using high learning rates.
        if 'learning_rate' in constrained and 'batch_size' in constrained:
            try:
                lr = float(constrained['learning_rate'])
                bs = int(constrained['batch_size'])
                if lr > self.config.risk_high_lr_threshold and bs < self.config.risk_small_batch_threshold:
                    constrained['batch_size'] = self.config.risk_small_batch_threshold
            except Exception:
                pass

        return constrained

    def apply_conservative_params(self, params: Dict) -> Dict:
        """
        Conservative fallback parameters used during fail-safe cooldown.
        """
        conservative = dict(params)
        if 'learning_rate' in conservative:
            try:
                lr_max = self.config.conservative_lr_max
                if self.config.max_safe_learning_rate is not None:
                    lr_max = min(float(lr_max), float(self.config.max_safe_learning_rate))
                conservative['learning_rate'] = min(float(conservative['learning_rate']), lr_max)
            except Exception:
                pass
        if 'batch_size' in conservative:
            try:
                bs = int(conservative['batch_size'])
                conservative['batch_size'] = max(32, min(bs, 128))
            except Exception:
                pass
        return conservative
    
    def _run_experiment_legacy_presample(self):
        """Main experiment runner for the release artifact."""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.config.experiment_name}")
        print(f"  Method: {self.config.method}")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Trials: {self.config.n_trials}")
        print(f"  Seeds: {self.config.n_seeds}")
        print(f"{'='*60}\n")
        
        all_results = []
        accuracies = []
        successful_accuracies = []
        self.completed_results = []
        consecutive_low_or_failed = 0
        fail_safe_remaining = 0
        
        use_vr_sampling = (
            self.variance_reducer is not None 
            and 'LLM' not in self.config.method 
            and self.config.method not in ('Bayesian', 'TPE', 'TPE+MV', 'TPE_MV', 'TPE+Pruner', 'TPE_Pruner')
        )

        if use_vr_sampling:
            configs = self.variance_reducer.combined_sampling(
                n_samples=self.config.n_trials,
                n_strata_per_dim=3,
                antithetic_ratio=0.5
            )
            configs = self.variance_reducer.add_multiple_seeds(
                configs, n_seeds=self.config.n_seeds
            )
        else:
            configs = []
            configs = []
            
            pbar = tqdm(range(self.config.n_trials * self.config.n_seeds), desc="Generating Configs")
            
            for i in pbar:
                hypothesis = ""
                context_logs = {}
                seed = 42 + (i % self.config.n_seeds)
                seed_trial_index = (i // self.config.n_seeds) + 1
                if 'LLM' in self.config.method:
                    if i < self.config.warm_start_trials and self.variance_reducer:
                        mix_configs = self.variance_reducer.mix_sampling(
                            n_samples=1,
                            mix_ratio=self.config.mix_random_ratio
                        )
                        params = mix_configs[0].params
                        hypothesis = "Warm-start sample (mixed structured/random)."
                        context_logs = {
                            "warm_start": True,
                            "mix_random_ratio": self.config.mix_random_ratio
                        }
                    else:
                        params, hypothesis, context_logs = self.suggest_hyperparameters_llm(trial_index=i)
                elif self.config.method == 'Random':
                    params = self.optimizer.suggest(1)[0]
                elif self.config.method == 'Bayesian':
                    params = self.optimizer.suggest(1)[0]
                elif self.config.method == 'TPE':
                    params, tpe_trial_id = self.optimizer.suggest()
                    context_logs['tpe_trial_id'] = tpe_trial_id
                elif self.config.method in ('TPE+MV', 'TPE_MV'):
                    params, tpe_trial_id = self.optimizer.suggest()
                    context_logs['tpe_trial_id'] = tpe_trial_id
                elif self.config.method in ('TPE+Pruner', 'TPE_Pruner'):
                    params, tpe_trial_id = self.optimizer.suggest()
                    context_logs['tpe_trial_id'] = tpe_trial_id
                else:
                    params = {'learning_rate': 0.01, 'batch_size': 64}
                
                configs.append({
                    'params': params,
                    'seed': seed,
                    'seed_trial_index': seed_trial_index,
                    'hypothesis': hypothesis,
                    'context_logs': context_logs
                })
        
        pbar = tqdm(enumerate(configs), total=len(configs), desc="Running Trials")
        for i, config in pbar:
            if isinstance(config, dict) and 'params' in config:
                params = config['params']
                seed = config['seed']
                seed_trial_index = config.get('seed_trial_index')
                hypothesis = config.get('hypothesis', "")
                context_logs = dict(config.get('context_logs', {}))
            else:
                params = config.params
                seed = config.seed if config.seed else 42
                seed_trial_index = None
                hypothesis = ""
                context_logs = {}
            
            # Constraint-aware clipping before trial execution.
            params = self.apply_stability_constraints(params)

            # Fail-safe mode: enforce conservative region for a few trials.
            if self.config.enable_fail_safe and fail_safe_remaining > 0:
                params = self.apply_conservative_params(params)
                context_logs['fail_safe_mode'] = True
                context_logs['fail_safe_remaining_before'] = fail_safe_remaining
                fail_safe_remaining -= 1
            else:
                context_logs['fail_safe_mode'] = False

            result = self.run_single_trial(i, params, seed, hypothesis=hypothesis)
            result['hypothesis'] = hypothesis
            result['seed_trial_index'] = seed_trial_index
            result['context_logs'] = context_logs # Save logs to result
            all_results.append(result)
            self.completed_results.append(result)
            accuracies.append(result['accuracy'])
            if result.get('status') == 'success':
                successful_accuracies.append(result['accuracy'])

            if result.get('status') != 'success':
                consecutive_low_or_failed += 1
            else:
                if result['accuracy'] < self.config.fail_safe_low_accuracy:
                    consecutive_low_or_failed += 1
                else:
                    consecutive_low_or_failed = 0

            if self.config.enable_fail_safe and consecutive_low_or_failed >= self.config.fail_safe_patience:
                fail_safe_remaining = max(fail_safe_remaining, self.config.fail_safe_cooldown)
                consecutive_low_or_failed = 0
                print(f"  [Fail-Safe] Activated conservative mode for next {fail_safe_remaining} trial(s).")
            
            if self.episodic_memory and result.get('status') == 'success':
                if not hypothesis:
                    hypothesis = f"Trial {i}: Testing hyperparameters {params}"
                
                self.episodic_memory.add_episode(
                    hyperparameters=params,
                    accuracy=result['accuracy'],
                    hypothesis=hypothesis,
                    timestamp=result['timestamp']
                )
            
            if self.config.method == 'Bayesian':
                self.optimizer.update(params, result['accuracy'])
            elif self.config.method == 'TPE':
                tpe_trial_id = context_logs.get('tpe_trial_id')
                if tpe_trial_id is not None:
                    self.optimizer.update(tpe_trial_id, float(result['accuracy']))
            elif self.config.method in ('TPE+MV', 'TPE_MV'):
                tpe_trial_id = context_logs.get('tpe_trial_id')
                if tpe_trial_id is not None:
                    self.optimizer.update(tpe_trial_id, float(result['accuracy']))
            elif self.config.method in ('TPE+Pruner', 'TPE_Pruner'):
                tpe_trial_id = context_logs.get('tpe_trial_id')
                if tpe_trial_id is not None:
                    was_pruned = self.optimizer.update(
                        tpe_trial_id,
                        float(result['accuracy']),
                        step=0
                    )
                    context_logs['tpe_pruned'] = bool(was_pruned)

            # Calibration update (LLM only, successful trials only)
            if 'LLM' in self.config.method and self.calibrator and result.get('status') == 'success':
                candidates = context_logs.get('llm_candidates_valid', [])
                for cand in candidates:
                    cand_params = cand.get('params', {})
                    if self._params_match(cand_params, params):
                        mu_raw = cand.get('mu_raw')
                        sigma_raw = cand.get('sigma_raw')
                        if isinstance(mu_raw, (int, float)) and isinstance(sigma_raw, (int, float)):
                            self.calibrator.update(float(mu_raw), float(sigma_raw), result['accuracy'])
                            self.calibrator.fit()
                            try:
                                self.calibrator.save(self.calibration_state_path)
                            except Exception as e:
                                print(f"  [Calibration] Failed to save state: {e}")
                        break
        
        self.save_results(all_results, accuracies, successful_accuracies)

        if self.episodic_memory:
            memory_file = self.episodic_memory_file
            try:
                self.episodic_memory.save(memory_file)
                print(f"  [Episodic Memory] Saved {len(self.episodic_memory.episodes)} episodes to {memory_file}")
            except Exception as e:
                print(f"  [Episodic Memory] Failed to save memory: {e}")
        
        self.analyze_results(accuracies, successful_accuracies)
        
        print(f"\n{'='*60}")
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Results saved: {self.results_file}")
        print(f"{'='*60}\n")

    def run_experiment(self):
        """Run trials sequentially so adaptive optimizers observe previous outcomes."""
        print(f"\n{'='*60}")
        print(f"Experiment start: {self.config.experiment_name}")
        print(f"  Method: {self.config.method}")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Trials: {self.config.n_trials}")
        print(f"  Seeds: {self.config.n_seeds}")
        print(f"{'='*60}\n")

        all_results = []
        accuracies = []
        successful_accuracies = []
        self.completed_results = []
        consecutive_low_or_failed = 0
        fail_safe_remaining = 0

        use_vr_sampling = (
            self.variance_reducer is not None
            and 'LLM' not in self.config.method
            and self.config.method not in ('Bayesian', 'TPE', 'TPE+MV', 'TPE_MV', 'TPE+Pruner', 'TPE_Pruner')
        )
        if use_vr_sampling:
            configs = self.variance_reducer.combined_sampling(
                n_samples=self.config.n_trials,
                n_strata_per_dim=3,
                antithetic_ratio=0.5
            )
            configs = self.variance_reducer.add_multiple_seeds(
                configs, n_seeds=self.config.n_seeds
            )
            trial_iter = enumerate(configs)
            total = len(configs)
            desc = "Running Trials"
        else:
            total = self.config.n_trials * self.config.n_seeds
            trial_iter = ((i, self._suggest_trial_config(i)) for i in range(total))
            desc = "Running Adaptive Trials"

        for i, config in tqdm(trial_iter, total=total, desc=desc):
            if isinstance(config, dict) and 'params' in config:
                params = config['params']
                seed = config['seed']
                seed_trial_index = config.get('seed_trial_index')
                hypothesis = config.get('hypothesis', "")
                context_logs = dict(config.get('context_logs', {}))
            else:
                params = config.params
                seed = config.seed if config.seed else 42
                seed_trial_index = None
                hypothesis = ""
                context_logs = {}

            params = self.apply_stability_constraints(params)
            if self.config.enable_fail_safe and fail_safe_remaining > 0:
                params = self.apply_conservative_params(params)
                context_logs['fail_safe_mode'] = True
                context_logs['fail_safe_remaining_before'] = fail_safe_remaining
                fail_safe_remaining -= 1
            else:
                context_logs['fail_safe_mode'] = False

            result = self.run_single_trial(i, params, seed, hypothesis=hypothesis)
            result['hypothesis'] = hypothesis
            result['seed_trial_index'] = seed_trial_index
            result['context_logs'] = context_logs
            all_results.append(result)
            self.completed_results.append(result)
            accuracies.append(result['accuracy'])
            if result.get('status') == 'success':
                successful_accuracies.append(result['accuracy'])

            if result.get('status') != 'success' or result['accuracy'] < self.config.fail_safe_low_accuracy:
                consecutive_low_or_failed += 1
            else:
                consecutive_low_or_failed = 0
            if self.config.enable_fail_safe and consecutive_low_or_failed >= self.config.fail_safe_patience:
                fail_safe_remaining = max(fail_safe_remaining, self.config.fail_safe_cooldown)
                consecutive_low_or_failed = 0
                print(f"  [Fail-Safe] Activated conservative mode for next {fail_safe_remaining} trial(s).")

            self._update_adaptive_state(result, params, hypothesis, context_logs, i)

        self.save_results(all_results, accuracies, successful_accuracies)
        if self.episodic_memory:
            memory_file = self.episodic_memory_file
            try:
                self.episodic_memory.save(memory_file)
                print(f"  [Episodic Memory] Saved {len(self.episodic_memory.episodes)} episodes to {memory_file}")
            except Exception as e:
                print(f"  [Episodic Memory] Failed to save memory: {e}")

        self.analyze_results(accuracies, successful_accuracies)
        print(f"\n{'='*60}")
        print(f"Experiment complete: {self.config.experiment_name}")
        print(f"Results: {self.results_file}")
        print(f"{'='*60}\n")

    def _suggest_trial_config(self, i: int) -> Dict:
        hypothesis = ""
        context_logs = {}
        seed = 42 + (i % self.config.n_seeds)
        seed_trial_index = (i // self.config.n_seeds) + 1
        if 'LLM' in self.config.method:
            if i < self.config.warm_start_trials and self.variance_reducer:
                mix_configs = self.variance_reducer.mix_sampling(
                    n_samples=1,
                    mix_ratio=self.config.mix_random_ratio
                )
                params = mix_configs[0].params
                hypothesis = "Warm-start sample (mixed structured/random)."
                context_logs = {
                    "warm_start": True,
                    "mix_random_ratio": self.config.mix_random_ratio
                }
            else:
                params, hypothesis, context_logs = self.suggest_hyperparameters_llm(trial_index=i)
        elif self.config.method == 'Random':
            params = self.optimizer.suggest(1)[0]
        elif self.config.method == 'Bayesian':
            params = self.optimizer.suggest(1)[0]
        elif self.config.method == 'TPE':
            params, tpe_trial_id = self.optimizer.suggest()
            context_logs['tpe_trial_id'] = tpe_trial_id
        elif self.config.method in ('TPE+MV', 'TPE_MV'):
            params, tpe_trial_id = self.optimizer.suggest()
            context_logs['tpe_trial_id'] = tpe_trial_id
        elif self.config.method in ('TPE+Pruner', 'TPE_Pruner'):
            params, tpe_trial_id = self.optimizer.suggest()
            context_logs['tpe_trial_id'] = tpe_trial_id
        else:
            params = {'learning_rate': 0.01, 'batch_size': 64}
        return {
            'params': params,
            'seed': seed,
            'seed_trial_index': seed_trial_index,
            'hypothesis': hypothesis,
            'context_logs': context_logs
        }

    def _update_adaptive_state(self, result: Dict, params: Dict, hypothesis: str, context_logs: Dict, trial_id: int) -> None:
        if self.episodic_memory and result.get('status') == 'success':
            episode_hypothesis = hypothesis or f"Trial {trial_id}: Testing hyperparameters {params}"
            self.episodic_memory.add_episode(
                hyperparameters=params,
                accuracy=result['accuracy'],
                hypothesis=episode_hypothesis,
                timestamp=result['timestamp']
            )

        if self.config.method == 'Bayesian':
            self.optimizer.update(params, result['accuracy'])
        elif self.config.method == 'TPE':
            tpe_trial_id = context_logs.get('tpe_trial_id')
            if tpe_trial_id is not None:
                self.optimizer.update(tpe_trial_id, float(result['accuracy']))
        elif self.config.method in ('TPE+MV', 'TPE_MV'):
            tpe_trial_id = context_logs.get('tpe_trial_id')
            if tpe_trial_id is not None:
                self.optimizer.update(tpe_trial_id, float(result['accuracy']))
        elif self.config.method in ('TPE+Pruner', 'TPE_Pruner'):
            tpe_trial_id = context_logs.get('tpe_trial_id')
            if tpe_trial_id is not None:
                was_pruned = self.optimizer.update(tpe_trial_id, float(result['accuracy']), step=0)
                context_logs['tpe_pruned'] = bool(was_pruned)

        if 'LLM' in self.config.method and self.calibrator and result.get('status') == 'success':
            candidates = context_logs.get('llm_candidates_valid', [])
            for cand in candidates:
                cand_params = cand.get('params', {})
                if self._params_match(cand_params, params):
                    mu_raw = cand.get('mu_raw')
                    sigma_raw = cand.get('sigma_raw')
                    if isinstance(mu_raw, (int, float)) and isinstance(sigma_raw, (int, float)):
                        self.calibrator.update(float(mu_raw), float(sigma_raw), result['accuracy'])
                        self.calibrator.fit()
                        try:
                            self.calibrator.save(self.calibration_state_path)
                        except Exception as e:
                            print(f"  [Calibration] Failed to save state: {e}")
                    break

    @staticmethod
    def _format_search_space_table(search_space: Dict) -> str:
        lines = [
            "| name | type | scale | bounds | choices |",
            "|---|---|---|---|---|"
        ]
        for name, spec in search_space.items():
            param_type = spec.get('type')
            scale = spec.get('scale', '') if param_type in ('float', 'int') else ''
            bounds = spec.get('bounds', '') if param_type in ('float', 'int') else ''
            if isinstance(bounds, list) and len(bounds) == 2:
                bounds_str = f"[{bounds[0]}, {bounds[1]}]"
            else:
                bounds_str = ''
            choices = spec.get('choices', '') if param_type == 'categorical' else ''
            if isinstance(choices, list):
                choices_str = ', '.join([str(c) for c in choices])
            else:
                choices_str = ''
            lines.append(
                f"| {name} | {param_type} | {scale} | {bounds_str} | {choices_str} |"
            )
        return "\n".join(lines)

    def _format_stability_guidance(self) -> str:
        lines = [
            f"- learning_rate should stay <= {self.config.max_safe_learning_rate if self.config.max_safe_learning_rate is not None else self.config.risk_high_lr_threshold}",
            f"- avoid batch_size < {self.config.risk_small_batch_threshold} when learning_rate is high",
            f"- prefer conservative regularization when dropout_rate > {self.config.risk_high_dropout_threshold}",
            f"- avoid simultaneously large mixup_alpha and cutmix_alpha near {self.config.risk_high_mix_threshold}/{self.config.risk_high_cutmix_threshold}",
        ]
        return "\n".join(lines)

    def serialize_prompt_context(self, D_t: Dict, c: Dict) -> str:
        sections = []
        sections.append("## Task")
        sections.append(f"dataset: {self.config.dataset_name}")
        sections.append(f"domain: {self.config.domain}")
        sections.append(f"objective_metric: {self.config.objective_metric}")
        sections.append(f"maximize: {self.config.maximize}")
        sections.append(f"noise_model: {self.config.noise_model}")
        sections.append(f"budget_T: {self.config.budget_T}")
        sections.append("")
        sections.append("## Search Space")
        sections.append(self._format_search_space_table(self.config.search_space))
        sections.append("")
        sections.append("## Stability Guidance")
        sections.append(self._format_stability_guidance())
        sections.append("")
        sections.append("## Episodic Memory")
        sections.append(D_t.get('episodic_summary', 'No episodic data.'))
        sections.append("")
        sections.append("## Meta-Learning")
        sections.append(c.get('meta_summary', 'No meta data.'))
        sections.append("")
        sections.append("## Output Format")
        sections.append(f"schema_version: {self.config.llm_output_schema_version}")
        sections.append(
            f"Return ONLY JSON with exactly {self.config.llm_num_candidates} candidates."
        )
        sections.append(
            "Schema: {\"schema_version\":\"v1\",\"candidates\":[{\"params\":{...},\"mu\":0.0,\"sigma\":0.0,\"reason\":\"...\"}]}"
        )
        return "\n".join(sections)

    def _build_llm_response_format(self) -> Dict:
        params_props = {}
        required_params = []
        for name, spec in self.config.search_space.items():
            param_type = spec.get('type')
            if param_type == 'int':
                param_schema = {"type": "integer"}
                bounds = spec.get('bounds')
                if isinstance(bounds, list) and len(bounds) == 2:
                    param_schema["minimum"] = int(bounds[0])
                    param_schema["maximum"] = int(bounds[1])
            elif param_type == 'float':
                param_schema = {"type": "number"}
                bounds = spec.get('bounds')
                if isinstance(bounds, list) and len(bounds) == 2:
                    param_schema["minimum"] = float(bounds[0])
                    param_schema["maximum"] = float(bounds[1])
            elif param_type == 'categorical':
                choices = spec.get('choices', [])
                if choices:
                    param_schema = {"enum": choices}
                else:
                    param_schema = {"type": ["string", "number", "integer", "boolean"]}
            else:
                param_schema = {"type": ["string", "number", "integer", "boolean"]}

            params_props[name] = param_schema
            required_params.append(name)

        candidate_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "params": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": params_props,
                    "required": required_params
                },
                "mu": {"type": "number"},
                "sigma": {"type": "number"},
                "reason": {"type": "string"}
            },
            "required": ["params", "mu", "sigma", "reason"]
        }

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {"type": "string"},
                "candidates": {
                    "type": "array",
                    "minItems": int(self.config.llm_num_candidates),
                    "maxItems": int(self.config.llm_num_candidates),
                    "items": candidate_schema
                }
            },
            "required": ["schema_version", "candidates"]
        }

        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"hpo_candidates_{self.config.llm_output_schema_version}",
                "schema": schema,
                "strict": True
            }
        }

    def _parse_llm_candidates_response(self, response: str) -> List[Dict]:
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return []
            payload = json.loads(json_match.group())
        except Exception:
            return []

        if isinstance(payload, dict) and isinstance(payload.get('candidates'), list):
            return payload.get('candidates', [])
        if isinstance(payload, dict):
            return [{"params": payload, "mu": None, "sigma": None, "reason": ""}]
        return []

    def validate_and_clip_params(self, params: Dict) -> Optional[Dict]:
        if not isinstance(params, dict):
            return None
        cleaned = {}
        for name, spec in self.config.search_space.items():
            if name not in params:
                return None
            value = params[name]
            param_type = spec.get('type')
            if param_type == 'categorical':
                choices = spec.get('choices', [])
                if value not in choices:
                    return None
                cleaned[name] = value
                continue

            if not isinstance(value, (int, float)):
                return None
            bounds = spec.get('bounds', [None, None])
            low, high = bounds[0], bounds[1]
            if param_type == 'int':
                value = int(round(value))
                if low is not None:
                    value = max(value, int(low))
                if high is not None:
                    value = min(value, int(high))
            else:
                value = float(value)
                if low is not None:
                    value = max(value, float(low))
                if high is not None:
                    value = min(value, float(high))
            cleaned[name] = value

        return cleaned

    @staticmethod
    def _params_match(a: Dict, b: Dict, tol: float = 1e-8) -> bool:
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        if set(a.keys()) != set(b.keys()):
            return False
        for key in a.keys():
            v1 = a[key]
            v2 = b[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if abs(float(v1) - float(v2)) > tol:
                    return False
            else:
                if v1 != v2:
                    return False
        return True

    def _get_kappa(self, trial_index: int) -> float:
        schedule = self.config.exploration_schedule
        if not isinstance(schedule, dict):
            return float(self.config.kappa)
        schedule_type = schedule.get('type', 'constant')
        if schedule_type == 'linear_decay':
            start = float(schedule.get('start', self.config.kappa))
            end = float(schedule.get('end', self.config.kappa))
            total = int(schedule.get('total_trials', self.config.n_trials))
            if total <= 1:
                return end
            alpha = min(max(trial_index / (total - 1), 0.0), 1.0)
            return start + (end - start) * alpha
        return float(self.config.kappa)

    def _get_temperature(self, trial_index: int) -> float:
        schedule = self.config.exploration_schedule
        base_temp = float(self.config.llm_temperature)
        if not isinstance(schedule, dict):
            return base_temp
        schedule_type = schedule.get('type', 'constant')
        if schedule_type == 'linear_decay':
            start = float(schedule.get('temp_start', base_temp))
            end = float(schedule.get('temp_end', base_temp))
            total = int(schedule.get('total_trials', self.config.n_trials))
            if total <= 1:
                return end
            alpha = min(max(trial_index / (total - 1), 0.0), 1.0)
            return start + (end - start) * alpha
        return base_temp

    def _get_best_so_far(self) -> Optional[float]:
        successful = [
            float(r.get('accuracy'))
            for r in getattr(self, 'completed_results', [])
            if r.get('status') == 'success' and isinstance(r.get('accuracy'), (int, float))
        ]
        if successful:
            return max(successful) if self.config.maximize else min(successful)
        if self.episodic_memory and self.episodic_memory.episodes:
            accuracies = [ep.accuracy for ep in self.episodic_memory.episodes]
            if accuracies:
                return max(accuracies) if self.config.maximize else min(accuracies)
        return None

    def _get_calibration_report(self) -> Dict:
        if not self.calibrator:
            return {}
        report = self.calibrator.report()
        return report if isinstance(report, dict) else {}

    def _get_episodic_gate_status(self) -> Dict:
        status = {
            'enabled_by_config': bool(self.config.use_episodic_search),
            'confidence_gated': bool(self.config.enable_confidence_gated_episodic),
            'use_memory': False,
            'memory_mode': 'off',
            'gate_level': 'closed',
            'gate_strength': 0.0,
            'reason': 'disabled',
            'num_episodes': 0,
            'calibration_report': self._get_calibration_report()
        }

        if not self.config.use_episodic_search or not self.episodic_memory:
            status['reason'] = 'episodic_disabled'
            return status

        num_episodes = len(self.episodic_memory.episodes)
        status['num_episodes'] = num_episodes
        if not self.config.enable_confidence_gated_episodic:
            status['use_memory'] = True
            status['memory_mode'] = 'full'
            status['gate_level'] = 'open'
            status['gate_strength'] = 1.0
            status['reason'] = 'confidence_gate_disabled'
            return status

        if num_episodes < self.config.episodic_gate_min_episodes:
            status['reason'] = 'insufficient_episodes'
            return status

        report = status['calibration_report']
        n_records = int(report.get('n', 0) or 0)
        if n_records < self.config.episodic_gate_min_calibration_records:
            status['reason'] = 'insufficient_calibration_records'
            return status

        def open_full():
            status['use_memory'] = True
            status['memory_mode'] = 'full'
            status['gate_level'] = 'open'
            status['gate_strength'] = 1.0

        def open_partial(reason: str):
            status['use_memory'] = True
            status['memory_mode'] = 'lite'
            status['gate_level'] = 'partial'
            status['gate_strength'] = 0.5
            status['reason'] = reason

        coverage = report.get('coverage_1sigma')
        if coverage is None:
            if self.config.episodic_gate_mode == 'three_state':
                open_partial('missing_coverage')
                return status
            status['reason'] = 'missing_coverage'
            return status
        if coverage < self.config.episodic_gate_min_coverage or coverage > self.config.episodic_gate_max_coverage:
            if self.config.episodic_gate_mode == 'three_state':
                open_partial('coverage_out_of_range')
                return status
            status['reason'] = 'coverage_out_of_range'
            return status

        calibration_error = report.get('calibration_error')
        if calibration_error is not None and calibration_error > self.config.episodic_gate_max_calibration_error:
            if self.config.episodic_gate_mode == 'three_state':
                open_partial('calibration_error_too_high')
                return status
            status['reason'] = 'calibration_error_too_high'
            return status

        nll = report.get('nll')
        if nll is not None and nll > self.config.episodic_gate_max_nll:
            if self.config.episodic_gate_mode == 'three_state':
                open_partial('nll_too_high')
                return status
            status['reason'] = 'nll_too_high'
            return status

        open_full()
        status['reason'] = 'pass'
        return status

    def _estimate_param_risk(self, params: Dict) -> Tuple[float, List[str]]:
        components = self._estimate_param_risk_components(params)
        total_risk = float(components.get('hard_risk', 0.0)) + float(components.get('soft_risk', 0.0))
        flags = list(components.get('hard_flags', [])) + list(components.get('soft_flags', []))
        return total_risk, flags

    def _estimate_param_risk_components(self, params: Dict) -> Dict:
        if not isinstance(params, dict):
            return {
                'hard_risk': 1.0,
                'soft_risk': 0.0,
                'hard_flags': ['invalid_params'],
                'soft_flags': []
            }

        hard_risk = 0.0
        soft_risk = 0.0
        hard_flags = []
        soft_flags = []
        lr = params.get('learning_rate')
        batch_size = params.get('batch_size')
        dropout = params.get('dropout_rate')
        mixup = params.get('mixup_alpha')
        cutmix = params.get('cutmix_alpha')
        warmup = params.get('warmup_steps')

        if isinstance(lr, (int, float)):
            safe_lr = self.config.max_safe_learning_rate
            high_lr_threshold = safe_lr if safe_lr is not None else self.config.risk_high_lr_threshold
            if float(lr) > float(high_lr_threshold):
                overflow = float(lr) / max(float(high_lr_threshold), 1e-12)
                hard_risk += min(max(overflow - 1.0, 0.0), 3.0)
                hard_flags.append('high_learning_rate')

        if isinstance(lr, (int, float)) and isinstance(batch_size, (int, float)):
            if float(lr) > self.config.risk_high_lr_threshold and int(batch_size) < self.config.risk_small_batch_threshold:
                hard_risk += 1.0
                hard_flags.append('high_lr_small_batch')

        if isinstance(dropout, (int, float)) and float(dropout) > self.config.risk_high_dropout_threshold:
            soft_risk += max(float(dropout) - self.config.risk_high_dropout_threshold, 0.0) * 2.0
            soft_flags.append('high_dropout')

        if isinstance(mixup, (int, float)) and float(mixup) > self.config.risk_high_mix_threshold:
            soft_risk += max(float(mixup) - self.config.risk_high_mix_threshold, 0.0) * 2.0
            soft_flags.append('high_mixup')

        if isinstance(cutmix, (int, float)) and float(cutmix) > self.config.risk_high_cutmix_threshold:
            soft_risk += max(float(cutmix) - self.config.risk_high_cutmix_threshold, 0.0) * 2.0
            soft_flags.append('high_cutmix')

        if isinstance(warmup, (int, float)) and float(warmup) > self.config.risk_high_warmup_threshold:
            soft_risk += (float(warmup) - self.config.risk_high_warmup_threshold) / 500.0
            soft_flags.append('high_warmup')

        if isinstance(dropout, (int, float)) and isinstance(mixup, (int, float)) and isinstance(cutmix, (int, float)):
            regularization_sum = float(dropout) + float(mixup) + float(cutmix)
            if regularization_sum > 1.5:
                soft_risk += (regularization_sum - 1.5)
                soft_flags.append('over_regularized_combo')

        return {
            'hard_risk': hard_risk,
            'soft_risk': soft_risk,
            'hard_flags': hard_flags,
            'soft_flags': soft_flags
        }

    def _select_candidate(
        self,
        candidates: List[Dict],
        best_so_far: Optional[float],
        trial_index: int,
        gate_status: Optional[Dict] = None,
        pre_scored: Optional[List[Dict]] = None
    ) -> Optional[Dict]:
        scored = pre_scored if pre_scored is not None else self._score_candidates(candidates, best_so_far, trial_index, gate_status)
        if not scored:
            return None
        ranked = sorted(scored, key=lambda c: float(c.get('acq_score', float('-inf'))), reverse=True)
        if self.config.reliability_control_mode != 'confidence_coupled' or not self.config.enable_near_optimal_rerank:
            return ranked[0]

        best_score = float(ranked[0].get('acq_score', float('-inf')))
        top_k = max(int(self.config.near_optimal_rerank_top_k), 1)
        epsilon = max(float(self.config.near_optimal_rerank_epsilon), 0.0)
        near_optimal = [
            cand for cand in ranked[:top_k]
            if float(cand.get('acq_score', float('-inf'))) >= best_score - epsilon
        ]
        if not near_optimal:
            return ranked[0]

        rerank_key = self.config.near_optimal_rerank_key
        if rerank_key not in ('soft_risk', 'hard_risk', 'total_risk'):
            rerank_key = 'soft_risk'
        selected = min(
            near_optimal,
            key=lambda c: (
                float(c.get(rerank_key, 0.0)),
                -float(c.get('acq_score', float('-inf')))
            )
        )
        selected_with_metadata = dict(selected)
        selected_with_metadata['selection_policy'] = 'near_optimal_rerank'
        selected_with_metadata['rerank_pool_size'] = len(near_optimal)
        selected_with_metadata['rerank_key'] = rerank_key
        selected_with_metadata['rerank_best_score'] = best_score
        selected_with_metadata['rerank_epsilon'] = epsilon
        return selected_with_metadata

    def _score_candidates(
        self,
        candidates: List[Dict],
        best_so_far: Optional[float],
        trial_index: int,
        gate_status: Optional[Dict] = None
    ) -> List[Dict]:
        if not candidates:
            return []
        acq_type = (self.config.acq_type or "ucb").lower()
        kappa = self._get_kappa(trial_index)
        gate_status = gate_status or self._get_episodic_gate_status()
        gate_strength = float(gate_status.get('gate_strength', 1.0 if gate_status.get('use_memory') else 0.0))
        scored = []
        for cand in candidates:
            mu = cand.get('mu')
            sigma = cand.get('sigma')
            if not isinstance(mu, (int, float)) or not isinstance(sigma, (int, float)):
                continue
            if acq_type == 'ei':
                score = llm_ei_score(float(mu), float(sigma), best_so_far, self.config.maximize)
            else:
                score = llm_ucb_score(float(mu), float(sigma), kappa)
            risk_components = self._estimate_param_risk_components(cand.get('params', {}))
            hard_risk = float(risk_components.get('hard_risk', 0.0))
            soft_risk = float(risk_components.get('soft_risk', 0.0))
            hard_flags = list(risk_components.get('hard_flags', []))
            soft_flags = list(risk_components.get('soft_flags', []))
            param_risk_score = hard_risk + soft_risk
            risk_flags = hard_flags + soft_flags
            sigma_penalty = 0.0
            param_risk_penalty = 0.0
            hard_risk_penalty = 0.0
            soft_risk_penalty = 0.0
            risk_adjusted_score = score
            if self.config.enable_risk_aware_selection:
                if self.config.reliability_control_mode == 'confidence_coupled':
                    coupling = (1.0 - gate_strength) if self.config.gate_coupled_soft_risk else 1.0
                    sigma_mode = self.config.risk_sigma_penalty_mode
                    if sigma_mode == 'none':
                        sigma_term = 0.0
                    elif sigma_mode == 'excess_only':
                        sigma_term = max(float(sigma) - self.config.risk_sigma_free_band, 0.0)
                    else:
                        sigma_term = float(sigma)
                    sigma_penalty = self.config.risk_penalty_lambda * sigma_term * coupling
                    hard_risk_penalty = self.config.hard_risk_weight * hard_risk
                    soft_risk_penalty = self.config.soft_risk_weight * soft_risk * coupling
                    param_risk_penalty = hard_risk_penalty + soft_risk_penalty
                else:
                    sigma_penalty = self.config.risk_penalty_lambda * float(sigma)
                    param_risk_penalty = self.config.param_risk_penalty_weight * float(param_risk_score)
                risk_adjusted_score = score - sigma_penalty - param_risk_penalty
            cand_with_score = dict(cand)
            cand_with_score['acq_score_base'] = score
            cand_with_score['acq_score'] = risk_adjusted_score
            cand_with_score['acq_type'] = acq_type
            cand_with_score['kappa'] = kappa
            cand_with_score['gate_strength'] = gate_strength
            cand_with_score['sigma_penalty'] = sigma_penalty
            cand_with_score['hard_risk'] = hard_risk
            cand_with_score['soft_risk'] = soft_risk
            cand_with_score['total_risk'] = param_risk_score
            cand_with_score['hard_risk_penalty'] = hard_risk_penalty
            cand_with_score['soft_risk_penalty'] = soft_risk_penalty
            cand_with_score['param_risk_score'] = param_risk_score
            cand_with_score['param_risk_penalty'] = param_risk_penalty
            cand_with_score['risk_penalty_total'] = sigma_penalty + param_risk_penalty
            cand_with_score['hard_risk_flags'] = hard_flags
            cand_with_score['soft_risk_flags'] = soft_flags
            cand_with_score['risk_flags'] = risk_flags
            scored.append(cand_with_score)
        return scored

    def suggest_hyperparameters_llm(self, trial_index: int = 0) -> Tuple[Dict, str, Dict]:
        """Suggest hyperparameters using a standardized prompt context."""
        print("\n  [Context Analysis]")

        memory_context = ""
        top_episodes = []
        worst_episodes = []
        diverse_episodes = []
        episodic_gate = self._get_episodic_gate_status()
        memory_mode = episodic_gate.get('memory_mode', 'off')
        if episodic_gate.get('use_memory'):
            retrieval_k = self.config.k_similar_episodes
            if memory_mode == 'lite':
                retrieval_k = max(1, int(np.ceil(self.config.k_similar_episodes * self.config.episodic_gate_partial_memory_ratio)))
            top_episodes = self.episodic_memory.retrieve_top_performing_episodes(k=retrieval_k)
            if memory_mode == 'full' or self.config.episodic_gate_partial_use_worst_diverse:
                worst_episodes = self.episodic_memory.retrieve_worst_episodes(k=retrieval_k)
                diverse_episodes = self.episodic_memory.retrieve_diverse_episodes(k=retrieval_k)
            memory_context = self.episodic_memory.format_episodes_for_prompt(
                top_episodes, worst_episodes, diverse_episodes
            )
        else:
            print(f"  [Episodic Memory] Gated off: {episodic_gate.get('reason')}")
            memory_context = f"Episodic memory disabled for this trial. reason={episodic_gate.get('reason')}"

        meta_context = ""
        similar_tasks = []
        if self.meta_learner:
            print(f"  [Meta-Learning] Searching for similar tasks to '{self.config.dataset_name}'...")
            similar_tasks = self.meta_learner.retrieve_similar_tasks(self.config.dataset_name, k=2)
            if similar_tasks:
                print(f"  [Meta-Learning] Found {len(similar_tasks)} similar tasks.")
                meta_context = self.meta_learner.format_for_prompt(similar_tasks)
            else:
                print("  [Meta-Learning] No similar tasks found.")
        else:
            print("  [Meta-Learning] Disabled or Not initialized.")

        D_t = {
            'episodic_summary': memory_context
        }
        c = {
            'meta_summary': meta_context if meta_context else 'No meta data.'
        }
        prompt = self.serialize_prompt_context(D_t, c)
        system_prompt = (
            "You are an expert hyperparameter optimizer."
            " Return only JSON that matches the schema."
        )
        response_format = self._build_llm_response_format()
        temperature = self._get_temperature(trial_index)
        response = self.llm_client.generate(
            system_prompt,
            prompt,
            temperature=temperature,
            response_format=response_format,
            seed=self.config.llm_seed
        )

        candidates_raw = self._parse_llm_candidates_response(response)
        valid_candidates = []
        for cand in candidates_raw:
            if not isinstance(cand, dict):
                continue
            params = cand.get('params')
            mu = cand.get('mu')
            sigma = cand.get('sigma')
            reason = cand.get('reason', '')
            if not isinstance(mu, (int, float)) or not isinstance(sigma, (int, float)):
                continue
            cleaned = self.validate_and_clip_params(params)
            if cleaned is None:
                continue
            cleaned = self.apply_stability_constraints(cleaned)
            mu_raw = float(mu)
            sigma_raw = float(sigma)
            mu_cal, sigma_cal = self.calibrator.calibrate(mu_raw, sigma_raw)
            valid_candidates.append({
                'params': cleaned,
                'mu': float(mu_cal),
                'sigma': float(sigma_cal),
                'mu_raw': mu_raw,
                'sigma_raw': sigma_raw,
                'reason': reason
            })

        best_so_far = self._get_best_so_far()
        scored_candidates = self._score_candidates(valid_candidates, best_so_far, trial_index, episodic_gate)
        selected_candidate = self._select_candidate(
            valid_candidates,
            best_so_far,
            trial_index,
            gate_status=episodic_gate,
            pre_scored=scored_candidates
        )
        if selected_candidate:
            params = selected_candidate['params']
            reasoning = selected_candidate.get('reason', '')
        else:
            params = {'learning_rate': 0.01, 'batch_size': 64, 'weight_decay': 1e-4}
            reasoning = 'Failed to select a valid candidate from response.'

        selection_criteria = {
            'acq_type': self.config.acq_type,
            'kappa': self._get_kappa(trial_index),
            'temperature': temperature,
            'best_so_far': best_so_far,
            'calibration_enabled': True,
            'calibration_report': self._get_calibration_report(),
            'episodic_gate': episodic_gate,
            'reliability_control_mode': self.config.reliability_control_mode,
            'risk_aware_selection': self.config.enable_risk_aware_selection,
        }
        context_logs = {
            'episodic_memory_top': [ep.to_dict() for ep in top_episodes] if top_episodes else [],
            'episodic_memory_worst': [ep.to_dict() for ep in worst_episodes] if worst_episodes else [],
            'episodic_memory_diverse': [ep.to_dict() for ep in diverse_episodes] if diverse_episodes else [],
            'episodic_gate': episodic_gate,
            'meta_learning': [task.dataset_name for task, _ in similar_tasks] if similar_tasks else [],
            'serialized_prompt': prompt,
            'llm_response_raw': response,
            'llm_candidates_raw': candidates_raw,
            'llm_candidates_valid': valid_candidates,
            'llm_selected_candidate': selected_candidate,
            'llm_candidate_scores': scored_candidates,
            'llm_selection_criteria': selection_criteria,
            'llm_best_so_far': best_so_far,
            'llm_acq_type': self.config.acq_type,
            'llm_kappa': self._get_kappa(trial_index),
            'llm_schema_version': self.config.llm_output_schema_version,
            'llm_num_candidates': self.config.llm_num_candidates,
            'llm_temperature': temperature,
            'llm_temperature_base': self.config.llm_temperature,
            'llm_exploration_schedule': self.config.exploration_schedule,
            'llm_seed': self.config.llm_seed,
            'run_metadata': self.run_metadata
        }

        return params, reasoning, context_logs

    @staticmethod
    def _mean_std(values: List[float]) -> Dict:
        if not values:
            return {'mean': None, 'std': None}
        mean_value = sum(values) / len(values)
        variance = sum((x - mean_value) ** 2 for x in values) / len(values)
        return {'mean': mean_value, 'std': variance ** 0.5}

    def _group_results_by_seed(self, all_results: List[Dict]) -> Dict[int, List[Dict]]:
        grouped = {}
        for row in all_results:
            seed = int(row.get('seed', 0))
            grouped.setdefault(seed, []).append(row)
        for seed, rows in grouped.items():
            rows.sort(key=lambda r: (
                int(r.get('seed_trial_index', 10 ** 9)) if r.get('seed_trial_index') is not None else 10 ** 9,
                int(r.get('trial_id', 10 ** 9))
            ))
        return grouped

    def _compute_best_so_far_curve(self, rows: List[Dict]) -> List[float]:
        curve = []
        current_best = None
        for row in rows:
            if row.get('status') == 'success' and isinstance(row.get('accuracy'), (int, float)):
                acc = float(row['accuracy'])
                if current_best is None:
                    current_best = acc
                else:
                    current_best = max(current_best, acc) if self.config.maximize else min(current_best, acc)
            curve.append(current_best if current_best is not None else 0.0)
        return curve

    @staticmethod
    def _first_reach_trial(curve: List[float], threshold: float, maximize: bool) -> Optional[int]:
        for idx, value in enumerate(curve, start=1):
            if maximize and value >= threshold:
                return idx
            if (not maximize) and value <= threshold:
                return idx
        return None

    def _build_seedwise_summary(self, all_results: List[Dict]) -> Dict:
        grouped = self._group_results_by_seed(all_results)
        best_at = {k: [] for k in self.config.report_best_at if k > 0}
        threshold_times = {f"{t:.2f}": [] for t in self.config.report_accuracy_thresholds}
        auc_values = []
        final_best_values = []
        seed_summaries = {}

        for seed, rows in grouped.items():
            curve = self._compute_best_so_far_curve(rows)
            auc = (sum(curve) / len(curve)) if curve else None
            final_best = curve[-1] if curve else None
            accuracies = [
                float(r['accuracy']) for r in rows
                if r.get('status') == 'success' and isinstance(r.get('accuracy'), (int, float))
            ]
            per_seed_best_at = {}
            per_seed_thresholds = {}
            for k in best_at.keys():
                capped_index = min(k, len(curve)) - 1
                per_seed_best_at[str(k)] = curve[capped_index] if capped_index >= 0 else None
                if capped_index >= 0:
                    best_at[k].append(curve[capped_index])

            for threshold in self.config.report_accuracy_thresholds:
                threshold_key = f"{threshold:.2f}"
                trial_hit = self._first_reach_trial(curve, threshold, self.config.maximize)
                per_seed_thresholds[threshold_key] = trial_hit
                if trial_hit is not None:
                    threshold_times[threshold_key].append(float(trial_hit))

            if auc is not None:
                auc_values.append(float(auc))
            if final_best is not None:
                final_best_values.append(float(final_best))

            seed_summaries[str(seed)] = {
                'n_trials': len(rows),
                'n_successful_trials': len(accuracies),
                'mean_accuracy': (sum(accuracies) / len(accuracies)) if accuracies else None,
                'best_accuracy': max(accuracies) if accuracies else None,
                'auc_best_so_far': auc,
                'final_best_so_far': final_best,
                'best_at': per_seed_best_at,
                'time_to_threshold': per_seed_thresholds,
            }

        best_at_summary = {
            str(k): {
                **self._mean_std(best_at[k]),
                'per_seed_values': [float(v) for v in best_at[k]]
            }
            for k in best_at.keys()
        }
        threshold_summary = {}
        total_seeds = max(len(grouped), 1)
        for key, values in threshold_times.items():
            stats = self._mean_std(values)
            threshold_summary[key] = {
                **stats,
                'success_rate': len(values) / total_seeds,
                'per_seed_values': [float(v) for v in values]
            }

        return {
            'seed_summaries': seed_summaries,
            'auc_best_so_far': self._mean_std(auc_values),
            'final_best_so_far': self._mean_std(final_best_values),
            'best_at': best_at_summary,
            'time_to_threshold': threshold_summary,
        }

    def _build_convergence_artifacts(self, all_results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        grouped = self._group_results_by_seed(all_results)
        by_seed_rows = []
        summary_rows = []
        max_len = max((len(rows) for rows in grouped.values()), default=0)

        per_seed_curves = {}
        for seed, rows in grouped.items():
            curve = self._compute_best_so_far_curve(rows)
            per_seed_curves[seed] = curve
            for idx, row in enumerate(rows, start=1):
                by_seed_rows.append({
                    'seed': seed,
                    'trial': idx,
                    'trial_id': row.get('trial_id'),
                    'accuracy': row.get('accuracy'),
                    'best_so_far': curve[idx - 1],
                    'status': row.get('status'),
                })

        for trial_idx in range(1, max_len + 1):
            acc_values = []
            best_values = []
            for seed, rows in grouped.items():
                if len(rows) >= trial_idx:
                    row = rows[trial_idx - 1]
                    if isinstance(row.get('accuracy'), (int, float)):
                        acc_values.append(float(row['accuracy']))
                    best_values.append(float(per_seed_curves[seed][trial_idx - 1]))
            summary_rows.append({
                'trial': trial_idx,
                'mean_accuracy': self._mean_std(acc_values)['mean'],
                'std_accuracy': self._mean_std(acc_values)['std'],
                'mean_best_so_far': self._mean_std(best_values)['mean'],
                'std_best_so_far': self._mean_std(best_values)['std'],
                'n_seeds': len(best_values),
            })

        return by_seed_rows, summary_rows
    
    def save_results(self, all_results: List[Dict], accuracies: List[float], successful_accuracies: List[float]):
        """Main experiment runner for the release artifact."""
        n_total = len(all_results)
        n_success = len(successful_accuracies)
        n_failed = n_total - n_success
        failure_rate = (n_failed / n_total) if n_total > 0 else 0.0

        if successful_accuracies:
            mean_success = sum(successful_accuracies) / len(successful_accuracies)
            std_success = (
                sum((x - mean_success) ** 2 for x in successful_accuracies) / len(successful_accuracies)
            ) ** 0.5
            best_success = max(successful_accuracies)
        else:
            mean_success = 0.0
            std_success = 0.0
            best_success = 0.0

        seedwise_summary = self._build_seedwise_summary(all_results)

        summary = {
            'experiment_name': self.config.experiment_name,
            'config': self.config.__dict__,
            'run_metadata': self.run_metadata,
            'summary_statistics': {
                'best_accuracy': best_success,
                'mean_accuracy': mean_success,
                'std_accuracy': std_success,
                'n_trials': n_total,
                'n_successful_trials': n_success,
                'n_failed_trials': n_failed,
                'failure_rate': failure_rate,
                'auc_best_so_far': seedwise_summary['auc_best_so_far'],
                'final_best_so_far': seedwise_summary['final_best_so_far'],
                'best_at': seedwise_summary['best_at'],
                'time_to_threshold': seedwise_summary['time_to_threshold'],
            },
            'seed_summaries': seedwise_summary['seed_summaries'],
            'all_results': all_results
        }
        

        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        # [SCI Enhancement] CSV Export for Plotting
        try:
            import pandas as pd
            df = pd.DataFrame(all_results)
            # Flatten hyperparameters for easier analysis
            if not df.empty and 'hyperparameters' in df.columns:
                params_df = pd.json_normalize(df['hyperparameters'])
                df = df.drop('hyperparameters', axis=1)
                df = df.join(params_df, rsuffix='_param')
            
            csv_path = self.results_file.replace('.json', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"  [Artifact] Saved CSV results to {csv_path}")
            
            convergence_by_seed, convergence_summary = self._build_convergence_artifacts(all_results)

            convergence_path = self.results_file.replace('.json', '_convergence.csv')
            pd.DataFrame(convergence_summary).to_csv(convergence_path, index=False)
            print(f"  [Artifact] Saved Convergence summary to {convergence_path}")

            convergence_by_seed_path = self.results_file.replace('.json', '_convergence_by_seed.csv')
            pd.DataFrame(convergence_by_seed).to_csv(convergence_by_seed_path, index=False)
            print(f"  [Artifact] Saved per-seed convergence to {convergence_by_seed_path}")

        except Exception as e:
            print(f"  [Warning] Failed to export CSV/Convergence data: {e}")
    
    def analyze_results(self, accuracies: List[float], successful_accuracies: List[float]):
        """Main experiment runner for the release artifact."""
        print("\n=== Experiment Results Summary ===")
        if not accuracies:
            print("No trials executed.")
            return

        n_total = len(accuracies)
        n_success = len(successful_accuracies)
        n_failed = n_total - n_success
        failure_rate = n_failed / n_total

        if successful_accuracies:
            mean_success = sum(successful_accuracies) / len(successful_accuracies)
            std_success = (
                sum((x - mean_success) ** 2 for x in successful_accuracies) / len(successful_accuracies)
            ) ** 0.5
            best_success = max(successful_accuracies)
        else:
            mean_success = 0.0
            std_success = 0.0
            best_success = 0.0

        print(f"Best Accuracy (success only): {best_success:.4f}")
        print(f"Mean Accuracy (success only): {mean_success:.4f}")
        print(f"Std Dev (success only): {std_success:.4f}")
        print(f"Failure Rate: {failure_rate:.2%} ({n_failed}/{n_total})")
        
        if successful_accuracies:
            self.convergence_analyzer.add_experiment_results(self.config.method, successful_accuracies)
            convergence_rate = self.convergence_analyzer.compute_convergence_rate(self.config.method)
            print(f"Convergence Rate: {convergence_rate} iterations to 95% of best")


def main():
    """Main experiment runner for the release artifact."""
    parser = argparse.ArgumentParser(description='Run HPO Experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--method', type=str, help='Override method (e.g., LLM+Episodic)')
    parser.add_argument('--output', type=str, help='Override output file path')
    
    parser.add_argument('--use-episodic', type=str, help='Enable/disable episodic search (true/false)')
    parser.add_argument('--use-meta', type=str, help='Enable/disable meta learning (true/false)')
    parser.add_argument('--use-variance', type=str, help='Enable/disable variance reduction (true/false)')
    parser.add_argument('--confidence-gated-episodic', type=str, help='Enable/disable confidence-gated episodic usage (true/false)')
    parser.add_argument('--risk-aware-selection', type=str, help='Enable/disable risk-aware candidate scoring (true/false)')
    parser.add_argument('--gpu', type=str, help='CUDA_VISIBLE_DEVICES override (e.g., 0)')
    parser.add_argument('--dataset', type=str, help='Override dataset name')
    parser.add_argument('--domain', type=str, help='Override domain')
    parser.add_argument('--n-trials', type=int, help='Override number of trials')
    parser.add_argument('--n-seeds', type=int, help='Override number of seeds')
    
    args = parser.parse_args()
    
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Accept UTF-8 files with or without BOM for cross-platform compatibility.
    with open(args.config, 'r', encoding='utf-8-sig') as f:
        config_dict = json.load(f)
    
    if args.method:
        config_dict['method'] = args.method

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    if args.use_episodic:
        config_dict['use_episodic_search'] = str2bool(args.use_episodic)
    if args.use_variance:
        config_dict['use_variance_reduction'] = str2bool(args.use_variance)
    if args.use_meta:
        config_dict['use_meta_learning'] = str2bool(args.use_meta)
    if args.confidence_gated_episodic:
        config_dict['enable_confidence_gated_episodic'] = str2bool(args.confidence_gated_episodic)
    if args.risk_aware_selection:
        config_dict['enable_risk_aware_selection'] = str2bool(args.risk_aware_selection)
    
    if args.dataset:
        config_dict['dataset'] = args.dataset
    if args.domain:
        config_dict['domain'] = args.domain
    if args.n_trials:
        config_dict['n_trials'] = args.n_trials
    if args.n_seeds:
        config_dict['n_seeds'] = args.n_seeds
        
    config = ExperimentConfig(config_dict)
    
    runner = ExperimentRunner(config, output_file=args.output)
    runner.run_experiment()


if __name__ == "__main__":
    main()
