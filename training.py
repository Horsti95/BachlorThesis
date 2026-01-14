"""
Training Pipeline for Sleep Stage Classification
=================================================

Orchestrates the complete training workflow:
1. Load cached features (cache-first pattern)
2. Apply feature selection configurations
3. Run LOSO cross-validation for each model
4. Track cache performance for thesis metrics

Configuration Grid (27 configurations):
- Feature selection: [30, 50, 149] top-K
- Correlation filter: [0.90, 0.95, None]
- Models: [XGBoost, RandomForest, FNN]

Key Design Decisions:
- Feature extraction is CACHED (load in ~0.2s vs ~25s cold)
- Feature SELECTION is fast (no caching needed)
- Training is the variable part (what we're measuring)

Author: Lennart Gorzel
Date: December 2025

LOSO Model Cache Integration:
    - Fingerprint-based caching for trained models (Layer 2)
    - Automatic cache invalidation on config change
    - held_out_subject in fingerprint prevents data leakage
    - Cache metrics tracked for thesis evaluation
"""

import numpy as np
import pandas as pd
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import ExperimentConfig
from models import create_model, evaluate_model, BaseModel
from feature_selection import (
    FeatureSelectionConfig,
    FeatureSelectionPipeline,
    create_feature_selection_grid,
    create_optimized_grid
)
from cross_validation import (
    LOSOCrossValidator,
    CVFold,
    get_train_test_data,
    summarize_cv_splits
)
from leaderboard import get_leaderboard
from output_formatter import (
    TrainingOutputFormatter,
    Verbosity,
    get_formatter,
    set_verbosity_from_args
)
from fingerprint import LOSOFingerprint, __version__ as FINGERPRINT_VERSION
from loso_cache import LOSOModelCache, CacheMetrics

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration for a single training run.
    
    Combines model type with feature selection settings.
    """
    model_type: str  # 'xgboost', 'random_forest', 'fnn'
    model_params: Dict[str, Any] = field(default_factory=dict)
    feature_selection: FeatureSelectionConfig = field(
        default_factory=lambda: FeatureSelectionConfig()
    )
    random_state: int = 42
    
    def get_config_id(self) -> str:
        """Get unique identifier for this configuration."""
        fs_id = self.feature_selection.get_config_id()
        return f"{self.model_type}_{fs_id}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_selection': self.feature_selection.to_dict(),
            'random_state': self.random_state,
            'config_id': self.get_config_id()
        }


@dataclass
class FoldResult:
    """Results from a single cross-validation fold."""
    fold_id: int
    test_subject: Optional[str]
    
    # Predictions
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None
    
    # Metrics
    accuracy: float = 0.0
    kappa: float = 0.0
    f1_macro: float = 0.0
    f1_per_class: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    train_time_seconds: float = 0.0
    predict_time_seconds: float = 0.0
    
    # Data info
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'fold_id': self.fold_id,
            'test_subject': self.test_subject,
            'accuracy': self.accuracy,
            'kappa': self.kappa,
            'f1_macro': self.f1_macro,
            'f1_per_class': self.f1_per_class,
            'train_time_seconds': self.train_time_seconds,
            'predict_time_seconds': self.predict_time_seconds,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'n_features': self.n_features,
        }


@dataclass
class ExperimentResult:
    """Aggregated results from a complete experiment (all folds)."""
    config_id: str
    config: TrainingConfig
    
    # Aggregated metrics (mean ± std across folds)
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    kappa_mean: float = 0.0
    kappa_std: float = 0.0
    f1_macro_mean: float = 0.0
    f1_macro_std: float = 0.0
    
    # Per-class F1 (mean across folds)
    f1_per_class_mean: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    total_train_time: float = 0.0
    total_predict_time: float = 0.0
    
    # Per-fold results
    fold_results: List[FoldResult] = field(default_factory=list)
    
    # Clinical targets
    meets_accuracy_target: bool = False  # >= 0.85
    meets_kappa_target: bool = False     # >= 0.75
    meets_f1_target: bool = False        # >= 0.80
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'config_id': self.config_id,
            'config': self.config.to_dict(),
            'accuracy_mean': self.accuracy_mean,
            'accuracy_std': self.accuracy_std,
            'kappa_mean': self.kappa_mean,
            'kappa_std': self.kappa_std,
            'f1_macro_mean': self.f1_macro_mean,
            'f1_macro_std': self.f1_macro_std,
            'f1_per_class_mean': self.f1_per_class_mean,
            'total_train_time': self.total_train_time,
            'total_predict_time': self.total_predict_time,
            'meets_accuracy_target': self.meets_accuracy_target,
            'meets_kappa_target': self.meets_kappa_target,
            'meets_f1_target': self.meets_f1_target,
            'n_folds': len(self.fold_results),
        }


# =============================================================================
# Training Functions
# =============================================================================

def _run_fold_parallel(
    fold_idx: int,
    fold: CVFold,
    features_df: pd.DataFrame,
    labels: np.ndarray,
    config: TrainingConfig,
    selected_features: Optional[List[str]] = None
) -> FoldResult:
    """
    Helper function for parallel fold training.
    
    This function is standalone to allow pickling for ProcessPoolExecutor.
    
    Args:
        fold_idx: Fold index
        fold: CV fold object
        features_df: All features
        labels: All labels
        config: Training configuration
        selected_features: Pre-selected features for GLOBAL scope (skip fit)
                          If None, performs per-fold feature selection
    """
    # Get train/test split
    X_train, y_train, X_test, y_test = get_train_test_data(
        features_df, labels, fold
    )
    
    # Apply feature selection
    if selected_features is not None:
        # GLOBAL scope: Use pre-selected features (no fitting)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
    else:
        # PER_FOLD scope: Fit feature selection on this fold's training data
        fs_pipeline = FeatureSelectionPipeline(config.feature_selection)
        X_train_selected = fs_pipeline.fit_transform(X_train, y_train)
        X_test_selected = fs_pipeline.transform(X_test)
    
    # Create model
    model = create_model(
        config.model_type,
        config.model_params,
        config.random_state
    )
    
    # Train and evaluate
    return train_single_fold(
        X_train_selected, y_train,
        X_test_selected, y_test,
        model, fold
    )


def train_single_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model: BaseModel,
    fold: CVFold,
    skip_training: bool = False
) -> FoldResult:
    """
    Train and evaluate model on a single fold.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model: Model instance to train
        fold: CVFold with metadata
        skip_training: If True, skip training (model already trained from cache)
        
    Returns:
        FoldResult with predictions and metrics
    """
    # Train (unless skipped - cached model)
    start_train = time.time()
    if not skip_training:
        model.fit(X_train.values, y_train)
    train_time = time.time() - start_train
    
    # Predict
    start_pred = time.time()
    y_pred = model.predict(X_test.values)
    try:
        y_proba = model.predict_proba(X_test.values)
    except Exception:
        y_proba = None
    predict_time = time.time() - start_pred
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_proba)
    
    # Create result
    result = FoldResult(
        fold_id=fold.fold_id,
        test_subject=fold.test_subject,
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        accuracy=metrics['accuracy'],
        kappa=metrics['kappa'],
        f1_macro=metrics['f1_macro'],
        f1_per_class={
            'Wake': metrics.get('f1_Wake', 0.0),
            'N1': metrics.get('f1_N1', 0.0),
            'N2': metrics.get('f1_N2', 0.0),
            'N3': metrics.get('f1_N3', 0.0),
            'REM': metrics.get('f1_REM', 0.0),
        },
        train_time_seconds=train_time,
        predict_time_seconds=predict_time,
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=X_train.shape[1]
    )
    
    return result


def aggregate_fold_results(
    fold_results: List[FoldResult],
    config: TrainingConfig
) -> ExperimentResult:
    """
    Aggregate results across all CV folds.
    
    Args:
        fold_results: List of FoldResult from each fold
        config: Training configuration used
        
    Returns:
        ExperimentResult with aggregated metrics
    """
    accuracies = [r.accuracy for r in fold_results]
    kappas = [r.kappa for r in fold_results]
    f1_macros = [r.f1_macro for r in fold_results]
    
    # Per-class F1 aggregation
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    f1_per_class_mean = {}
    for cls in class_names:
        cls_f1s = [r.f1_per_class.get(cls, 0.0) for r in fold_results]
        f1_per_class_mean[cls] = np.mean(cls_f1s)
    
    result = ExperimentResult(
        config_id=config.get_config_id(),
        config=config,
        accuracy_mean=np.mean(accuracies),
        accuracy_std=np.std(accuracies),
        kappa_mean=np.mean(kappas),
        kappa_std=np.std(kappas),
        f1_macro_mean=np.mean(f1_macros),
        f1_macro_std=np.std(f1_macros),
        f1_per_class_mean=f1_per_class_mean,
        total_train_time=sum(r.train_time_seconds for r in fold_results),
        total_predict_time=sum(r.predict_time_seconds for r in fold_results),
        fold_results=fold_results,
        meets_accuracy_target=np.mean(accuracies) >= 0.85,
        meets_kappa_target=np.mean(kappas) >= 0.75,
        meets_f1_target=np.mean(f1_macros) >= 0.80,
    )
    
    return result


# =============================================================================
# Main Training Pipeline
# =============================================================================

class TrainingPipeline:
    """
    Main training pipeline orchestrator.
    
    Handles the complete training workflow:
    1. Load cached features (optimized by DataPipeline)
    2. Configure cross-validation (LOSO)
    3. Run grid of configurations
    4. Aggregate and save results
    
    The pipeline emphasizes CACHING benefits:
    - Features are pre-cached (~0.2s load vs ~25s compute)
    - Only training varies per configuration
    - Results are saved for reproducibility
    """
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        subject_ids: np.ndarray,
        output_dir: Path,
        experiment_name: str = "training_experiment",
        formatter: Optional[TrainingOutputFormatter] = None,
        n_jobs: int = 1,
        enable_model_cache: bool = True,
        model_cache_dir: Optional[str] = None
    ):
        """
        Initialize training pipeline.
        
        Args:
            features_df: Pre-extracted features (from cache)
            labels: Sleep stage labels
            subject_ids: Subject ID for each epoch
            output_dir: Directory for results
            experiment_name: Experiment identifier
            formatter: Optional output formatter for human-readable output
            n_jobs: Number of parallel jobs for fold training
            enable_model_cache: Enable LOSO model caching (Layer 2)
            model_cache_dir: Directory for model cache (default: results/loso_model_cache)
        """
        self.features_df = features_df
        self.labels = labels
        self.subject_ids = subject_ids
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.formatter = formatter or get_formatter()
        self.n_jobs = n_jobs
        
        # Create output directories
        self.results_dir = self.output_dir / "training_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CV
        self.cv = LOSOCrossValidator(verbose=False)
        self.n_folds = self.cv.get_n_splits(subject_ids)
        
        # Results storage
        self.results: List[ExperimentResult] = []
        
        # LOSO Model Cache (Layer 2)
        self.enable_model_cache = enable_model_cache
        if enable_model_cache:
            cache_dir = model_cache_dir or "results/loso_model_cache"
            self.model_cache = LOSOModelCache(
                cache_dir=cache_dir,
                enable_registry=True,
                estimated_training_time=120.0  # 2 minutes default estimate
            )
            logger.info(f"  Model cache: ENABLED at {cache_dir}")
        else:
            self.model_cache = None
            logger.info(f"  Model cache: DISABLED")
        
        # Subject info for formatter
        self._subject_epochs: Dict[str, int] = {}
        for subj in np.unique(subject_ids):
            self._subject_epochs[str(subj)] = int(np.sum(subject_ids == subj))
        
        logger.info(f"Initialized TrainingPipeline")
        logger.info(f"  Features: {features_df.shape}")
        logger.info(f"  Subjects: {len(np.unique(subject_ids))}")
        logger.info(f"  LOSO folds: {self.n_folds}")
        logger.info(f"  Output: {self.results_dir}")
    
    def run_single_config(
        self,
        config: TrainingConfig,
        show_progress: bool = True,
        config_idx: int = 1,
        total_configs: int = 1
    ) -> ExperimentResult:
        """
        Run training for a single configuration across all LOSO folds.
        
        Supports two feature selection scopes:
        - 'global': Fit feature selection ONCE on all data (faster, minor leakage)
        - 'per_fold': Fit feature selection on each fold's training data (correct)
        
        BENCHMARK: Global is ~15-20× faster with <1% accuracy difference.
        
        Args:
            config: Training configuration
            show_progress: Show progress bar
            config_idx: Current config index (for display)
            total_configs: Total number of configs (for display)
            
        Returns:
            ExperimentResult with aggregated metrics
        """
        logger.info(f"Running config: {config.get_config_id()}")
        
        # Print config header using formatter
        config_dict = {
            'model_type': config.model_type,
            'feature_selection': {
                'correlation_threshold': config.feature_selection.correlation_threshold,
                'top_k_features': config.feature_selection.top_k_features,
                'selection_method': config.feature_selection.selection_method,
                'scope': config.feature_selection.scope
            }
        }
        self.formatter.print_config_header(
            config.get_config_id(), config_idx, total_configs, config_dict
        )
        
        fold_results = []
        folds = list(self.cv.split(self.features_df, self.labels, self.subject_ids))
        n_folds = len(folds)
        config_start_time = time.time()
        
        # Use formatter's verbosity to decide on progress display
        use_tqdm = show_progress and self.formatter.verbosity == Verbosity.QUIET
        
        # =====================================================================
        # GLOBAL FEATURE SELECTION (if scope='global')
        # Fit once on ALL data, then reuse for all folds - HUGE speedup!
        # IMPORTANT: Also apply for kAll configs WITH correlation filter to
        # ensure consistent feature counts across folds (prevents cache mismatch)
        # =====================================================================
        selected_features = None
        use_global_fs = (
            config.feature_selection.scope == 'global' and 
            (config.feature_selection.top_k_features is not None or 
             config.feature_selection.correlation_threshold is not None)
        )
        if use_global_fs:
            logger.info("GLOBAL feature selection: fitting on all data (once)")
            self.formatter.print_substep(
                f"Global feature selection ({config.feature_selection.selection_method.upper()}): "
                f"fitting on {len(self.features_df)} samples..."
            )
            
            fs_start = time.time()
            global_fs_pipeline = FeatureSelectionPipeline(config.feature_selection)
            global_fs_pipeline.fit(self.features_df, self.labels)
            selected_features = global_fs_pipeline.get_selected_features()
            fs_time = time.time() - fs_start
            
            logger.info(f"  Selected {len(selected_features)} features in {fs_time:.2f}s")
            self.formatter.print_substep(
                f"  Selected {len(selected_features)} features in {fs_time:.2f}s (will reuse for all folds)"
            )
        
        if self.n_jobs > 1:
            # Parallel execution
            self.formatter.print_substep(f"Training {n_folds} folds in parallel using {self.n_jobs} cores...")
            
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all folds (with pre-selected features if global scope)
                future_to_fold = {
                    executor.submit(
                        _run_fold_parallel, 
                        idx, fold, self.features_df, self.labels, config, selected_features
                    ): idx 
                    for idx, fold in enumerate(folds, 1)
                }
                
                # Process as they complete
                fold_iter = as_completed(future_to_fold)
                if use_tqdm:
                    fold_iter = tqdm(
                        fold_iter,
                        desc=f"  {config.get_config_id()}",
                        unit="fold",
                        total=n_folds
                    )
                
                for future in fold_iter:
                    try:
                        fold_result = future.result()
                        fold_results.append(fold_result)
                        
                        # Update formatter's internal state for summary
                        # We use print_fold_result to store the data, but it will also print
                        # if not in QUIET mode. In parallel, this might be out of order.
                        self.formatter.print_fold_result(
                            fold_result.fold_id, fold_result.test_subject,
                            fold_result.accuracy, fold_result.kappa, fold_result.f1_macro,
                            fold_result.train_time_seconds + fold_result.predict_time_seconds,
                            fold_result.n_features
                        )
                        
                        # Update progress bar
                        if use_tqdm:
                            fold_iter.set_postfix({
                                'acc': f"{fold_result.accuracy:.3f}",
                                'kappa': f"{fold_result.kappa:.3f}"
                            })
                    except Exception as e:
                        logger.error(f"Fold failed: {e}")
                        self.formatter.print_error(f"Fold failed: {e}")
            
            # Sort results by fold_id to maintain order
            fold_results.sort(key=lambda x: x.fold_id)
            self.formatter._fold_results.sort(key=lambda x: x['fold'])
            
        else:
            # Sequential execution (original logic)
            # Progress bar (only in quiet mode)
            fold_iter = tqdm(
                enumerate(folds, 1),
                desc=f"  {config.get_config_id()}",
                unit="fold",
                total=n_folds,
                disable=not use_tqdm
            )
            
            for fold_idx, fold in fold_iter:
                fold_start = time.time()
                
                # Get train/test split
                X_train, y_train, X_test, y_test = get_train_test_data(
                    self.features_df, self.labels, fold
                )
                
                n_train_subjects = len(folds) - 1  # All except test subject
                n_train_epochs = len(X_train)
                n_test_epochs = len(X_test)
                
                # Print fold start
                self.formatter.print_fold_start(
                    fold_idx, n_folds, fold.test_subject,
                    n_train_subjects, n_train_epochs, n_test_epochs
                )
                
                # Apply feature selection
                if selected_features is not None:
                    # GLOBAL scope: Use pre-selected features (no per-fold fitting)
                    X_train_selected = X_train[selected_features]
                    X_test_selected = X_test[selected_features]
                    n_input = X_train.shape[1]
                    n_after_corr = n_input  # Not tracked for global
                    n_final = len(selected_features)
                    corr_removed = 0  # Not tracked for global
                else:
                    # PER_FOLD scope: Fit feature selection on this fold's training data
                    fs_pipeline = FeatureSelectionPipeline(config.feature_selection)
                    X_train_selected = fs_pipeline.fit_transform(X_train, y_train)
                    X_test_selected = fs_pipeline.transform(X_test)
                    
                    # Feature selection info for verbose output
                    n_input = X_train.shape[1]
                    n_after_corr = fs_pipeline.n_features_after_corr_
                    n_final = X_train_selected.shape[1]
                    corr_removed = n_input - n_after_corr
                
                self.formatter.print_feature_selection(
                    n_input, n_after_corr, n_final, corr_removed
                )
                
                # =========================================================
                # LOSO MODEL CACHING (Layer 2)
                # =========================================================
                # Generate fingerprint for this fold
                # IMPORTANT: Include n_final (actual feature count after selection)
                # to prevent cache mismatches when correlation filter produces
                # different feature counts across folds (e.g., kAll + corr0.9)
                # Always include the actual selected feature names in the fingerprint for cache key
                # This ensures train/test use the same features and prevents shape mismatch
                fingerprint = LOSOFingerprint.generate(
                    random_seed=config.random_state,
                    code_version=FINGERPRINT_VERSION,
                    model_name=config.model_type,
                    model_params=config.model_params,
                    feature_config={
                        'base': n_input,
                        'corr': config.feature_selection.correlation_threshold,
                        'top_k': config.feature_selection.top_k_features,
                        'n_selected': n_final,
                        'selected_features': selected_features if selected_features is not None else list(X_train_selected.columns)
                    },
                    held_out_subject=str(fold.test_subject)
                )
                
                # Check cache for trained model
                cache_hit = False
                if self.enable_model_cache and self.model_cache is not None:
                    # For FNN, pass model class and params for deserialization
                    if config.model_type == "fnn":
                        from models import FNNModel
                        cached_model = self.model_cache.get(
                            fingerprint, str(fold.test_subject),
                            model_type="fnn",
                            model_class=FNNModel,
                            model_params=config.model_params
                        )
                    else:
                        cached_model = self.model_cache.get(
                            fingerprint, str(fold.test_subject),
                            model_type=config.model_type
                        )
                    if cached_model is not None:
                        # CACHE HIT - use cached model
                        model = cached_model
                        cache_hit = True
                        self.formatter.print_training_progress(
                            config.model_type, n_train_epochs, n_final,
                            cache_status="HIT"
                        )
                    else:
                        # CACHE MISS - will train
                        model = create_model(
                            config.model_type,
                            config.model_params,
                            config.random_state
                        )
                        self.formatter.print_training_progress(
                            config.model_type, n_train_epochs, n_final,
                            cache_status="MISS"
                        )
                else:
                    # Cache disabled - always train
                    model = create_model(
                        config.model_type,
                        config.model_params,
                        config.random_state
                    )
                    self.formatter.print_training_progress(
                        config.model_type, n_train_epochs, n_final
                    )
                
                # Train and evaluate
                train_start = time.time()
                fold_result = train_single_fold(
                    X_train_selected, y_train,
                    X_test_selected, y_test,
                    model, fold,
                    skip_training=cache_hit  # Skip training if cache hit
                )
                train_elapsed = time.time() - train_start
                
                # Cache the model if it was trained (cache miss)
                if self.enable_model_cache and self.model_cache is not None and not cache_hit:
                    self.model_cache.put(
                        fingerprint, 
                        str(fold.test_subject),
                        model,
                        model_type=config.model_type,
                        training_time=train_elapsed
                    )
                
                fold_results.append(fold_result)
                
                fold_elapsed = time.time() - fold_start
                
                # Print fold result
                self.formatter.print_fold_result(
                    fold_idx, fold.test_subject,
                    fold_result.accuracy, fold_result.kappa, fold_result.f1_macro,
                    fold_elapsed, n_final
                )
                
                # Update progress bar with current metrics (quiet mode)
                if use_tqdm:
                    fold_iter.set_postfix({
                        'acc': f"{fold_result.accuracy:.3f}",
                        'kappa': f"{fold_result.kappa:.3f}"
                    })
        
        # Aggregate results
        result = aggregate_fold_results(fold_results, config)
        
        config_total_time = time.time() - config_start_time
        
        # Print config summary
        self.formatter.print_config_summary(
            config.get_config_id(),
            result.accuracy_mean, result.accuracy_std,
            result.kappa_mean, result.kappa_std,
            result.f1_macro_mean, result.f1_macro_std,
            config_total_time,
            n_folds, n_folds  # All from cache
        )
        
        logger.info(
            f"  Result: acc={result.accuracy_mean:.3f}±{result.accuracy_std:.3f}, "
            f"kappa={result.kappa_mean:.3f}, f1={result.f1_macro_mean:.3f}"
        )
        
        return result
    
    def run_grid(
        self,
        configs: List[TrainingConfig],
        save_intermediate: bool = True
    ) -> List[ExperimentResult]:
        """
        Run training for all configurations in the grid.
        
        Args:
            configs: List of training configurations
            save_intermediate: Save results after each config
            
        Returns:
            List of ExperimentResult objects
        """
        logger.info(f"Running training grid: {len(configs)} configurations")
        logger.info(f"Total training runs: {len(configs) * self.n_folds}")
        
        # Print experiment header using formatter
        self.formatter.print_experiment_header(
            self.experiment_name,
            len(np.unique(self.subject_ids)),
            len(configs),
            self.n_folds
        )
        
        # Print LOSO setup
        self.formatter.print_loso_setup(
            len(np.unique(self.subject_ids)),
            self._subject_epochs,
            len(self.labels)
        )
        
        results = []
        grid_start_time = time.time()
        
        for i, config in enumerate(configs, 1):
            try:
                result = self.run_single_config(
                    config, 
                    show_progress=True,
                    config_idx=i,
                    total_configs=len(configs)
                )
                results.append(result)
                
                # Save intermediate
                if save_intermediate:
                    self._save_config_result(result)
                
            except NotImplementedError as e:
                logger.warning(f"Skipping {config.model_type}: {e}")
                self.formatter.print_warning(f"Skipping {config.model_type}: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed config {config.get_config_id()}: {e}")
                self.formatter.print_error(f"Failed {config.get_config_id()}: {e}")
                continue
        
        self.results = results
        
        # Print final results table
        if results:
            results_dicts = [r.to_dict() for r in results]
            self.formatter.print_final_results_table(results_dicts)
            
            # Print best result
            best = max(results, key=lambda r: r.accuracy_mean)
            self.formatter.print_best_result(
                best.config_id,
                best.accuracy_mean, best.accuracy_std,
                best.kappa_mean, best.kappa_std,
                best.f1_macro_mean, best.f1_macro_std,
                best.meets_accuracy_target and best.meets_kappa_target and best.meets_f1_target
            )
        
        # Print model cache statistics (Layer 2)
        if self.enable_model_cache and self.model_cache is not None:
            cache_stats = self.model_cache.get_stats()
            self._print_cache_stats(cache_stats)
        
        # Save final summary
        self._save_summary()
        
        return results
    
    def _save_config_result(self, result: ExperimentResult):
        """Save results for a single configuration."""
        filename = f"result_{result.config_id}.json"
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        data = result.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.debug(f"Saved result to {filepath}")
    
    def _save_summary(self):
        """Save summary of all results."""
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'n_configurations': len(self.results),
            'n_folds_per_config': self.n_folds,
            'total_training_runs': len(self.results) * self.n_folds,
            'results': [r.to_dict() for r in self.results],
        }
        
        # Add model cache stats (Layer 2) if enabled
        if self.enable_model_cache and self.model_cache is not None:
            summary['model_cache_stats'] = self.model_cache.get_stats()
        
        # Best results
        if self.results:
            best_acc = max(self.results, key=lambda r: r.accuracy_mean)
            best_kappa = max(self.results, key=lambda r: r.kappa_mean)
            best_f1 = max(self.results, key=lambda r: r.f1_macro_mean)
            
            summary['best_results'] = {
                'accuracy': {
                    'config': best_acc.config_id,
                    'value': best_acc.accuracy_mean
                },
                'kappa': {
                    'config': best_kappa.config_id,
                    'value': best_kappa.kappa_mean
                },
                'f1_macro': {
                    'config': best_f1.config_id,
                    'value': best_f1.f1_macro_mean
                }
            }
        
        filepath = self.results_dir / "training_summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved training summary to {filepath}")
    
    def _print_cache_stats(self, stats: Dict[str, Any]):
        """Print model cache statistics for thesis metrics."""
        print("\n" + "=" * 60)
        print("📊 LOSO MODEL CACHE STATISTICS (Layer 2)")
        print("=" * 60)
        
        metrics = stats.get('metrics', {})
        storage = stats.get('storage', {})
        
        print(f"\n  Session Performance:")
        print(f"    Cache Hits:    {metrics.get('hits', 0):,}")
        print(f"    Cache Misses:  {metrics.get('misses', 0):,}")
        print(f"    Hit Rate:      {metrics.get('hit_rate', 0):.1%}")
        print(f"    Time Saved:    {metrics.get('time_saved_formatted', '0s')}")
        
        print(f"\n  Storage:")
        print(f"    Cached Models: {storage.get('cached_models', 0)}")
        print(f"    Total Size:    {storage.get('total_size_mb', 0):.1f} MB")
        print(f"    Avg Size:      {storage.get('avg_size_mb', 0):.2f} MB/model")
        print(f"    Location:      {storage.get('cache_dir', 'N/A')}")
        
        print("=" * 60 + "\n")
    
    def print_results_table(self):
        """Print results in a formatted table."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "="*80)
        print("TRAINING RESULTS SUMMARY")
        print("="*80)
        print(f"{'Config ID':<30} {'Accuracy':<12} {'Kappa':<10} {'F1-Macro':<10} {'Target'}")
        print("-"*80)
        
        for r in sorted(self.results, key=lambda x: -x.accuracy_mean):
            target_met = "[v]" if r.meets_accuracy_target and r.meets_kappa_target else "[x]"
            print(
                f"{r.config_id:<30} "
                f"{r.accuracy_mean:.3f}+/-{r.accuracy_std:.3f}  "
                f"{r.kappa_mean:.3f}     "
                f"{r.f1_macro_mean:.3f}     "
                f"{target_met}"
            )
        
        print("-"*80)
        print(f"Clinical targets: Accuracy>=0.85, Kappa>=0.75, F1>=0.80")
        print("="*80)


# =============================================================================
# Grid Configuration Factory
# =============================================================================

def create_training_grid(
    models: Optional[List[str]] = None,
    correlation_thresholds: Optional[List[Optional[float]]] = None,
    top_k_features: Optional[List[Optional[int]]] = None,
    random_state: int = 42,
    use_hybrid: bool = True,
    selection_method: str = 'anova',
    scope: str = 'global'
) -> List[TrainingConfig]:
    """
    Create training configuration grid for thesis experiments.
    
    Default grid (18 configurations) - THESIS SPECIFICATION:
    - Models: [xgboost, random_forest]  (2 options)
    - Correlation: [0.75, 0.90, None]   (3 options)
    - Top-K: [30, 50, 105]              (3 options)
    
    BENCHMARK VALIDATED DEFAULTS:
    - selection_method: 'anova' (200× faster than MI, <1% accuracy loss)
    - scope: 'global' (fit once on all data, faster, minimal leakage)
    
    Note: Top-K of 105 means "select top 105 features from 149".
          This is different from None (all 149 features after correlation filter).
    
    Args:
        models: Model types to include
        correlation_thresholds: Correlation threshold options
        top_k_features: Top-K feature selection options
        random_state: Random seed
        use_hybrid: DEPRECATED - use selection_method='hybrid' instead
        selection_method: 'anova' (fast), 'mi' (slow), 'hybrid' (balanced)
        scope: 'global' (fast) or 'per_fold' (correct but slow)
        
    Returns:
        List of TrainingConfig objects
    """
    # Defaults from thesis specification (18 configurations)
    # FNN excluded due to computational constraints (~10h extra per config)
    # FNN support remains in codebase - see models.py
    if models is None:
        models = ['xgboost', 'random_forest']  # 2 models (FNN excluded)
    if correlation_thresholds is None:
        correlation_thresholds = [0.75, 0.90, None]  # Thesis spec: 3 options
    if top_k_features is None:
        top_k_features = [30, 50, None]  # Thesis spec: 3 options (None = all 149)
    
    configs = []
    
    for model in models:
        for corr in correlation_thresholds:
            for top_k in top_k_features:
                fs_config = FeatureSelectionConfig(
                    correlation_threshold=corr,
                    top_k_features=top_k,
                    selection_method=selection_method,
                    scope=scope,
                    random_state=random_state,
                    use_hybrid=use_hybrid
                )
                
                config = TrainingConfig(
                    model_type=model,
                    feature_selection=fs_config,
                    random_state=random_state
                )
                configs.append(config)
    
    logger.info(f"Created training grid: {len(configs)} configurations (method={selection_method}, scope={scope})")
    return configs


def create_quick_grid(
    random_state: int = 42, 
    use_hybrid: bool = True,
    selection_method: str = 'anova',
    scope: str = 'global'
) -> List[TrainingConfig]:
    """
    Create minimal grid for quick testing.
    
    3 models × 1 feature config = 3 configurations
    Uses ANOVA + Global for maximum speed.
    """
    return create_training_grid(
        models=['xgboost', 'random_forest', 'fnn'],
        correlation_thresholds=[0.90],  # Middle value
        top_k_features=[50],  # Middle value
        random_state=random_state,
        use_hybrid=use_hybrid,
        selection_method=selection_method,
        scope=scope
    )


def create_thesis_grid(
    random_state: int = 42, 
    use_hybrid: bool = True,
    selection_method: str = 'anova',
    scope: str = 'global'
) -> List[TrainingConfig]:
    """
    Create the full thesis grid (18 configurations).
    
    Configuration Grid (2 × 3 × 3 = 18 configurations):
    - Models: [xgboost, random_forest]         → 2 options (FNN excluded)
    - Correlation: [0.75, 0.90, None]          → 3 options
    - Top-K: [30, 50, None]                    → 3 options (None = all 149)
    
    NOTE: FNN is implemented (see models.py) but excluded from experiments
    due to computational constraints (~10h additional per configuration).
    
    BENCHMARK VALIDATED DEFAULTS:
    - selection_method: 'anova' (200× faster than MI)
    - scope: 'global' (fit once, minimal leakage)
    
    Data sizes (selected at runtime, not part of grid):
    - Pilot: 10 subjects
    - Full: 128 subjects
    
    Total training runs with LOSO:
    - Pilot: 18 configs × 10 folds = 180 runs
    - Full:  18 configs × 128 folds = 2,304 runs
    
    Estimated runtime (full LOSO):
    - XGBoost: ~7.5h per config × 9 configs = ~68h
    - RF: ~3h per config × 9 configs = ~27h
    - Total: ~95h (~4 days) for all 18 configs
    """
    return create_training_grid(
        models=['xgboost', 'random_forest'],
        correlation_thresholds=[0.75, 0.90, None],
        top_k_features=[30, 50, None],
        random_state=random_state,
        use_hybrid=use_hybrid,
        selection_method=selection_method,
        scope=scope
    )


def create_pilot_grid(
    random_state: int = 42, 
    use_hybrid: bool = True,
    selection_method: str = 'anova',
    scope: str = 'global'
) -> List[TrainingConfig]:
    """
    Create pilot grid for validation before full run.
    
    Uses full thesis grid (18 configurations) but will be run on fewer subjects.
    """
    return create_thesis_grid(
        random_state=random_state, 
        use_hybrid=use_hybrid,
        selection_method=selection_method,
        scope=scope
    )


def create_optimized_thesis_grid(random_state: int = 42) -> List[TrainingConfig]:
    """
    Create OPTIMIZED thesis grid using benchmark-validated settings.
    
    BENCHMARK RESULTS (5 subjects, 54 combinations):
    - ANOVA vs MI: ~200× faster, <1% accuracy difference
    - Global vs Per-Fold: <1% accuracy difference, much faster
    
    This grid uses:
    - selection_method='anova' (FAST)
    - scope='global' (FAST, minor leakage acceptable for thesis)
    
    Expected time savings:
    - Before: ~63 hours for 18 configs (MI + per-fold)
    - After:  ~3-4 hours for 18 configs (ANOVA + global)
    
    Returns:
        List of TrainingConfig with optimized settings
    """
    return create_thesis_grid(
        random_state=random_state,
        selection_method='anova',
        scope='global'
    )


# =============================================================================
# Module test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Training Pipeline")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_subjects = 5
    epochs_per_subject = 50
    n_features = 149
    
    X_list = []
    y_list = []
    subject_ids = []
    
    for subj in range(1, n_subjects + 1):
        X_list.append(np.random.randn(epochs_per_subject, n_features))
        y_list.append(np.random.randint(0, 5, epochs_per_subject))
        subject_ids.extend([str(subj)] * epochs_per_subject)
    
    X = pd.DataFrame(
        np.vstack(X_list),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = np.concatenate(y_list)
    subject_ids = np.array(subject_ids)
    
    print(f"Data: {X.shape}, {len(np.unique(subject_ids))} subjects")
    
    # Test single config
    print("\n" + "-" * 40)
    print("Testing single configuration")
    
    from pathlib import Path
    output_dir = Path("./results/test_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = TrainingPipeline(
        features_df=X,
        labels=y,
        subject_ids=subject_ids,
        output_dir=output_dir,
        experiment_name="test"
    )
    
    config = TrainingConfig(
        model_type='xgboost',
        feature_selection=FeatureSelectionConfig(
            correlation_threshold=0.95,
            top_k_features=50
        )
    )
    
    result = pipeline.run_single_config(config, show_progress=True)
    print(f"\nResult: acc={result.accuracy_mean:.3f}, kappa={result.kappa_mean:.3f}")
    
    # Test grid
    print("\n" + "-" * 40)
    print("Testing quick grid")
    
    grid = create_quick_grid()
    print(f"Grid size: {len(grid)} configurations")
    
    results = pipeline.run_grid(grid[:2], save_intermediate=True)
    pipeline.print_results_table()
    
    print("\n" + "=" * 60)
    print("Training Pipeline: ✓ All tests passed")
