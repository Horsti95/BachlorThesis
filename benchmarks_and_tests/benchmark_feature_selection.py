#!/usr/bin/env python3
"""
Benchmark Module for Feature Selection Strategies

This module tests ALL combinations of:
- Feature Selection Methods: MI, ANOVA (f_classif), HYBRID
- Feature Selection Scope: Per-Fold vs Global
- Correlation Thresholds: 0.75, 0.9, 0.95, None
- Top-K Features: 30, 50, 105

Uses 5 random subjects with full LOSO cross-validation (5 folds)
to compare time and quality metrics.

Author: Benchmark for Bachelor Thesis
Date: December 2024
"""

import json
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from itertools import product

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

# Local imports
from feature_cache import load_features_from_cache
from models import XGBoostModel

# Constants
RANDOM_STATE = 42
GLOBAL_CACHE_DIR = Path("./results/features_cache_global")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiment."""
    n_subjects: int = 5
    random_seed: int = 42
    
    # Feature selection methods to test
    selection_methods: List[str] = field(default_factory=lambda: [
        'anova',      # f_classif only (fast)
        'mi',         # Mutual Information only (slow)
        'hybrid',     # f_classif -> MI (balanced)
    ])
    
    # Feature selection scope
    selection_scopes: List[str] = field(default_factory=lambda: [
        'per_fold',   # Fit on each fold's training data (correct, slower)
        'global',     # Fit once on all data (faster, but potential leakage)
    ])
    
    # Correlation thresholds (None = no filter)
    correlation_thresholds: List[Optional[float]] = field(default_factory=lambda: [
        0.75, 0.9, 0.95, None
    ])
    
    # Top-K features
    top_k_values: List[int] = field(default_factory=lambda: [30, 50, 105])
    
    # Output directory
    output_dir: str = './results/benchmark'


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    # Configuration
    method: str
    scope: str
    correlation: Optional[float]
    top_k: int
    
    # Timing (seconds)
    time_feature_selection: float
    time_training: float
    time_total: float
    time_per_fold_avg: float
    
    # Quality metrics (mean ± std across folds)
    accuracy_mean: float
    accuracy_std: float
    kappa_mean: float
    kappa_std: float
    f1_mean: float
    f1_std: float
    
    # Additional info
    n_features_selected: int
    n_folds: int
    n_samples: int


# =============================================================================
# Feature Selection Methods
# =============================================================================

class CorrelationFilter:
    """Remove highly correlated features."""
    
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.features_to_keep_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'CorrelationFilter':
        if self.threshold is None:
            self.features_to_keep_ = np.arange(X.shape[1])
            return self
            
        corr_matrix = np.corrcoef(X.T)
        n_features = X.shape[1]
        to_remove = set()
        
        for i in range(n_features):
            if i in to_remove:
                continue
            for j in range(i + 1, n_features):
                if j in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > self.threshold:
                    to_remove.add(j)
        
        self.features_to_keep_ = np.array([i for i in range(n_features) if i not in to_remove])
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.features_to_keep_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class FeatureSelector:
    """Unified feature selector supporting multiple methods."""
    
    def __init__(self, method: str = 'hybrid', k: int = 50, correlation: Optional[float] = 0.9):
        self.method = method
        self.k = k
        self.correlation = correlation
        
        self.corr_filter_ = None
        self.selector_ = None
        self.selected_indices_ = None
        
        # Timing
        self.time_correlation_ = 0
        self.time_selection_ = 0
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        # Step 1: Correlation filter
        t0 = time.time()
        if self.correlation is not None:
            self.corr_filter_ = CorrelationFilter(self.correlation)
            X_filtered = self.corr_filter_.fit_transform(X, y)
        else:
            X_filtered = X
        self.time_correlation_ = time.time() - t0
        
        # Determine actual k
        actual_k = min(self.k, X_filtered.shape[1])
        
        # Step 2: Feature selection based on method
        t0 = time.time()
        
        if self.method == 'anova':
            # Fast: ANOVA F-test only
            self.selector_ = SelectKBest(f_classif, k=actual_k)
            self.selector_.fit(X_filtered, y)
            
        elif self.method == 'mi':
            # Slow: Mutual Information only
            mi_scores = mutual_info_classif(X_filtered, y, random_state=RANDOM_STATE)
            top_indices = np.argsort(mi_scores)[-actual_k:]
            self.selector_ = {'type': 'mi', 'indices': top_indices, 'scores': mi_scores}
            
        elif self.method == 'hybrid':
            # Balanced: ANOVA prefilter -> MI on reduced set
            n_prefilter = min(2 * actual_k, X_filtered.shape[1])
            
            # Stage 1: ANOVA
            f_scores, _ = f_classif(X_filtered, y)
            prefilter_indices = np.argsort(f_scores)[-n_prefilter:]
            X_prefiltered = X_filtered[:, prefilter_indices]
            
            # Stage 2: MI on reduced set
            mi_scores = mutual_info_classif(X_prefiltered, y, random_state=RANDOM_STATE)
            top_mi_indices = np.argsort(mi_scores)[-actual_k:]
            final_indices = prefilter_indices[top_mi_indices]
            
            self.selector_ = {'type': 'hybrid', 'indices': final_indices}
        
        self.time_selection_ = time.time() - t0
        
        # Store final selected indices (relative to correlation-filtered features)
        if isinstance(self.selector_, dict):
            self._selected_in_filtered = self.selector_['indices']
        else:
            self._selected_in_filtered = self.selector_.get_support(indices=True)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Apply correlation filter
        if self.corr_filter_ is not None:
            X_filtered = self.corr_filter_.transform(X)
        else:
            X_filtered = X
        
        # Apply feature selection
        return X_filtered[:, self._selected_in_filtered]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
    
    @property
    def total_time(self) -> float:
        return self.time_correlation_ + self.time_selection_
    
    @property
    def n_features_out(self) -> int:
        return len(self._selected_in_filtered)


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Run benchmark experiments."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.cache_dir = GLOBAL_CACHE_DIR
        self.results: List[BenchmarkResult] = []
    
    def list_cached_subjects(self) -> List[int]:
        """List all subjects with cached features."""
        if not self.cache_dir.exists():
            return []
        
        subjects = []
        for f in self.cache_dir.glob("subject_*_full.npz"):
            try:
                subj_id = int(f.stem.split('_')[1])
                subjects.append(subj_id)
            except (ValueError, IndexError):
                continue
        return sorted(subjects)
    
    def load_subject(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load features and labels for a single subject."""
        cache_path = self.cache_dir / f"subject_{subject_id}_full.npz"
        result = load_features_from_cache(cache_path)
        if result is None:
            raise FileNotFoundError(f"No cache found for subject {subject_id}")
        features_df, labels, n_channels = result
        return features_df.values, labels
        
    def load_subjects(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data for random subjects."""
        # Set seed for reproducibility
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        # Get all cached subjects
        all_subjects = self.list_cached_subjects()
        if len(all_subjects) < self.config.n_subjects:
            raise ValueError(f"Not enough cached subjects. Found {len(all_subjects)}, need {self.config.n_subjects}")
        
        # Select random subjects
        selected_subjects = sorted(random.sample(all_subjects, self.config.n_subjects))
        logger.info(f"Selected subjects: {selected_subjects}")
        
        # Load data
        X_list, y_list, subjects_list = [], [], []
        for subj in selected_subjects:
            X_subj, y_subj = self.load_subject(subj)
            X_list.append(X_subj)
            y_list.append(y_subj)
            subjects_list.extend([subj] * len(y_subj))
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        subjects = np.array(subjects_list)
        
        logger.info(f"Loaded {len(X)} samples, {X.shape[1]} features from {self.config.n_subjects} subjects")
        return X, y, subjects
    
    def run_single_benchmark(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        method: str,
        scope: str,
        correlation: Optional[float],
        top_k: int
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        
        logo = LeaveOneGroupOut()
        n_folds = logo.get_n_splits(X, y, subjects)
        
        fold_times = []
        fold_metrics = {'accuracy': [], 'kappa': [], 'f1': []}
        total_feature_selection_time = 0
        total_training_time = 0
        n_features_selected = 0
        
        # Global feature selection (if scope == 'global')
        global_selector = None
        if scope == 'global':
            t0 = time.time()
            global_selector = FeatureSelector(method=method, k=top_k, correlation=correlation)
            global_selector.fit(X, y)
            total_feature_selection_time = global_selector.total_time
            n_features_selected = global_selector.n_features_out
        
        # LOSO cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, subjects)):
            fold_start = time.time()
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Feature selection
            if scope == 'per_fold':
                t0 = time.time()
                selector = FeatureSelector(method=method, k=top_k, correlation=correlation)
                X_train_sel = selector.fit_transform(X_train, y_train)
                X_test_sel = selector.transform(X_test)
                total_feature_selection_time += selector.total_time
                n_features_selected = selector.n_features_out
            else:
                X_train_sel = global_selector.transform(X_train)
                X_test_sel = global_selector.transform(X_test)
            
            # Training
            t0 = time.time()
            model = XGBoostModel(random_seed=RANDOM_STATE)
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_test_sel)
            training_time = time.time() - t0
            total_training_time += training_time
            
            # Metrics
            fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_metrics['kappa'].append(cohen_kappa_score(y_test, y_pred))
            fold_metrics['f1'].append(f1_score(y_test, y_pred, average='macro'))
            
            fold_times.append(time.time() - fold_start)
        
        # Aggregate results
        return BenchmarkResult(
            method=method,
            scope=scope,
            correlation=correlation,
            top_k=top_k,
            time_feature_selection=total_feature_selection_time,
            time_training=total_training_time,
            time_total=sum(fold_times),
            time_per_fold_avg=np.mean(fold_times),
            accuracy_mean=np.mean(fold_metrics['accuracy']),
            accuracy_std=np.std(fold_metrics['accuracy']),
            kappa_mean=np.mean(fold_metrics['kappa']),
            kappa_std=np.std(fold_metrics['kappa']),
            f1_mean=np.mean(fold_metrics['f1']),
            f1_std=np.std(fold_metrics['f1']),
            n_features_selected=n_features_selected,
            n_folds=n_folds,
            n_samples=len(X)
        )
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark combinations."""
        
        # Load data
        X, y, subjects = self.load_subjects()
        
        # Generate all combinations
        combinations = list(product(
            self.config.selection_methods,
            self.config.selection_scopes,
            self.config.correlation_thresholds,
            self.config.top_k_values
        ))
        
        logger.info(f"Running {len(combinations)} benchmark combinations...")
        
        results = []
        for method, scope, correlation, top_k in tqdm(combinations, desc="Benchmarking"):
            corr_str = f"{correlation}" if correlation else "None"
            logger.info(f"Testing: {method} | {scope} | corr={corr_str} | k={top_k}")
            
            try:
                result = self.run_single_benchmark(
                    X, y, subjects, method, scope, correlation, top_k
                )
                results.append(result)
                
                # Log result
                logger.info(f"  -> Acc={result.accuracy_mean:.3f}±{result.accuracy_std:.3f}, "
                          f"Time={result.time_total:.1f}s ({result.time_per_fold_avg:.1f}s/fold)")
            except Exception as e:
                logger.error(f"  -> FAILED: {e}")
        
        self.results = results
        return results
    
    def save_results(self, output_dir: Optional[str] = None):
        """Save benchmark results to files."""
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = output_dir / f"benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)
        logger.info(f"Saved JSON results to {json_path}")
        
        # Save as CSV
        csv_path = output_dir / f"benchmark_{timestamp}.csv"
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV results to {csv_path}")
        
        # Generate summary report
        self._generate_report(output_dir, timestamp)
        
        return json_path, csv_path
    
    def _generate_report(self, output_dir: Path, timestamp: str):
        """Generate markdown report."""
        report_path = output_dir / f"benchmark_report_{timestamp}.md"
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        with open(report_path, 'w') as f:
            f.write("# Feature Selection Benchmark Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Subjects:** {self.config.n_subjects}\n\n")
            f.write(f"**Combinations tested:** {len(self.results)}\n\n")
            
            # Best by accuracy
            f.write("## Top 10 by Accuracy\n\n")
            top_acc = df.nlargest(10, 'accuracy_mean')
            f.write(top_acc[['method', 'scope', 'correlation', 'top_k', 
                           'accuracy_mean', 'accuracy_std', 'time_total']].to_markdown(index=False))
            f.write("\n\n")
            
            # Best by time (with accuracy > 0.7)
            f.write("## Top 10 Fastest (Accuracy > 0.7)\n\n")
            fast = df[df['accuracy_mean'] > 0.7].nsmallest(10, 'time_total')
            f.write(fast[['method', 'scope', 'correlation', 'top_k',
                         'accuracy_mean', 'time_total', 'time_per_fold_avg']].to_markdown(index=False))
            f.write("\n\n")
            
            # Method comparison
            f.write("## Method Comparison (Average)\n\n")
            method_stats = df.groupby('method').agg({
                'accuracy_mean': 'mean',
                'kappa_mean': 'mean',
                'time_total': 'mean',
                'time_feature_selection': 'mean'
            }).round(3)
            f.write(method_stats.to_markdown())
            f.write("\n\n")
            
            # Scope comparison
            f.write("## Scope Comparison (Average)\n\n")
            scope_stats = df.groupby('scope').agg({
                'accuracy_mean': 'mean',
                'time_total': 'mean',
                'time_feature_selection': 'mean'
            }).round(3)
            f.write(scope_stats.to_markdown())
            f.write("\n\n")
            
            # Correlation comparison
            f.write("## Correlation Threshold Comparison\n\n")
            corr_stats = df.groupby('correlation').agg({
                'accuracy_mean': 'mean',
                'time_total': 'mean'
            }).round(3)
            f.write(corr_stats.to_markdown())
            f.write("\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            best_quality = df.loc[df['accuracy_mean'].idxmax()]
            f.write(f"### Best Quality\n")
            f.write(f"- **Method:** {best_quality['method']}\n")
            f.write(f"- **Scope:** {best_quality['scope']}\n")
            f.write(f"- **Correlation:** {best_quality['correlation']}\n")
            f.write(f"- **Top-K:** {best_quality['top_k']}\n")
            f.write(f"- **Accuracy:** {best_quality['accuracy_mean']:.3f} ± {best_quality['accuracy_std']:.3f}\n")
            f.write(f"- **Time:** {best_quality['time_total']:.1f}s\n\n")
            
            # Best speed with good quality
            good_quality = df[df['accuracy_mean'] > df['accuracy_mean'].quantile(0.75)]
            if len(good_quality) > 0:
                best_speed = good_quality.loc[good_quality['time_total'].idxmin()]
                f.write(f"### Best Speed (Top 25% Quality)\n")
                f.write(f"- **Method:** {best_speed['method']}\n")
                f.write(f"- **Scope:** {best_speed['scope']}\n")
                f.write(f"- **Correlation:** {best_speed['correlation']}\n")
                f.write(f"- **Top-K:** {best_speed['top_k']}\n")
                f.write(f"- **Accuracy:** {best_speed['accuracy_mean']:.3f} ± {best_speed['accuracy_std']:.3f}\n")
                f.write(f"- **Time:** {best_speed['time_total']:.1f}s\n")
        
        logger.info(f"Saved report to {report_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Feature Selection Strategies")
    parser.add_argument('--subjects', '-n', type=int, default=5,
                       help='Number of random subjects to test (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for subject selection')
    parser.add_argument('--output', '-o', type=str, default='./results/benchmark',
                       help='Output directory')
    parser.add_argument('--methods', nargs='+', default=['anova', 'mi', 'hybrid'],
                       choices=['anova', 'mi', 'hybrid'],
                       help='Feature selection methods to test')
    parser.add_argument('--scopes', nargs='+', default=['per_fold', 'global'],
                       choices=['per_fold', 'global'],
                       help='Feature selection scopes to test')
    parser.add_argument('--correlations', nargs='+', type=float, default=[0.75, 0.9, 0.95],
                       help='Correlation thresholds (use 0 for None)')
    parser.add_argument('--top-k', nargs='+', type=int, default=[30, 50, 105],
                       help='Top-K feature values to test')
    
    args = parser.parse_args()
    
    # Process correlations (0 -> None)
    correlations = [c if c > 0 else None for c in args.correlations]
    
    # Create config
    config = BenchmarkConfig(
        n_subjects=args.subjects,
        random_seed=args.seed,
        selection_methods=args.methods,
        selection_scopes=args.scopes,
        correlation_thresholds=correlations,
        top_k_values=args.top_k,
        output_dir=args.output
    )
    
    # Calculate total combinations
    n_combinations = (len(config.selection_methods) * 
                     len(config.selection_scopes) * 
                     len(config.correlation_thresholds) * 
                     len(config.top_k_values))
    
    print("\n" + "=" * 60)
    print("FEATURE SELECTION BENCHMARK")
    print("=" * 60)
    print(f"Subjects:      {config.n_subjects}")
    print(f"Methods:       {config.selection_methods}")
    print(f"Scopes:        {config.selection_scopes}")
    print(f"Correlations:  {config.correlation_thresholds}")
    print(f"Top-K values:  {config.top_k_values}")
    print(f"Combinations:  {n_combinations}")
    print(f"Output:        {config.output_dir}")
    print("=" * 60 + "\n")
    
    # Run benchmarks
    runner = BenchmarkRunner(config)
    results = runner.run_all_benchmarks()
    
    # Save results
    runner.save_results()
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    df = pd.DataFrame([asdict(r) for r in results])
    
    print("\n### Method Comparison ###")
    print(df.groupby('method')[['accuracy_mean', 'time_total']].mean().round(3))
    
    print("\n### Scope Comparison ###")
    print(df.groupby('scope')[['accuracy_mean', 'time_total']].mean().round(3))
    
    print(f"\n### Best Configuration ###")
    best = df.loc[df['accuracy_mean'].idxmax()]
    print(f"Method: {best['method']}, Scope: {best['scope']}, "
          f"Corr: {best['correlation']}, K: {best['top_k']}")
    print(f"Accuracy: {best['accuracy_mean']:.3f} ± {best['accuracy_std']:.3f}")
    print(f"Time: {best['time_total']:.1f}s")


if __name__ == '__main__':
    main()
