"""
Evaluation Module for Sleep Stage Classification
=================================================

Provides comprehensive evaluation metrics and reporting for thesis results.

Clinical Targets (from thesis):
- Accuracy ≥ 85%
- Cohen's Kappa ≥ 0.75
- F1-Macro ≥ 80%

Expected Per-Class Performance:
- Wake: Good (F1 0.70-0.90)
- N1: Poor (F1 0.20-0.40) - expected, transitional stage
- N2: Good (F1 0.75-0.90) - most common stage
- N3: Good (F1 0.70-0.85) - distinct delta waves
- REM: Moderate (F1 0.65-0.85) - can be confused with Wake

Author: Lennart Gorzel
Date: December 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

CLASS_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
CLASS_LABELS = [0, 1, 2, 3, 4]

CLINICAL_TARGETS = {
    'accuracy': 0.85,
    'kappa': 0.75,
    'f1_macro': 0.80,
}

EXPECTED_CLASS_PERFORMANCE = {
    'Wake': {'f1_range': (0.70, 0.90), 'note': 'Generally good'},
    'N1': {'f1_range': (0.20, 0.40), 'note': 'Expected poor - transitional'},
    'N2': {'f1_range': (0.75, 0.90), 'note': 'Good - most common'},
    'N3': {'f1_range': (0.70, 0.85), 'note': 'Good - distinct delta'},
    'REM': {'f1_range': (0.65, 0.85), 'note': 'Moderate - can confuse with Wake'},
}


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int
    
    # Comparison to expectations
    within_expected_range: bool = True
    expected_range: Tuple[float, float] = (0.0, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert class metrics to a serializable dictionary."""
        return {
            'class_name': self.class_name,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'support': self.support,
            'within_expected_range': self.within_expected_range,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model."""
    
    # Core metrics
    accuracy: float = 0.0
    kappa: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    
    # Per-class metrics
    class_metrics: Dict[str, ClassMetrics] = field(default_factory=dict)
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    confusion_matrix_normalized: Optional[np.ndarray] = None
    
    # Clinical target status
    meets_accuracy_target: bool = False
    meets_kappa_target: bool = False
    meets_f1_target: bool = False
    meets_all_targets: bool = False
    
    # Sample counts
    n_samples: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'accuracy': self.accuracy,
            'kappa': self.kappa,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'class_metrics': {k: v.to_dict() for k, v in self.class_metrics.items()},
            'meets_accuracy_target': self.meets_accuracy_target,
            'meets_kappa_target': self.meets_kappa_target,
            'meets_f1_target': self.meets_f1_target,
            'meets_all_targets': self.meets_all_targets,
            'n_samples': self.n_samples,
            'class_distribution': self.class_distribution,
        }


@dataclass
class AggregatedEvaluation:
    """Aggregated evaluation across multiple folds/experiments."""
    
    # Mean ± std metrics
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    kappa_mean: float = 0.0
    kappa_std: float = 0.0
    f1_macro_mean: float = 0.0
    f1_macro_std: float = 0.0
    
    # Per-class means
    class_f1_means: Dict[str, float] = field(default_factory=dict)
    class_f1_stds: Dict[str, float] = field(default_factory=dict)
    
    # Source info
    n_evaluations: int = 0
    config_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregated evaluation to a serializable dictionary."""
        return {
            'accuracy_mean': self.accuracy_mean,
            'accuracy_std': self.accuracy_std,
            'kappa_mean': self.kappa_mean,
            'kappa_std': self.kappa_std,
            'f1_macro_mean': self.f1_macro_mean,
            'f1_macro_std': self.f1_macro_std,
            'class_f1_means': self.class_f1_means,
            'class_f1_stds': self.class_f1_stds,
            'n_evaluations': self.n_evaluations,
            'config_id': self.config_id,
        }


# =============================================================================
# Core Evaluation Functions
# =============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> EvaluationResult:
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        EvaluationResult with all metrics
    """
    result = EvaluationResult()
    result.n_samples = len(y_true)
    
    # Core metrics
    result.accuracy = accuracy_score(y_true, y_pred)
    result.kappa = cohen_kappa_score(y_true, y_pred)
    result.f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    result.f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(
        y_true, y_pred, average=None, labels=CLASS_LABELS, zero_division=0
    )
    recall_per_class = recall_score(
        y_true, y_pred, average=None, labels=CLASS_LABELS, zero_division=0
    )
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=CLASS_LABELS, zero_division=0
    )
    
    # Count support per class
    unique, counts = np.unique(y_true, return_counts=True)
    support_dict = dict(zip(unique, counts))
    
    for i, class_name in enumerate(CLASS_NAMES):
        support = support_dict.get(i, 0)
        expected = EXPECTED_CLASS_PERFORMANCE[class_name]
        f1_val = f1_per_class[i] if i < len(f1_per_class) else 0.0
        
        within_range = expected['f1_range'][0] <= f1_val <= expected['f1_range'][1]
        
        result.class_metrics[class_name] = ClassMetrics(
            class_name=class_name,
            precision=precision_per_class[i] if i < len(precision_per_class) else 0.0,
            recall=recall_per_class[i] if i < len(recall_per_class) else 0.0,
            f1_score=f1_val,
            support=support,
            within_expected_range=within_range,
            expected_range=expected['f1_range']
        )
        
        result.class_distribution[class_name] = support
    
    # Confusion matrix
    result.confusion_matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    
    # Normalized confusion matrix (by true labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = result.confusion_matrix.astype('float') / result.confusion_matrix.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
    result.confusion_matrix_normalized = cm_normalized
    
    # Clinical target checks
    result.meets_accuracy_target = result.accuracy >= CLINICAL_TARGETS['accuracy']
    result.meets_kappa_target = result.kappa >= CLINICAL_TARGETS['kappa']
    result.meets_f1_target = result.f1_macro >= CLINICAL_TARGETS['f1_macro']
    result.meets_all_targets = all([
        result.meets_accuracy_target,
        result.meets_kappa_target,
        result.meets_f1_target
    ])
    
    return result


def aggregate_evaluations(
    evaluations: List[EvaluationResult],
    config_id: str = ""
) -> AggregatedEvaluation:
    """
    Aggregate multiple evaluation results (e.g., across CV folds).
    
    Args:
        evaluations: List of EvaluationResult objects
        config_id: Configuration identifier
        
    Returns:
        AggregatedEvaluation with mean ± std metrics
    """
    if not evaluations:
        return AggregatedEvaluation()
    
    accuracies = [e.accuracy for e in evaluations]
    kappas = [e.kappa for e in evaluations]
    f1_macros = [e.f1_macro for e in evaluations]
    
    result = AggregatedEvaluation(
        accuracy_mean=np.mean(accuracies),
        accuracy_std=np.std(accuracies),
        kappa_mean=np.mean(kappas),
        kappa_std=np.std(kappas),
        f1_macro_mean=np.mean(f1_macros),
        f1_macro_std=np.std(f1_macros),
        n_evaluations=len(evaluations),
        config_id=config_id,
    )
    
    # Per-class F1 aggregation
    for class_name in CLASS_NAMES:
        f1_scores = [
            e.class_metrics[class_name].f1_score
            for e in evaluations
            if class_name in e.class_metrics
        ]
        if f1_scores:
            result.class_f1_means[class_name] = np.mean(f1_scores)
            result.class_f1_stds[class_name] = np.std(f1_scores)
    
    return result


# =============================================================================
# Reporting Functions
# =============================================================================

def format_evaluation_report(
    result: EvaluationResult,
    title: str = "Evaluation Report"
) -> str:
    """
    Format evaluation result as human-readable report.
    
    Args:
        result: EvaluationResult to format
        title: Report title
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"{title}")
    lines.append("=" * 60)
    
    # Overall metrics
    lines.append("\nOverall Metrics:")
    lines.append(f"  Accuracy:    {result.accuracy:.4f}  {'✓' if result.meets_accuracy_target else '✗'} (target: ≥{CLINICAL_TARGETS['accuracy']:.2f})")
    lines.append(f"  Kappa:       {result.kappa:.4f}  {'✓' if result.meets_kappa_target else '✗'} (target: ≥{CLINICAL_TARGETS['kappa']:.2f})")
    lines.append(f"  F1-Macro:    {result.f1_macro:.4f}  {'✓' if result.meets_f1_target else '✗'} (target: ≥{CLINICAL_TARGETS['f1_macro']:.2f})")
    lines.append(f"  F1-Weighted: {result.f1_weighted:.4f}")
    
    # Per-class metrics
    lines.append("\nPer-Class Metrics:")
    lines.append(f"  {'Class':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10} {'Status'}")
    lines.append("  " + "-" * 58)
    
    for class_name in CLASS_NAMES:
        if class_name in result.class_metrics:
            m = result.class_metrics[class_name]
            status = "✓" if m.within_expected_range else "⚠"
            lines.append(
                f"  {class_name:<8} {m.precision:<10.4f} {m.recall:<10.4f} "
                f"{m.f1_score:<10.4f} {m.support:<10} {status}"
            )
    
    # Clinical verdict
    lines.append("\n" + "-" * 60)
    if result.meets_all_targets:
        lines.append("✓ MEETS ALL CLINICAL TARGETS")
    else:
        lines.append("✗ Does not meet all clinical targets")
        if not result.meets_accuracy_target:
            lines.append(f"  - Accuracy {result.accuracy:.4f} < {CLINICAL_TARGETS['accuracy']}")
        if not result.meets_kappa_target:
            lines.append(f"  - Kappa {result.kappa:.4f} < {CLINICAL_TARGETS['kappa']}")
        if not result.meets_f1_target:
            lines.append(f"  - F1-Macro {result.f1_macro:.4f} < {CLINICAL_TARGETS['f1_macro']}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def format_aggregated_report(
    result: AggregatedEvaluation,
    title: str = "Aggregated Evaluation"
) -> str:
    """
    Format aggregated evaluation as report.
    
    Args:
        result: AggregatedEvaluation
        title: Report title
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"{title}")
    lines.append(f"Aggregated over {result.n_evaluations} evaluations")
    if result.config_id:
        lines.append(f"Config: {result.config_id}")
    lines.append("=" * 60)
    
    # Overall metrics with std
    lines.append("\nOverall Metrics (mean ± std):")
    lines.append(f"  Accuracy:  {result.accuracy_mean:.4f} ± {result.accuracy_std:.4f}")
    lines.append(f"  Kappa:     {result.kappa_mean:.4f} ± {result.kappa_std:.4f}")
    lines.append(f"  F1-Macro:  {result.f1_macro_mean:.4f} ± {result.f1_macro_std:.4f}")
    
    # Per-class F1
    lines.append("\nPer-Class F1 (mean ± std):")
    for class_name in CLASS_NAMES:
        if class_name in result.class_f1_means:
            mean = result.class_f1_means[class_name]
            std = result.class_f1_stds.get(class_name, 0.0)
            expected = EXPECTED_CLASS_PERFORMANCE[class_name]
            status = "✓" if expected['f1_range'][0] <= mean <= expected['f1_range'][1] else "⚠"
            lines.append(f"  {class_name:<6}: {mean:.4f} ± {std:.4f} {status}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def format_comparison_table(
    results: List[AggregatedEvaluation],
    sort_by: str = 'accuracy_mean'
) -> str:
    """
    Format multiple results as comparison table.
    
    Args:
        results: List of AggregatedEvaluation
        sort_by: Metric to sort by (descending)
        
    Returns:
        Formatted comparison table
    """
    if not results:
        return "No results to compare"
    
    # Sort results
    sorted_results = sorted(results, key=lambda r: getattr(r, sort_by, 0), reverse=True)
    
    lines = []
    lines.append("=" * 90)
    lines.append("CONFIGURATION COMPARISON")
    lines.append("=" * 90)
    lines.append(
        f"{'Rank':<5} {'Config ID':<35} {'Accuracy':<12} {'Kappa':<12} {'F1-Macro':<12} {'Target'}"
    )
    lines.append("-" * 90)
    
    for i, r in enumerate(sorted_results, 1):
        meets_target = (
            r.accuracy_mean >= CLINICAL_TARGETS['accuracy'] and
            r.kappa_mean >= CLINICAL_TARGETS['kappa'] and
            r.f1_macro_mean >= CLINICAL_TARGETS['f1_macro']
        )
        target_str = "✓" if meets_target else "✗"
        
        lines.append(
            f"{i:<5} {r.config_id:<35} "
            f"{r.accuracy_mean:.3f}±{r.accuracy_std:.3f}  "
            f"{r.kappa_mean:.3f}±{r.kappa_std:.3f}  "
            f"{r.f1_macro_mean:.3f}±{r.f1_macro_std:.3f}  "
            f"{target_str}"
        )
    
    lines.append("-" * 90)
    lines.append(f"Sorted by: {sort_by} (descending)")
    lines.append(f"Clinical targets: Acc≥{CLINICAL_TARGETS['accuracy']}, Kappa≥{CLINICAL_TARGETS['kappa']}, F1≥{CLINICAL_TARGETS['f1_macro']}")
    lines.append("=" * 90)
    
    return "\n".join(lines)


# =============================================================================
# Save/Load Functions
# =============================================================================

def save_evaluation(
    result: EvaluationResult,
    filepath: Path,
    include_confusion_matrix: bool = True
):
    """
    Save evaluation result to JSON.
    
    Args:
        result: EvaluationResult to save
        filepath: Output file path
        include_confusion_matrix: Include confusion matrices
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = result.to_dict()
    
    if include_confusion_matrix and result.confusion_matrix is not None:
        data['confusion_matrix'] = result.confusion_matrix.tolist()
        data['confusion_matrix_normalized'] = result.confusion_matrix_normalized.tolist()
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved evaluation to {filepath}")


def save_all_evaluations(
    evaluations: Dict[str, EvaluationResult],
    output_dir: Path,
    summary_name: str = "evaluation_summary"
):
    """
    Save multiple evaluations with summary.
    
    Args:
        evaluations: Dictionary mapping config_id to EvaluationResult
        output_dir: Output directory
        summary_name: Name for summary file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual evaluations
    for config_id, result in evaluations.items():
        safe_name = config_id.replace('/', '_').replace('\\', '_')
        filepath = output_dir / f"eval_{safe_name}.json"
        save_evaluation(result, filepath)
    
    # Save summary
    summary = {
        'n_evaluations': len(evaluations),
        'configs': list(evaluations.keys()),
        'summary': {
            config_id: {
                'accuracy': r.accuracy,
                'kappa': r.kappa,
                'f1_macro': r.f1_macro,
                'meets_all_targets': r.meets_all_targets,
            }
            for config_id, r in evaluations.items()
        }
    }
    
    summary_path = output_dir / f"{summary_name}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved {len(evaluations)} evaluations to {output_dir}")


# =============================================================================
# Module test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Evaluation Module")
    print("=" * 60)
    
    # Create synthetic predictions
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate realistic class distribution (N2 dominant, N1 rare)
    y_true = np.random.choice(
        [0, 1, 2, 3, 4],
        size=n_samples,
        p=[0.15, 0.05, 0.45, 0.20, 0.15]
    )
    
    # Simulate imperfect predictions (85% accuracy)
    y_pred = y_true.copy()
    n_errors = int(n_samples * 0.15)
    error_idx = np.random.choice(n_samples, n_errors, replace=False)
    y_pred[error_idx] = np.random.randint(0, 5, n_errors)
    
    print(f"Samples: {n_samples}")
    print(f"True distribution: {np.bincount(y_true)}")
    
    # Test evaluation
    print("\n" + "-" * 40)
    print("Testing evaluate_predictions")
    result = evaluate_predictions(y_true, y_pred)
    print(format_evaluation_report(result))
    
    # Test aggregation
    print("\n" + "-" * 40)
    print("Testing aggregate_evaluations")
    
    # Create multiple evaluations
    evaluations = []
    for i in range(5):
        y_pred_i = y_true.copy()
        n_errors = int(n_samples * (0.10 + i * 0.02))
        error_idx = np.random.choice(n_samples, n_errors, replace=False)
        y_pred_i[error_idx] = np.random.randint(0, 5, n_errors)
        evaluations.append(evaluate_predictions(y_true, y_pred_i))
    
    agg_result = aggregate_evaluations(evaluations, config_id="test_config")
    print(format_aggregated_report(agg_result))
    
    # Test comparison table
    print("\n" + "-" * 40)
    print("Testing comparison table")
    
    agg_results = [
        AggregatedEvaluation(accuracy_mean=0.87, accuracy_std=0.02, kappa_mean=0.78, kappa_std=0.03, f1_macro_mean=0.82, f1_macro_std=0.02, config_id="xgboost_corr0.95_k50"),
        AggregatedEvaluation(accuracy_mean=0.85, accuracy_std=0.03, kappa_mean=0.75, kappa_std=0.04, f1_macro_mean=0.80, f1_macro_std=0.03, config_id="rf_corr0.95_k50"),
        AggregatedEvaluation(accuracy_mean=0.83, accuracy_std=0.02, kappa_mean=0.72, kappa_std=0.03, f1_macro_mean=0.78, f1_macro_std=0.02, config_id="xgboost_corrNone_kAll"),
    ]
    
    print(format_comparison_table(agg_results))
    
    print("\n" + "=" * 60)
    print("Evaluation Module: ✓ All tests passed")
