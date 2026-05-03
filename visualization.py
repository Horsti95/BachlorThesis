"""
Visualization Module for Sleep Stage Classification
=====================================================

Generates publication-quality figures for thesis results.

Visualizations:
1. Confusion matrices (raw and normalized)
2. Per-subject performance heatmaps
3. Feature importance plots
4. Class distribution charts
5. Performance comparison bar charts
6. Cache performance metrics

All figures are saved at 300 DPI for thesis inclusion.

Author: Lennart Gorzel
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from evaluation import (
    EvaluationResult,
    AggregatedEvaluation,
    CLASS_NAMES,
    CLINICAL_TARGETS,
    EXPECTED_CLASS_PERFORMANCE
)

logger = logging.getLogger(__name__)

# =============================================================================
# Plot Style Configuration
# =============================================================================

# Set consistent style for thesis figures
plt.style.use('seaborn-v0_8-whitegrid')

FIGURE_PARAMS = {
    'dpi': 300,
    'figsize_single': (8, 6),
    'figsize_wide': (12, 6),
    'figsize_square': (8, 8),
    'font_size': 12,
    'title_size': 14,
    'label_size': 11,
}

# Color palette for sleep stages
STAGE_COLORS = {
    'Wake': '#E74C3C',  # Red
    'N1': '#F39C12',    # Orange
    'N2': '#3498DB',    # Blue
    'N3': '#2ECC71',    # Green
    'REM': '#9B59B6',   # Purple
}

# Color palette for models
MODEL_COLORS = {
    'xgboost': '#2ECC71',
    'random_forest': '#3498DB',
    'fnn': '#E74C3C',
}


def setup_plot_style():
    """Configure matplotlib for thesis-quality figures."""
    plt.rcParams.update({
        'font.size': FIGURE_PARAMS['font_size'],
        'axes.titlesize': FIGURE_PARAMS['title_size'],
        'axes.labelsize': FIGURE_PARAMS['label_size'],
        'xtick.labelsize': FIGURE_PARAMS['label_size'],
        'ytick.labelsize': FIGURE_PARAMS['label_size'],
        'legend.fontsize': FIGURE_PARAMS['label_size'],
        'figure.dpi': FIGURE_PARAMS['dpi'],
        'savefig.dpi': FIGURE_PARAMS['dpi'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


# =============================================================================
# Confusion Matrix Visualizations
# =============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: Tuple[int, int] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix (5x5 for sleep stages)
        title: Plot title
        normalize: Normalize by true labels
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display figure
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    figsize = figsize or FIGURE_PARAMS['figsize_square']
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_plot = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_plot)
        fmt = '.2f'
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = 'd'
        vmax = None
    
    # Create heatmap
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0,
        vmax=vmax,
        ax=ax,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_confusion_matrices_comparison(
    results: Dict[str, EvaluationResult],
    figsize: Tuple[int, int] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot confusion matrices for multiple configurations side by side.
    
    Args:
        results: Dictionary mapping config_id to EvaluationResult
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    n_configs = len(results)
    if n_configs == 0:
        logger.warning("No results to plot")
        return None
    
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    figsize = figsize or (5 * n_cols, 5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_configs == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (config_id, result) in enumerate(results.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        
        if result.confusion_matrix is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(config_id)
            continue
        
        # Normalize for comparison
        cm = result.confusion_matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            vmin=0,
            vmax=1,
            ax=ax,
            square=True,
            cbar=False
        )
        
        # Shorten config_id for title
        short_id = config_id[:30] + "..." if len(config_id) > 30 else config_id
        ax.set_title(f"{short_id}\nAcc={result.accuracy:.3f}", fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Hide empty subplots
    for idx in range(n_configs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')
    
    plt.suptitle("Confusion Matrix Comparison (Normalized)", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
        logger.info(f"Saved confusion matrix comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Performance Comparison Visualizations
# =============================================================================

def plot_performance_comparison(
    results: List[AggregatedEvaluation],
    metric: str = 'accuracy_mean',
    title: str = "Model Performance Comparison",
    figsize: Tuple[int, int] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Bar chart comparing performance across configurations.
    
    Args:
        results: List of AggregatedEvaluation
        metric: Metric to plot ('accuracy_mean', 'kappa_mean', 'f1_macro_mean')
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    if not results:
        logger.warning("No results to plot")
        return None
    
    figsize = figsize or FIGURE_PARAMS['figsize_wide']
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    config_ids = [r.config_id for r in results]
    metric_means = [getattr(r, metric, 0) for r in results]
    metric_stds = [getattr(r, metric.replace('_mean', '_std'), 0) for r in results]
    
    # Sort by metric value
    sorted_indices = np.argsort(metric_means)[::-1]
    config_ids = [config_ids[i] for i in sorted_indices]
    metric_means = [metric_means[i] for i in sorted_indices]
    metric_stds = [metric_stds[i] for i in sorted_indices]
    
    # Color bars by model type
    colors = []
    for config_id in config_ids:
        if 'xgboost' in config_id.lower():
            colors.append(MODEL_COLORS['xgboost'])
        elif 'random_forest' in config_id.lower() or 'rf' in config_id.lower():
            colors.append(MODEL_COLORS['random_forest'])
        elif 'fnn' in config_id.lower():
            colors.append(MODEL_COLORS['fnn'])
        else:
            colors.append('#7F8C8D')
    
    # Create bar chart
    x = np.arange(len(config_ids))
    bars = ax.bar(x, metric_means, yerr=metric_stds, capsize=3, color=colors, alpha=0.8)
    
    # Add clinical target line
    target_value = CLINICAL_TARGETS.get(metric.replace('_mean', ''), None)
    if target_value:
        ax.axhline(y=target_value, color='red', linestyle='--', linewidth=2, label=f'Target ({target_value})')
    
    # Labels
    ax.set_xlabel('Configuration')
    ax.set_ylabel(metric.replace('_mean', '').replace('_', ' ').title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=45, ha='right', fontsize=9)
    
    if target_value:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
        logger.info(f"Saved performance comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_multi_metric_comparison(
    results: List[AggregatedEvaluation],
    title: str = "Multi-Metric Performance Comparison",
    figsize: Tuple[int, int] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Grouped bar chart comparing accuracy, kappa, and F1 across configs.
    
    Args:
        results: List of AggregatedEvaluation
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    if not results:
        return None
    
    figsize = figsize or FIGURE_PARAMS['figsize_wide']
    fig, ax = plt.subplots(figsize=figsize)
    
    metrics = ['accuracy_mean', 'kappa_mean', 'f1_macro_mean']
    metric_labels = ['Accuracy', 'Kappa', 'F1-Macro']
    metric_colors = ['#3498DB', '#2ECC71', '#E74C3C']
    
    # Sort results by accuracy
    sorted_results = sorted(results, key=lambda r: r.accuracy_mean, reverse=True)
    config_ids = [r.config_id for r in sorted_results]
    
    x = np.arange(len(config_ids))
    width = 0.25
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        values = [getattr(r, metric, 0) for r in sorted_results]
        stds = [getattr(r, metric.replace('_mean', '_std'), 0) for r in sorted_results]
        ax.bar(x + i * width, values, width, yerr=stds, label=label, color=color, alpha=0.8, capsize=2)
    
    # Add target lines
    for metric, label, color in zip(metrics, metric_labels, metric_colors):
        target = CLINICAL_TARGETS.get(metric.replace('_mean', ''))
        if target:
            ax.axhline(y=target, color=color, linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(config_ids, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
        logger.info(f"Saved multi-metric comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Per-Class Performance Visualizations
# =============================================================================

def plot_per_class_f1(
    result: AggregatedEvaluation,
    title: str = "Per-Class F1 Scores",
    figsize: Tuple[int, int] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Bar chart showing F1 score per sleep stage with expected ranges.
    
    Args:
        result: AggregatedEvaluation with per-class metrics
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    figsize = figsize or FIGURE_PARAMS['figsize_single']
    fig, ax = plt.subplots(figsize=figsize)
    
    # Data
    classes = CLASS_NAMES
    f1_means = [result.class_f1_means.get(c, 0) for c in classes]
    f1_stds = [result.class_f1_stds.get(c, 0) for c in classes]
    colors = [STAGE_COLORS[c] for c in classes]
    
    x = np.arange(len(classes))
    bars = ax.bar(x, f1_means, yerr=f1_stds, capsize=5, color=colors, alpha=0.8)
    
    # Add expected range indicators
    for i, class_name in enumerate(classes):
        expected = EXPECTED_CLASS_PERFORMANCE[class_name]
        low, high = expected['f1_range']
        ax.hlines(y=[low, high], xmin=i-0.3, xmax=i+0.3, colors='gray', linestyles='--', alpha=0.5)
        ax.fill_between([i-0.3, i+0.3], [low, low], [high, high], alpha=0.1, color='gray')
    
    ax.set_xlabel('Sleep Stage')
    ax.set_ylabel('F1 Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.0)
    
    # Add note about expected ranges
    ax.text(0.02, 0.98, 'Gray bands: expected F1 range', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
        logger.info(f"Saved per-class F1 plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_class_distribution(
    labels: np.ndarray,
    title: str = "Sleep Stage Distribution",
    figsize: Tuple[int, int] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Pie/bar chart showing class distribution in dataset.
    
    Args:
        labels: Array of sleep stage labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    figsize = figsize or FIGURE_PARAMS['figsize_wide']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Count labels
    unique, counts = np.unique(labels, return_counts=True)
    stage_counts = {CLASS_NAMES[int(u)]: c for u, c in zip(unique, counts) if int(u) < len(CLASS_NAMES)}
    
    # Pie chart
    colors = [STAGE_COLORS[c] for c in stage_counts.keys()]
    ax1.pie(
        stage_counts.values(),
        labels=stage_counts.keys(),
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )
    ax1.set_title('Class Proportions')
    
    # Bar chart
    x = np.arange(len(stage_counts))
    ax2.bar(x, stage_counts.values(), color=colors, alpha=0.8)
    ax2.set_xlabel('Sleep Stage')
    ax2.set_ylabel('Count')
    ax2.set_title('Class Counts')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stage_counts.keys())
    
    # Add count labels on bars
    for i, (stage, count) in enumerate(stage_counts.items()):
        ax2.text(i, count + 100, str(count), ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
        logger.info(f"Saved class distribution to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Feature Importance Visualization
# =============================================================================

def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_k: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Horizontal bar chart of top feature importances.
    
    Args:
        feature_names: List of feature names
        importance_scores: Array of importance scores
        top_k: Number of top features to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    figsize = figsize or (10, 8)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by importance
    indices = np.argsort(importance_scores)[::-1][:top_k]
    top_features = [feature_names[i] for i in indices]
    top_scores = importance_scores[indices]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_scores, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
        logger.info(f"Saved feature importance to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Cache Performance Visualization
# =============================================================================

def plot_cache_performance(
    cache_times: Dict[str, List[float]],
    title: str = "Cache Performance Analysis",
    figsize: Tuple[int, int] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Box plot comparing cache hit vs miss times.
    
    Args:
        cache_times: Dict with 'hit' and 'miss' lists of times
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    figsize = figsize or FIGURE_PARAMS['figsize_single']
    fig, ax = plt.subplots(figsize=figsize)
    
    data = []
    labels = []
    
    if 'hit' in cache_times and cache_times['hit']:
        data.append(cache_times['hit'])
        labels.append(f"Cache Hit\n(n={len(cache_times['hit'])})")
    
    if 'miss' in cache_times and cache_times['miss']:
        data.append(cache_times['miss'])
        labels.append(f"Cache Miss\n(n={len(cache_times['miss'])})")
    
    if not data:
        ax.text(0.5, 0.5, 'No cache data available', ha='center', va='center')
        return fig
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    colors = ['#2ECC71', '#E74C3C']  # Green for hit, red for miss
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    
    # Add mean annotations
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.annotate(f'μ={mean_val:.2f}s', xy=(i+1, mean_val), xytext=(5, 5),
                   textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_PARAMS['dpi'], bbox_inches='tight')
        logger.info(f"Saved cache performance to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Report Generation
# =============================================================================

def generate_all_figures(
    results: Dict[str, AggregatedEvaluation],
    output_dir: Path,
    labels: Optional[np.ndarray] = None,
    cache_times: Optional[Dict] = None,
    show: bool = False
):
    """
    Generate all thesis figures and save to output directory.
    
    Args:
        results: Dictionary of aggregated results
        output_dir: Directory to save figures
        labels: Dataset labels for distribution plot
        cache_times: Cache timing data
        show: Whether to display figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating figures to {output_dir}")
    
    results_list = list(results.values())
    
    # 1. Performance comparison (accuracy)
    plot_performance_comparison(
        results_list,
        metric='accuracy_mean',
        title='Accuracy Comparison Across Configurations',
        save_path=output_dir / 'accuracy_comparison.png',
        show=show
    )
    
    # 2. Multi-metric comparison
    plot_multi_metric_comparison(
        results_list,
        title='Multi-Metric Performance Comparison',
        save_path=output_dir / 'multi_metric_comparison.png',
        show=show
    )
    
    # 3. Per-class F1 for best model
    if results_list:
        best_result = max(results_list, key=lambda r: r.accuracy_mean)
        plot_per_class_f1(
            best_result,
            title=f'Per-Class F1 Scores ({best_result.config_id})',
            save_path=output_dir / 'per_class_f1_best.png',
            show=show
        )
    
    # 4. Class distribution
    if labels is not None:
        plot_class_distribution(
            labels,
            title='Sleep Stage Distribution in Dataset',
            save_path=output_dir / 'class_distribution.png',
            show=show
        )
    
    # 5. Cache performance
    if cache_times:
        plot_cache_performance(
            cache_times,
            title='Cache Performance: Hit vs Miss Times',
            save_path=output_dir / 'cache_performance.png',
            show=show
        )
    
    logger.info(f"Generated all figures to {output_dir}")


# =============================================================================
# Module test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Visualization Module")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    
    # Synthetic confusion matrix
    cm = np.array([
        [800, 50, 30, 10, 20],   # Wake
        [80, 150, 100, 20, 30],  # N1
        [40, 80, 4000, 200, 50], # N2
        [10, 20, 150, 1800, 30], # N3
        [50, 60, 40, 20, 1200],  # REM
    ])
    
    # Create synthetic results
    from evaluation import EvaluationResult
    
    result1 = EvaluationResult(
        accuracy=0.87,
        kappa=0.79,
        f1_macro=0.72,
        confusion_matrix=cm
    )
    
    agg_results = [
        AggregatedEvaluation(
            accuracy_mean=0.87, accuracy_std=0.02,
            kappa_mean=0.79, kappa_std=0.03,
            f1_macro_mean=0.72, f1_macro_std=0.02,
            class_f1_means={'Wake': 0.85, 'N1': 0.35, 'N2': 0.88, 'N3': 0.80, 'REM': 0.78},
            class_f1_stds={'Wake': 0.03, 'N1': 0.05, 'N2': 0.02, 'N3': 0.03, 'REM': 0.04},
            config_id='xgboost_corr0.95_k50'
        ),
        AggregatedEvaluation(
            accuracy_mean=0.85, accuracy_std=0.03,
            kappa_mean=0.75, kappa_std=0.04,
            f1_macro_mean=0.70, f1_macro_std=0.03,
            class_f1_means={'Wake': 0.82, 'N1': 0.30, 'N2': 0.85, 'N3': 0.77, 'REM': 0.75},
            class_f1_stds={'Wake': 0.04, 'N1': 0.06, 'N2': 0.03, 'N3': 0.04, 'REM': 0.05},
            config_id='rf_corr0.95_k50'
        ),
    ]
    
    # Test directory
    test_dir = Path("./results/test_viz")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Test confusion matrix
    print("\nTesting confusion matrix plot...")
    plot_confusion_matrix(cm, title="Test Confusion Matrix", normalize=True,
                         save_path=test_dir / "test_cm.png", show=False)
    
    # Test performance comparison
    print("Testing performance comparison...")
    plot_performance_comparison(agg_results, metric='accuracy_mean',
                               save_path=test_dir / "test_perf.png", show=False)
    
    # Test per-class F1
    print("Testing per-class F1 plot...")
    plot_per_class_f1(agg_results[0], save_path=test_dir / "test_f1.png", show=False)
    
    # Test class distribution
    print("Testing class distribution...")
    labels = np.random.choice([0, 1, 2, 3, 4], size=10000, p=[0.15, 0.05, 0.45, 0.20, 0.15])
    plot_class_distribution(labels, save_path=test_dir / "test_dist.png", show=False)
    
    print(f"\nTest figures saved to {test_dir}")
    print("\n" + "=" * 60)
    print("Visualization Module: All tests passed")
