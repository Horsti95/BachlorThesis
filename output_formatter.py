"""
Training Output Formatter for Human-Readable Console Output
============================================================

Provides structured, readable output for the training pipeline.
Separates display logic from training logic for clean architecture.

Features:
- Box-style headers for configurations
- LOSO fold progress with clear train/test splits
- Cache status indicators
- Aggregated results tables
- Data leakage prevention confirmation

Author: Lennart Gorzel
Date: December 2025
"""

import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class Verbosity(Enum):
    """Output verbosity levels."""
    QUIET = 0    # Only final summaries
    NORMAL = 1   # Fold progress + summaries (default)
    VERBOSE = 2  # Detailed per-fold output


def _supports_unicode() -> bool:
    """Check if terminal supports Unicode output."""
    # On Windows, check for UTF-8 output
    if sys.platform == 'win32':
        try:
            # Try to encode some unicode chars
            '\u2713'.encode(sys.stdout.encoding or 'cp1252')
            return True
        except (UnicodeEncodeError, LookupError):
            return False
    return True


# Use ASCII fallbacks for Windows terminals that don't support Unicode
USE_UNICODE = _supports_unicode()

# Box drawing characters - with ASCII fallbacks
if USE_UNICODE:
    BOX = {
        'tl': '╔', 'tr': '╗', 'bl': '╚', 'br': '╝',
        'h': '═', 'v': '║',
        'ml': '╠', 'mr': '╣', 'mh': '─',
        'ltee': '├', 'rtee': '┤',
        'cross': '┼',
    }
    LIGHT_BOX = {
        'tl': '┌', 'tr': '┐', 'bl': '└', 'br': '┘',
        'h': '─', 'v': '│',
        'ml': '├', 'mr': '┤',
        'ltee': '├', 'rtee': '┤',
        'cross': '┼',
    }
    ICONS = {
        'check': '✅',
        'cross': '❌',
        'warning': '⚠️',
        'trophy': '🏆',
        'tick': '✓',
        'success': '✅',
        'compute': '⏳',
        'cache': '💾',
    }
else:
    BOX = {
        'tl': '+', 'tr': '+', 'bl': '+', 'br': '+',
        'h': '=', 'v': '|',
        'ml': '+', 'mr': '+', 'mh': '-',
        'ltee': '+', 'rtee': '+',
        'cross': '+',
    }
    LIGHT_BOX = {
        'tl': '+', 'tr': '+', 'bl': '+', 'br': '+',
        'h': '-', 'v': '|',
        'ml': '+', 'mr': '+',
        'ltee': '+', 'rtee': '+',
        'cross': '+',
    }
    ICONS = {
        'check': '[OK]',
        'cross': '[X]',
        'warning': '[!]',
        'trophy': '[#1]',
        'tick': '[v]',
        'success': '[OK]',
        'compute': '[..]',
        'cache': '[C]',
    }


def _make_line(char: str, width: int) -> str:
    """Create a horizontal line of given width."""
    return char * width


def _center_text(text: str, width: int) -> str:
    """Center text within given width."""
    return text.center(width)


def _pad_text(text: str, width: int, align: str = 'left') -> str:
    """Pad text to given width with alignment."""
    if align == 'left':
        return text.ljust(width)
    elif align == 'right':
        return text.rjust(width)
    else:
        return text.center(width)


class TrainingOutputFormatter:
    """
    Formats training output for human readability.
    
    Provides structured console output with:
    - Configuration headers
    - LOSO fold details
    - Feature selection summaries
    - Results tables
    - Cache performance metrics
    """
    
    WIDTH = 80  # Standard terminal width
    
    def __init__(self, verbosity: Verbosity = Verbosity.NORMAL):
        """
        Initialize formatter.
        
        Args:
            verbosity: Output verbosity level
        """
        self.verbosity = verbosity
        self._fold_results: List[Dict] = []
        self._current_config: Optional[str] = None
    
    def set_verbosity(self, verbosity: Verbosity):
        """Set verbosity level."""
        self.verbosity = verbosity
    
    # =========================================================================
    # Major Section Headers
    # =========================================================================
    
    def print_experiment_header(self, experiment_name: str, n_subjects: int,
                                 n_configs: int, n_folds: int):
        """Print main experiment header."""
        w = self.WIDTH
        print()
        print(BOX['tl'] + BOX['h'] * (w - 2) + BOX['tr'])
        print(BOX['v'] + _center_text('TRAINING EXPERIMENT', w - 2) + BOX['v'])
        print(BOX['ml'] + BOX['h'] * (w - 2) + BOX['mr'])
        print(BOX['v'] + f"  Experiment:     {experiment_name}".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + f"  Subjects:       {n_subjects}".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + f"  Configurations: {n_configs}".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + f"  LOSO Folds:     {n_folds} per config".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + f"  Total Runs:     {n_configs * n_folds}".ljust(w - 2) + BOX['v'])
        print(BOX['bl'] + BOX['h'] * (w - 2) + BOX['br'])
        print()
    
    def print_cache_status(self, n_subjects: int, n_cached: int, 
                           cache_path: str, load_time: float = 0.0):
        """Print cache status information."""
        w = self.WIDTH
        hit_rate = (n_cached / n_subjects * 100) if n_subjects > 0 else 0
        status_icon = ICONS['check'] if hit_rate == 100 else ICONS['warning'] if hit_rate > 0 else ICONS['cross']
        
        print("=" * w)
        print("CACHE STATUS")
        print("=" * w)
        print(f"{status_icon} Feature Cache: {'ACTIVE' if n_cached > 0 else 'EMPTY'}")
        print(f"   * Location: {cache_path}")
        print(f"   * Subjects cached: {n_cached}/{n_subjects} ({hit_rate:.0f}%)")
        print(f"   * Features per subject: extracted from cache")
        if load_time > 0:
            print(f"   * Cache load time: {load_time:.2f}s (vs ~25s cold extraction)")
        print()
        print(f"{ICONS['warning']}  Note: Feature SELECTION is computed per-fold (not cached)")
        print("    This is correct behavior to prevent data leakage.")
        print("=" * w)
        print()
    
    def print_loso_setup(self, n_subjects: int, subject_epochs: Dict[str, int],
                         total_epochs: int):
        """Print LOSO cross-validation setup."""
        if self.verbosity == Verbosity.QUIET:
            return
            
        w = self.WIDTH
        print("=" * w)
        print("LOSO CROSS-VALIDATION SETUP")
        print("=" * w)
        print(f"Strategy:           Leave-One-Subject-Out (LOSO)")
        print(f"Total Subjects:     {n_subjects}")
        print(f"Total Folds:        {n_subjects} (one per subject)")
        print(f"Classification:     Per-Epoch (each 30s epoch classified independently)")
        print()
        print("Data Leakage Prevention:")
        print(f"  {ICONS['check']} Feature selection fitted on TRAINING data only")
        print(f"  {ICONS['check']} Test subject completely held out during training")
        print(f"  {ICONS['check']} No epoch from test subject seen during training")
        print()
        
        if self.verbosity == Verbosity.VERBOSE and subject_epochs:
            print("Epoch Distribution:")
            for subj_id, n_epochs in list(subject_epochs.items())[:5]:
                print(f"  Subject_{subj_id}: {n_epochs:,} epochs")
            if len(subject_epochs) > 5:
                print(f"  ... ({len(subject_epochs) - 5} more subjects)")
            print(f"  Total: {total_epochs:,} epochs")
        
        print("=" * w)
        print()
    
    # =========================================================================
    # Configuration Headers
    # =========================================================================
    
    def print_config_header(self, config_name: str, config_idx: int, 
                            total_configs: int, config: Dict[str, Any]):
        """Print configuration header box."""
        self._current_config = config_name
        self._fold_results = []
        
        w = self.WIDTH
        
        # Extract config details
        model_type = config.get('model_type', 'unknown')
        corr_thresh = config.get('feature_selection', {}).get('correlation_threshold', None)
        top_k = config.get('feature_selection', {}).get('top_k_features', None)
        
        corr_str = f"{corr_thresh}" if corr_thresh else "None (keep all)"
        topk_str = f"{top_k}" if top_k else "All"
        
        print()
        print(BOX['tl'] + BOX['h'] * (w - 2) + BOX['tr'])
        print(BOX['v'] + _center_text(f'CONFIGURATION {config_idx}/{total_configs}', w - 2) + BOX['v'])
        print(BOX['v'] + _center_text(config_name, w - 2) + BOX['v'])
        print(BOX['ml'] + BOX['h'] * (w - 2) + BOX['mr'])
        print(BOX['v'] + f"  Model:              {model_type.upper()}".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + f"  Correlation Filter: {corr_str}".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + f"  Top-K Features:     {topk_str}".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + f"  Random State:       42".ljust(w - 2) + BOX['v'])
        print(BOX['bl'] + BOX['h'] * (w - 2) + BOX['br'])
        print()
    
    # =========================================================================
    # Fold Output
    # =========================================================================
    
    def print_fold_start(self, fold_idx: int, n_folds: int, test_subject: str,
                         n_train_subjects: int, n_train_epochs: int, n_test_epochs: int):
        """Print fold start information."""
        if self.verbosity == Verbosity.QUIET:
            return
        
        w = self.WIDTH
        
        if self.verbosity == Verbosity.VERBOSE:
            print(LIGHT_BOX['tl'] + LIGHT_BOX['h'] * (w - 2) + LIGHT_BOX['tr'])
            print(LIGHT_BOX['v'] + f" LOSO FOLD {fold_idx}/{n_folds}".ljust(w - 2) + LIGHT_BOX['v'])
            print(LIGHT_BOX['ml'] + LIGHT_BOX['h'] * (w - 2) + LIGHT_BOX['mr'])
            print(LIGHT_BOX['v'] + f" Test Subject:    Subject_{test_subject}".ljust(w - 2) + LIGHT_BOX['v'])
            print(LIGHT_BOX['v'] + f" Train Subjects:  {n_train_subjects} subjects (excluding Subject_{test_subject})".ljust(w - 2) + LIGHT_BOX['v'])
            print(LIGHT_BOX['v'] + f" Train Epochs:    {n_train_epochs:,}".ljust(w - 2) + LIGHT_BOX['v'])
            print(LIGHT_BOX['v'] + f" Test Epochs:     {n_test_epochs:,}".ljust(w - 2) + LIGHT_BOX['v'])
            print(LIGHT_BOX['ml'] + LIGHT_BOX['h'] * (w - 2) + LIGHT_BOX['mr'])
        else:
            # Normal mode: compact output
            print(f"  Fold {fold_idx:2d}/{n_folds}: Test=Subject_{test_subject} "
                  f"(train: {n_train_epochs:,} epochs, test: {n_test_epochs:,} epochs)", end="")
    
    def print_feature_selection(self, n_input: int, n_after_corr: int, n_final: int,
                                 corr_removed: int):
        """Print feature selection summary."""
        if self.verbosity != Verbosity.VERBOSE:
            return
        
        w = self.WIDTH
        print(LIGHT_BOX['v'] + " Feature Selection (fitted on TRAIN only - no data leakage):".ljust(w - 2) + LIGHT_BOX['v'])
        print(LIGHT_BOX['v'] + f"   * Input features:        {n_input}".ljust(w - 2) + LIGHT_BOX['v'])
        
        if n_after_corr != n_input:
            print(LIGHT_BOX['v'] + f"   * After correlation:     {n_after_corr}  (removed {corr_removed} highly correlated)".ljust(w - 2) + LIGHT_BOX['v'])
        else:
            print(LIGHT_BOX['v'] + f"   * After correlation:     {n_after_corr}  (no filtering)".ljust(w - 2) + LIGHT_BOX['v'])
        
        print(LIGHT_BOX['v'] + f"   * After top-K:           {n_final}  (selected by mutual information)".ljust(w - 2) + LIGHT_BOX['v'])
        print(LIGHT_BOX['ml'] + LIGHT_BOX['h'] * (w - 2) + LIGHT_BOX['mr'])
    
    def print_feature_selection_warning(self, requested_k: int, available: int, used_k: int):
        """Print warning when requested k > available features.
        
        Args:
            requested_k: Originally requested top-K features
            available: Number of features available after correlation filter
            used_k: Actual number of features selected (min of above)
        """
        w = self.WIDTH
        print()
        print(f"{ICONS['warning']} FEATURE SELECTION WARNING:")
        print(f"   Requested k={requested_k} features, but only {available} available after correlation filter.")
        print(f"   Using all {used_k} features instead.")
        print()
    
    def print_training_progress(self, model_type: str, n_samples: int, n_features: int,
                                 cache_status: Optional[str] = None):
        """Print training progress with optional cache status."""
        if self.verbosity != Verbosity.VERBOSE:
            return
        
        w = self.WIDTH
        
        # Build status string
        if cache_status == "HIT":
            status_str = f" {ICONS['success']} CACHE HIT - Loading cached model"
            print(LIGHT_BOX['v'] + status_str.ljust(w - 2) + LIGHT_BOX['v'])
        elif cache_status == "MISS":
            status_str = f" {ICONS['compute']} CACHE MISS - Training: {model_type} on {n_samples:,} samples x {n_features} features"
            print(LIGHT_BOX['v'] + status_str.ljust(w - 2) + LIGHT_BOX['v'])
        else:
            print(LIGHT_BOX['v'] + f" Training: {model_type} on {n_samples:,} samples x {n_features} features".ljust(w - 2) + LIGHT_BOX['v'])
    
    def print_fold_result(self, fold_idx: int, test_subject: str,
                          accuracy: float, kappa: float, f1: float, 
                          elapsed: float, n_features: int = 0):
        """Print single fold results."""
        # Store for summary
        self._fold_results.append({
            'fold': fold_idx,
            'subject': test_subject,
            'accuracy': accuracy,
            'kappa': kappa,
            'f1': f1,
            'time': elapsed,
            'n_features': n_features
        })
        
        if self.verbosity == Verbosity.QUIET:
            return
        
        w = self.WIDTH
        
        if self.verbosity == Verbosity.VERBOSE:
            print(LIGHT_BOX['v'] + f" Elapsed:  {elapsed:.1f}s".ljust(w - 2) + LIGHT_BOX['v'])
            print(LIGHT_BOX['ml'] + LIGHT_BOX['h'] * (w - 2) + LIGHT_BOX['mr'])
            print(LIGHT_BOX['v'] + " FOLD RESULTS:".ljust(w - 2) + LIGHT_BOX['v'])
            print(LIGHT_BOX['v'] + f"   Accuracy:  {accuracy:.3f}".ljust(w - 2) + LIGHT_BOX['v'])
            print(LIGHT_BOX['v'] + f"   Kappa:     {kappa:.3f}".ljust(w - 2) + LIGHT_BOX['v'])
            print(LIGHT_BOX['v'] + f"   F1-Macro:  {f1:.3f}".ljust(w - 2) + LIGHT_BOX['v'])
            print(LIGHT_BOX['bl'] + LIGHT_BOX['h'] * (w - 2) + LIGHT_BOX['br'])
            print()
        else:
            # Normal mode: inline result
            print(f" -> acc={accuracy:.3f}, k={kappa:.3f}, F1={f1:.3f} ({elapsed:.1f}s)")
    
    # =========================================================================
    # Configuration Summary
    # =========================================================================
    
    def print_config_summary(self, config_name: str, 
                             accuracy_mean: float, accuracy_std: float,
                             kappa_mean: float, kappa_std: float,
                             f1_mean: float, f1_std: float,
                             total_time: float,
                             cache_hits: int = 0, cache_total: int = 0):
        """Print final configuration summary with all folds."""
        w = self.WIDTH
        
        print()
        print("=" * w)
        print(f"CONFIGURATION COMPLETE: {config_name}")
        print("=" * w)
        print()
        
        # Aggregate metrics table
        print(f"AGGREGATE RESULTS ({len(self._fold_results)} folds):")
        print(LIGHT_BOX['tl'] + LIGHT_BOX['h'] * 12 + LIGHT_BOX['h'] + 
              LIGHT_BOX['h'] * 10 + LIGHT_BOX['h'] + 
              LIGHT_BOX['h'] * 10 + LIGHT_BOX['h'] +
              LIGHT_BOX['h'] * 12 + LIGHT_BOX['tr'])
        print(f"{LIGHT_BOX['v']} {'Metric':<10} {LIGHT_BOX['v']} {'Mean':>8} {LIGHT_BOX['v']} {'Std':>8} {LIGHT_BOX['v']} {'Range':>10} {LIGHT_BOX['v']}")
        print(LIGHT_BOX['ml'] + LIGHT_BOX['h'] * 12 + LIGHT_BOX['cross'] + 
              LIGHT_BOX['h'] * 10 + LIGHT_BOX['cross'] + 
              LIGHT_BOX['h'] * 10 + LIGHT_BOX['cross'] +
              LIGHT_BOX['h'] * 12 + LIGHT_BOX['mr'])
        
        if self._fold_results:
            accs = [r['accuracy'] for r in self._fold_results]
            kappas = [r['kappa'] for r in self._fold_results]
            f1s = [r['f1'] for r in self._fold_results]
            
            print(f"{LIGHT_BOX['v']} {'Accuracy':<10} {LIGHT_BOX['v']} {accuracy_mean:>8.3f} {LIGHT_BOX['v']} +/-{accuracy_std:>6.3f} {LIGHT_BOX['v']} {min(accs):.2f}-{max(accs):.2f} {LIGHT_BOX['v']}")
            print(f"{LIGHT_BOX['v']} {'Kappa':<10} {LIGHT_BOX['v']} {kappa_mean:>8.3f} {LIGHT_BOX['v']} +/-{kappa_std:>6.3f} {LIGHT_BOX['v']} {min(kappas):.2f}-{max(kappas):.2f} {LIGHT_BOX['v']}")
            print(f"{LIGHT_BOX['v']} {'F1-Macro':<10} {LIGHT_BOX['v']} {f1_mean:>8.3f} {LIGHT_BOX['v']} +/-{f1_std:>6.3f} {LIGHT_BOX['v']} {min(f1s):.2f}-{max(f1s):.2f} {LIGHT_BOX['v']}")
        
        print(LIGHT_BOX['bl'] + LIGHT_BOX['h'] * 12 + LIGHT_BOX['h'] + 
              LIGHT_BOX['h'] * 10 + LIGHT_BOX['h'] + 
              LIGHT_BOX['h'] * 10 + LIGHT_BOX['h'] +
              LIGHT_BOX['h'] * 12 + LIGHT_BOX['br'])
        print()
        
        # Per-fold breakdown (verbose mode)
        if self.verbosity == Verbosity.VERBOSE and self._fold_results:
            print("PER-FOLD BREAKDOWN:")
            print(LIGHT_BOX['tl'] + LIGHT_BOX['h'] * 6 + LIGHT_BOX['h'] + 
                  LIGHT_BOX['h'] * 13 + LIGHT_BOX['h'] + 
                  LIGHT_BOX['h'] * 10 + LIGHT_BOX['h'] +
                  LIGHT_BOX['h'] * 8 + LIGHT_BOX['h'] +
                  LIGHT_BOX['h'] * 7 + LIGHT_BOX['h'] +
                  LIGHT_BOX['h'] * 10 + LIGHT_BOX['tr'])
            print(f"{LIGHT_BOX['v']} {'Fold':>4} {LIGHT_BOX['v']} {'Test Subject':^11} {LIGHT_BOX['v']} {'Accuracy':>8} {LIGHT_BOX['v']} {'Kappa':>6} {LIGHT_BOX['v']} {'F1':>5} {LIGHT_BOX['v']} {'Time (s)':>8} {LIGHT_BOX['v']}")
            print(LIGHT_BOX['ml'] + LIGHT_BOX['h'] * 6 + LIGHT_BOX['cross'] + 
                  LIGHT_BOX['h'] * 13 + LIGHT_BOX['cross'] + 
                  LIGHT_BOX['h'] * 10 + LIGHT_BOX['cross'] +
                  LIGHT_BOX['h'] * 8 + LIGHT_BOX['cross'] +
                  LIGHT_BOX['h'] * 7 + LIGHT_BOX['cross'] +
                  LIGHT_BOX['h'] * 10 + LIGHT_BOX['mr'])
            
            for r in self._fold_results:
                subj_str = f"Subject_{r['subject']}"
                print(f"{LIGHT_BOX['v']} {r['fold']:>4} {LIGHT_BOX['v']} {subj_str:^11} {LIGHT_BOX['v']} {r['accuracy']:>8.3f} {LIGHT_BOX['v']} {r['kappa']:>6.3f} {LIGHT_BOX['v']} {r['f1']:>5.3f} {LIGHT_BOX['v']} {r['time']:>8.1f} {LIGHT_BOX['v']}")
            
            # Average row
            avg_time = sum(r['time'] for r in self._fold_results) / len(self._fold_results)
            print(LIGHT_BOX['ml'] + LIGHT_BOX['h'] * 6 + LIGHT_BOX['cross'] + 
                  LIGHT_BOX['h'] * 13 + LIGHT_BOX['cross'] + 
                  LIGHT_BOX['h'] * 10 + LIGHT_BOX['cross'] +
                  LIGHT_BOX['h'] * 8 + LIGHT_BOX['cross'] +
                  LIGHT_BOX['h'] * 7 + LIGHT_BOX['cross'] +
                  LIGHT_BOX['h'] * 10 + LIGHT_BOX['mr'])
            print(f"{LIGHT_BOX['v']} {'AVG':>4} {LIGHT_BOX['v']} {'-':^11} {LIGHT_BOX['v']} {accuracy_mean:>8.3f} {LIGHT_BOX['v']} {kappa_mean:>6.3f} {LIGHT_BOX['v']} {f1_mean:>5.3f} {LIGHT_BOX['v']} {avg_time:>8.1f} {LIGHT_BOX['v']}")
            
            print(LIGHT_BOX['bl'] + LIGHT_BOX['h'] * 6 + LIGHT_BOX['h'] + 
                  LIGHT_BOX['h'] * 13 + LIGHT_BOX['h'] + 
                  LIGHT_BOX['h'] * 10 + LIGHT_BOX['h'] +
                  LIGHT_BOX['h'] * 8 + LIGHT_BOX['h'] +
                  LIGHT_BOX['h'] * 7 + LIGHT_BOX['h'] +
                  LIGHT_BOX['h'] * 10 + LIGHT_BOX['br'])
            print()
        
        # Cache performance
        if cache_total > 0:
            time_saved = cache_hits * 25  # ~25s per subject cold extraction
            print("CACHE PERFORMANCE THIS CONFIG:")
            print(f"  * Feature loads: {cache_hits} (all from cache)")
            print(f"  * Cache hit rate: {cache_hits/cache_total*100:.0f}%")
            print(f"  * Time saved: ~{time_saved//60}min {time_saved%60}s (vs cold extraction)")
            print()
        
        print(f"Total config time: {total_time/60:.0f}m {total_time%60:.0f}s")
        print("=" * w)
        print()
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    
    def print_final_results_table(self, results: List[Dict], 
                                   sort_by: str = 'accuracy_mean'):
        """Print final results comparison table."""
        w = self.WIDTH
        
        # Sort results
        sorted_results = sorted(results, key=lambda x: x.get(sort_by, 0), reverse=True)
        
        print()
        print(BOX['tl'] + BOX['h'] * (w - 2) + BOX['tr'])
        print(BOX['v'] + _center_text('FINAL RESULTS SUMMARY', w - 2) + BOX['v'])
        print(BOX['ml'] + BOX['h'] * (w - 2) + BOX['mr'])
        
        # Header
        header = f"{'Rank':<5} {'Configuration':<32} {'Accuracy':>10} {'Kappa':>8} {'F1':>8}"
        print(BOX['v'] + f" {header}".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + " " + "-" * 70 + " " * (w - 73) + BOX['v'])
        
        # Results rows
        for i, r in enumerate(sorted_results, 1):
            config_id = r.get('config_id', 'unknown')[:30]
            acc = r.get('accuracy_mean', 0)
            acc_std = r.get('accuracy_std', 0)
            kappa = r.get('kappa_mean', 0)
            f1 = r.get('f1_macro_mean', 0)
            
            # Highlight best
            prefix = ICONS['trophy'] if i == 1 else f"{i:2d}"
            
            row = f"{prefix:<5} {config_id:<32} {acc:.3f}+/-{acc_std:.3f} {kappa:>8.3f} {f1:>8.3f}"
            print(BOX['v'] + f" {row}".ljust(w - 2) + BOX['v'])
        
        print(BOX['ml'] + BOX['h'] * (w - 2) + BOX['mr'])
        print(BOX['v'] + " Clinical targets: Accuracy>=0.85, Kappa>=0.75, F1>=0.80".ljust(w - 2) + BOX['v'])
        print(BOX['bl'] + BOX['h'] * (w - 2) + BOX['br'])
        print()
    
    def print_best_result(self, config_id: str, accuracy: float, accuracy_std: float,
                          kappa: float, kappa_std: float, f1: float, f1_std: float,
                          meets_targets: bool):
        """Print best configuration highlight."""
        w = self.WIDTH
        
        print()
        print(f"{ICONS['trophy']} BEST CONFIGURATION: " + config_id)
        print("-" * 40)
        print(f"   Accuracy:  {accuracy:.4f} +/- {accuracy_std:.4f}")
        print(f"   Kappa:     {kappa:.4f} +/- {kappa_std:.4f}")
        print(f"   F1-Macro:  {f1:.4f} +/- {f1_std:.4f}")
        print()
        
        if meets_targets:
            print(f"{ICONS['check']} MEETS ALL CLINICAL TARGETS")
        else:
            print(f"{ICONS['cross']} Does not meet all clinical targets")
        print()
    
    def print_cache_summary(self, hit_rate: float, cold_time: float, warm_time: float,
                            time_saved: float, speedup: float):
        """Print final cache performance summary."""
        w = self.WIDTH
        
        print(BOX['tl'] + BOX['h'] * (w - 2) + BOX['tr'])
        print(BOX['v'] + _center_text('CACHE PERFORMANCE SUMMARY (THESIS FOCUS)', w - 2) + BOX['v'])
        print(BOX['ml'] + BOX['h'] * (w - 2) + BOX['mr'])
        print(BOX['v'] + f"  Cache Hit Rate:    {hit_rate*100:.1f}%".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + f"  Cold Start Time:   {cold_time/60:.1f} minutes".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + f"  Warm Start Time:   {warm_time/60:.1f} minutes".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + f"  Speedup Factor:    {speedup:.0f}x".ljust(w - 2) + BOX['v'])
        print(BOX['v'] + f"  Time Saved:        {time_saved/60:.1f} minutes".ljust(w - 2) + BOX['v'])
        print(BOX['bl'] + BOX['h'] * (w - 2) + BOX['br'])
        print()
    
    # =========================================================================
    # Progress Indicators
    # =========================================================================
    
    def print_step(self, step_num: int, total_steps: int, description: str):
        """Print pipeline step indicator."""
        print(f"\n[{step_num}/{total_steps}] {description}...")
    
    def print_substep(self, text: str, icon: str = "tick"):
        """Print substep completion."""
        icon_char = ICONS.get(icon, ICONS['tick'])
        print(f"  {icon_char} {text}")
    
    def print_warning(self, text: str):
        """Print warning message."""
        print(f"  {ICONS['warning']} {text}")
    
    def print_error(self, text: str):
        """Print error message."""
        print(f"  {ICONS['cross']} {text}")


# =============================================================================
# Convenience Functions
# =============================================================================

# Global formatter instance
_formatter: Optional[TrainingOutputFormatter] = None


def get_formatter() -> TrainingOutputFormatter:
    """Get or create the global formatter instance."""
    global _formatter
    if _formatter is None:
        _formatter = TrainingOutputFormatter()
    return _formatter


def set_verbosity(verbosity: Verbosity):
    """Set global formatter verbosity."""
    get_formatter().set_verbosity(verbosity)


def set_verbosity_from_args(verbose: bool = False, quiet: bool = False):
    """Set verbosity from CLI arguments."""
    if quiet:
        set_verbosity(Verbosity.QUIET)
    elif verbose:
        set_verbosity(Verbosity.VERBOSE)
    else:
        set_verbosity(Verbosity.NORMAL)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test the formatter
    fmt = TrainingOutputFormatter(Verbosity.VERBOSE)
    
    # Test experiment header
    fmt.print_experiment_header("test_experiment", 10, 18, 10)
    
    # Test cache status
    fmt.print_cache_status(10, 10, "results/features_cache_global/", 0.12)
    
    # Test LOSO setup
    subject_epochs = {str(i): 800 + i * 50 for i in range(1, 11)}
    fmt.print_loso_setup(10, subject_epochs, sum(subject_epochs.values()))
    
    # Test config header
    config = {
        'model_type': 'random_forest',
        'feature_selection': {
            'correlation_threshold': 0.9,
            'top_k_features': 50
        }
    }
    fmt.print_config_header("random_forest_corr0.9_k50", 14, 18, config)
    
    # Test fold output
    fmt.print_fold_start(1, 10, "001", 9, 8375, 892)
    fmt.print_feature_selection(149, 71, 50, 78)
    fmt.print_training_progress("Random Forest", 8375, 50)
    fmt.print_fold_result(1, "001", 0.807, 0.710, 0.683, 5.6, 50)
    
    # Simulate more folds
    for i in range(2, 11):
        fmt._fold_results.append({
            'fold': i, 'subject': f'{i:03d}',
            'accuracy': 0.7 + (i % 3) * 0.05,
            'kappa': 0.5 + (i % 4) * 0.05,
            'f1': 0.55 + (i % 3) * 0.05,
            'time': 5 + i * 0.3,
            'n_features': 50
        })
    
    # Test config summary
    fmt.print_config_summary("random_forest_corr0.9_k50", 
                             0.713, 0.071, 0.450, 0.142, 0.456, 0.098,
                             70.5, 10, 10)
    
    # Test final summary
    results = [
        {'config_id': 'xgboost_corrNone_k105', 'accuracy_mean': 0.771, 'accuracy_std': 0.055, 'kappa_mean': 0.598, 'f1_macro_mean': 0.580},
        {'config_id': 'random_forest_corr0.9_k50', 'accuracy_mean': 0.713, 'accuracy_std': 0.071, 'kappa_mean': 0.450, 'f1_macro_mean': 0.456},
    ]
    fmt.print_final_results_table(results)
    
    fmt.print_best_result('xgboost_corrNone_k105', 0.771, 0.055, 0.598, 0.129, 0.580, 0.088, False)
    
    fmt.print_cache_summary(1.0, 3200, 1200, 2000, 2.7)
    
    print(f"\n{ICONS['check']} Output formatter test complete!")
