"""
Cache Performance Leaderboard
=============================

Tracks cache performance metrics across all experiments for thesis validation.

Research Focus:
- Cache hit rates
- Time savings (cold vs cached)
- Loading times by stage (RAM, Disk, Cold)
- Fingerprint validity

Author: Lennart Gorzel
Date: December 2025
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import time

logger = logging.getLogger(__name__)


@dataclass
class CacheEvent:
    """Single cache access event."""
    timestamp: str
    subject_id: str
    event_type: str  # 'hit_ram', 'hit_disk', 'miss_cold'
    load_time_seconds: float
    fingerprint_hash: str
    features_loaded: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentRun:
    """Metrics from a single experiment run."""
    experiment_name: str
    timestamp: str
    
    # Cache metrics
    total_subjects: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Timing metrics (seconds)
    total_time: float = 0.0
    time_loading: float = 0.0
    time_preprocessing: float = 0.0
    time_feature_extraction: float = 0.0
    time_from_cache: float = 0.0
    
    # Estimated time saved
    estimated_cold_time: float = 0.0  # What it would have taken without cache
    actual_time: float = 0.0
    time_saved: float = 0.0
    
    # Configuration
    config_summary: Dict = field(default_factory=dict)
    
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['cache_hit_rate_percent'] = self.cache_hit_rate()
        return d


@dataclass 
class CacheTimingTargets:
    """
    Expected cache timing targets (ESTIMATES - to be validated).
    
    From thesis design:
    - RAM Hit ≈ 0.1s
    - Disk Hit ≈ 1s
    - Cold Miss ≈ 15-30s per subject
    
    These are educated guesses requiring experimental validation.
    """
    ram_hit_target: float = 0.1  # seconds (ESTIMATE)
    disk_hit_target: float = 1.0  # seconds (ESTIMATE)
    cold_miss_target: float = 20.0  # seconds avg (ESTIMATE: 15-30s)
    
    # Actual measured values (populated during experiments)
    ram_hit_actual: List[float] = field(default_factory=list)
    disk_hit_actual: List[float] = field(default_factory=list)
    cold_miss_actual: List[float] = field(default_factory=list)
    
    def add_measurement(self, event_type: str, time_seconds: float):
        """Add a timing measurement."""
        if event_type == 'hit_ram':
            self.ram_hit_actual.append(time_seconds)
        elif event_type == 'hit_disk':
            self.disk_hit_actual.append(time_seconds)
        elif event_type == 'miss_cold':
            self.cold_miss_actual.append(time_seconds)
    
    def get_averages(self) -> Dict[str, Optional[float]]:
        """Get average actual timings."""
        def avg(lst):
            return sum(lst) / len(lst) if lst else None
        
        return {
            'ram_hit_avg': avg(self.ram_hit_actual),
            'disk_hit_avg': avg(self.disk_hit_actual),
            'cold_miss_avg': avg(self.cold_miss_actual)
        }
    
    def format_comparison(self) -> str:
        """Format target vs actual comparison."""
        avgs = self.get_averages()
        lines = [
            "Cache Timing Analysis (Target vs Actual):",
            "=" * 50,
            f"{'Event Type':<15} {'Target':<12} {'Actual':<12} {'Status':<10}",
            "-" * 50,
        ]
        
        for event, target, actual_key in [
            ('RAM Hit', self.ram_hit_target, 'ram_hit_avg'),
            ('Disk Hit', self.disk_hit_target, 'disk_hit_avg'),
            ('Cold Miss', self.cold_miss_target, 'cold_miss_avg'),
        ]:
            actual = avgs[actual_key]
            if actual is not None:
                status = "[OK] PASS" if actual <= target * 1.5 else "[!] SLOW"
                lines.append(f"{event:<15} {target:<12.2f}s {actual:<12.2f}s {status:<10}")
            else:
                lines.append(f"{event:<15} {target:<12.2f}s {'N/A':<12} {'No data':<10}")
        
        lines.append("-" * 50)
        lines.append("Note: Targets are ESTIMATES requiring validation (Chapter 5)")
        
        return "\n".join(lines)


class CacheLeaderboard:
    """
    Tracks and persists cache performance across all experiments.
    
    This is central to the thesis goal of validating:
    - SQ1: Efficiency (time saved)
    - SQ2: Reproducibility (cache hits with same fingerprint)
    - SQ3: Scalability (performance across dataset sizes)
    """
    
    def __init__(self, leaderboard_path: str = "results/cache_leaderboard.json"):
        self.leaderboard_path = Path(leaderboard_path)
        self.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self.runs: List[ExperimentRun] = []
        self.timing_targets = CacheTimingTargets()
        self.cache_events: List[CacheEvent] = []
        
        # Aggregated stats
        self.total_time_saved: float = 0.0
        self.total_cache_hits: int = 0
        self.total_cache_misses: int = 0
        
        self._load()
        logger.info(f"CacheLeaderboard initialized: {self.leaderboard_path}")
    
    def _load(self):
        """Load leaderboard from disk."""
        if self.leaderboard_path.exists():
            try:
                with open(self.leaderboard_path, 'r') as f:
                    data = json.load(f)
                
                # Restore runs
                for run_dict in data.get('runs', []):
                    run = ExperimentRun(**{k: v for k, v in run_dict.items() 
                                          if k != 'cache_hit_rate_percent'})
                    self.runs.append(run)
                
                # Restore aggregates
                self.total_time_saved = data.get('total_time_saved', 0.0)
                self.total_cache_hits = data.get('total_cache_hits', 0)
                self.total_cache_misses = data.get('total_cache_misses', 0)
                
                # Restore timing measurements
                timing_data = data.get('timing_measurements', {})
                self.timing_targets.ram_hit_actual = timing_data.get('ram_hit', [])
                self.timing_targets.disk_hit_actual = timing_data.get('disk_hit', [])
                self.timing_targets.cold_miss_actual = timing_data.get('cold_miss', [])
                
                logger.info(f"Loaded {len(self.runs)} previous experiment runs")
                
            except Exception as e:
                logger.warning(f"Could not load leaderboard: {e}")
    
    def _save(self):
        """Save leaderboard to disk."""
        data = {
            'last_updated': datetime.now().isoformat(),
            'total_runs': len(self.runs),
            'total_time_saved': self.total_time_saved,
            'total_cache_hits': self.total_cache_hits,
            'total_cache_misses': self.total_cache_misses,
            'overall_hit_rate_percent': self.overall_hit_rate(),
            'timing_measurements': {
                'ram_hit': self.timing_targets.ram_hit_actual,
                'disk_hit': self.timing_targets.disk_hit_actual,
                'cold_miss': self.timing_targets.cold_miss_actual,
            },
            'timing_targets': {
                'ram_hit_target_s': self.timing_targets.ram_hit_target,
                'disk_hit_target_s': self.timing_targets.disk_hit_target,
                'cold_miss_target_s': self.timing_targets.cold_miss_target,
                'note': 'ESTIMATES - require experimental validation'
            },
            'runs': [run.to_dict() for run in self.runs]
        }
        
        with open(self.leaderboard_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Leaderboard saved to {self.leaderboard_path}")
    
    def overall_hit_rate(self) -> float:
        """Calculate overall cache hit rate across all experiments."""
        total = self.total_cache_hits + self.total_cache_misses
        if total == 0:
            return 0.0
        return (self.total_cache_hits / total) * 100
    
    def record_cache_event(
        self,
        subject_id: str,
        event_type: str,
        load_time: float,
        fingerprint_hash: str = "",
        features_loaded: int = 0
    ):
        """
        Record a single cache access event.
        
        Args:
            subject_id: Subject identifier
            event_type: 'hit_ram', 'hit_disk', or 'miss_cold'
            load_time: Time in seconds
            fingerprint_hash: Cache fingerprint
            features_loaded: Number of features loaded
        """
        event = CacheEvent(
            timestamp=datetime.now().isoformat(),
            subject_id=subject_id,
            event_type=event_type,
            load_time_seconds=load_time,
            fingerprint_hash=fingerprint_hash,
            features_loaded=features_loaded
        )
        self.cache_events.append(event)
        self.timing_targets.add_measurement(event_type, load_time)
        
        # Update counters
        if event_type.startswith('hit'):
            self.total_cache_hits += 1
        else:
            self.total_cache_misses += 1
    
    def record_experiment_run(self, run: ExperimentRun):
        """Record a complete experiment run."""
        self.runs.append(run)
        self.total_time_saved += run.time_saved
        self._save()
    
    def start_run(self, experiment_name: str, config_summary: Dict = None) -> ExperimentRun:
        """Start tracking a new experiment run."""
        run = ExperimentRun(
            experiment_name=experiment_name,
            timestamp=datetime.now().isoformat(),
            config_summary=config_summary or {}
        )
        return run
    
    def finalize_run(
        self,
        run: ExperimentRun,
        cache_hits: int,
        cache_misses: int,
        total_time: float,
        estimated_cold_time: Optional[float] = None
    ):
        """Finalize and save an experiment run.
        
        Time saved calculation:
        - Each cache hit saves ~20s of cold feature extraction (ESTIMATE)
        - time_saved = cache_hits × cold_miss_target
        """
        run.cache_hits = cache_hits
        run.cache_misses = cache_misses
        run.actual_time = total_time
        
        # Calculate time saved: each cache hit saves a cold extraction (~20s)
        # This is an ESTIMATE - actual cold time varies per subject (15-30s)
        run.time_saved = cache_hits * self.timing_targets.cold_miss_target
        
        # Also store estimated cold time for reference
        if estimated_cold_time is not None:
            run.estimated_cold_time = estimated_cold_time
        else:
            # Estimate: all subjects would have been cold
            total_subjects = cache_hits + cache_misses
            run.estimated_cold_time = total_subjects * self.timing_targets.cold_miss_target
        
        # Update global totals
        self.total_cache_hits += cache_hits
        self.total_cache_misses += cache_misses
        
        self.record_experiment_run(run)
    
    def get_summary(self) -> str:
        """Get a formatted summary of cache performance."""
        lines = [
            "",
            "=" * 60,
            "CACHE PERFORMANCE LEADERBOARD",
            "=" * 60,
            f"Total Experiment Runs: {len(self.runs)}",
            f"Overall Cache Hit Rate: {self.overall_hit_rate():.1f}%",
            f"Total Time Saved: {self.total_time_saved:.1f}s ({self.total_time_saved/60:.1f} min)",
            f"Total Cache Hits: {self.total_cache_hits}",
            f"Total Cache Misses: {self.total_cache_misses}",
            "",
        ]
        
        # Add timing analysis
        lines.append(self.timing_targets.format_comparison())
        
        # Recent runs
        if self.runs:
            lines.append("\nRecent Runs (last 5):")
            lines.append("-" * 60)
            for run in self.runs[-5:]:
                lines.append(
                    f"  {run.experiment_name[:30]:<30} "
                    f"Hit: {run.cache_hit_rate():.0f}% "
                    f"Saved: {run.time_saved:.1f}s"
                )
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def print_summary(self):
        """Print the leaderboard summary to console."""
        print(self.get_summary())


# Time estimation functions
def estimate_experiment_time(
    n_subjects: int,
    n_models: int,
    n_feature_configs: int,
    cache_hit_rate: float = 0.0,
    cached_time_per_subject: float = 1.0,
    cold_time_per_subject: float = 20.0
) -> Dict[str, float]:
    """
    Estimate experiment time based on configuration.
    
    Args:
        n_subjects: Number of subjects
        n_models: Number of models to train
        n_feature_configs: Number of feature configurations
        cache_hit_rate: Expected cache hit rate (0.0 to 1.0)
        cached_time_per_subject: Time per subject with cache hit (seconds)
        cold_time_per_subject: Time per subject with cache miss (seconds)
    
    Returns:
        Dict with time estimates
    
    Note: These are EDUCATED GUESSES requiring validation!
    """
    # Base times (ESTIMATES)
    LOAD_TIME_PER_SUBJECT = 5.0  # seconds
    PREPROCESS_TIME_PER_SUBJECT = 20.0  # seconds
    FEATURE_EXTRACT_COLD = cold_time_per_subject  # seconds
    FEATURE_EXTRACT_CACHED = cached_time_per_subject  # seconds
    TRAIN_TIME_PER_MODEL = 30.0  # seconds (varies by model)
    
    total_configs = n_models * n_feature_configs
    
    # Feature extraction time (affected by cache)
    cold_feature_time = n_subjects * FEATURE_EXTRACT_COLD
    cached_feature_time = n_subjects * FEATURE_EXTRACT_CACHED
    
    expected_feature_time = (
        cache_hit_rate * cached_feature_time +
        (1 - cache_hit_rate) * cold_feature_time
    )
    
    # Loading and preprocessing (not cached)
    load_preprocess_time = n_subjects * (LOAD_TIME_PER_SUBJECT + PREPROCESS_TIME_PER_SUBJECT)
    
    # Training time
    # LOSO = n_subjects folds per config
    train_time = total_configs * n_subjects * TRAIN_TIME_PER_MODEL / 60  # rough estimate
    
    # Total estimates
    first_run_time = load_preprocess_time + cold_feature_time
    cached_run_time = load_preprocess_time + cached_feature_time
    expected_time = load_preprocess_time + expected_feature_time
    
    return {
        'n_subjects': n_subjects,
        'n_models': n_models,
        'n_feature_configs': n_feature_configs,
        'total_configurations': total_configs,
        'first_run_minutes': first_run_time / 60,
        'cached_run_minutes': cached_run_time / 60,
        'expected_run_minutes': expected_time / 60,
        'potential_time_saved_minutes': (cold_feature_time - cached_feature_time) / 60,
        'note': 'ESTIMATES - actual times may vary significantly'
    }


def format_time_estimate(estimate: Dict) -> str:
    """Format time estimate for display."""
    lines = [
        f"Time Estimates (EDUCATED GUESSES - require validation):",
        f"  First run (cold):    ~{estimate['first_run_minutes']:.1f} minutes",
        f"  Cached run:          ~{estimate['cached_run_minutes']:.1f} minutes", 
        f"  Potential savings:   ~{estimate['potential_time_saved_minutes']:.1f} minutes",
    ]
    return "\n".join(lines)


# Clinical targets
CLINICAL_TARGETS = {
    'accuracy': 0.85,
    'kappa': 0.75,
    'f1_macro': 0.80,
    'note': 'Standard clinical thresholds for sleep staging'
}

EXPECTED_CLASS_PERFORMANCE = {
    'Wake': {'f1_range': (0.70, 0.90), 'note': 'Generally good'},
    'N1': {'f1_range': (0.20, 0.40), 'note': 'Expected poor - transitional stage'},
    'N2': {'f1_range': (0.75, 0.90), 'note': 'Good - most common stage'},
    'N3': {'f1_range': (0.70, 0.85), 'note': 'Good - distinct delta waves'},
    'REM': {'f1_range': (0.65, 0.85), 'note': 'Moderate - can be confused with Wake'},
}


def print_clinical_targets():
    """Print clinical target information."""
    print("\n" + "=" * 60)
    print("CLINICAL TARGETS FOR SLEEP STAGING")
    print("=" * 60)
    print(f"  Accuracy:      ≥ {CLINICAL_TARGETS['accuracy']*100:.0f}%")
    print(f"  Cohen's Kappa: ≥ {CLINICAL_TARGETS['kappa']:.2f}")
    print(f"  F1 Macro:      ≥ {CLINICAL_TARGETS['f1_macro']*100:.0f}%")
    print("\nExpected Per-Class Performance:")
    print("-" * 60)
    for stage, perf in EXPECTED_CLASS_PERFORMANCE.items():
        f1_low, f1_high = perf['f1_range']
        print(f"  {stage:<6} F1: {f1_low*100:.0f}-{f1_high*100:.0f}%  ({perf['note']})")
    print("-" * 60)
    print("Note: N1 (light sleep) is notoriously difficult to classify.")
    print("      Low N1 F1 (~20-40%) is expected and acceptable.")
    print("=" * 60 + "\n")


# Singleton leaderboard instance
_leaderboard_instance: Optional[CacheLeaderboard] = None


def get_leaderboard() -> CacheLeaderboard:
    """Get or create the global leaderboard instance."""
    global _leaderboard_instance
    if _leaderboard_instance is None:
        _leaderboard_instance = CacheLeaderboard()
    return _leaderboard_instance
