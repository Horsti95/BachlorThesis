"""
Interactive Menu for ML Caching Research Pipeline
=================================================

Provides an interactive interface for configuring and running experiments.
Focused on cache performance validation for thesis research.

Research Questions:
- SQ1: Efficiency (time saved via caching)
- SQ2: Reproducibility (deterministic results)
- SQ3: Scalability (performance across dataset sizes)
- SQ4: Trade-offs (storage vs compute)

Author: Lennart Gorzel
Date: December 2025
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from config import (
    ConfigManager, ExperimentConfig, DataConfig, PreprocessingConfig,
    FeatureConfig, ModelConfig, CrossValidationConfig
)
from leaderboard import (
    get_leaderboard, estimate_experiment_time, format_time_estimate,
    print_clinical_targets, CLINICAL_TARGETS
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Grid (2 × 3 × 3 = 18 configurations)
# =============================================================================

CORRELATION_THRESHOLDS = [0.75, 0.95]  # 2 options
TOP_K_FEATURES = [30, 50, None]  # 3 options (None = all)
MODELS = ['xgboost', 'random_forest', 'fnn']  # 3 options

# Preset combinations
FEATURE_STRATEGIES = {
    'quick': {
        'name': 'Quick Test',
        'description': 'Single config: corr=0.95, top_k=30',
        'configs': [(0.95, 30)]
    },
    'comprehensive': {
        'name': 'Comprehensive',
        'description': '4 combinations: corr=[0.75,0.95] × top_k=[30,50]',
        'configs': [
            (0.75, 30), (0.75, 50),
            (0.95, 30), (0.95, 50)
        ]
    },
    'full_grid': {
        'name': 'Full Grid',
        'description': '6 combinations: corr=[0.75,0.95] × top_k=[30,50,all]',
        'configs': [
            (corr, k) for corr in CORRELATION_THRESHOLDS for k in TOP_K_FEATURES
        ]
    }
}

# Subject presets
SUBJECT_PRESETS = {
    '1': {
        'name': 'Single Subject',
        'subjects': [1],
        'description': 'Quick test (~30 seconds)',
        'time_estimate_cold': 0.5,  # minutes (ESTIMATE)
        'time_estimate_cached': 0.1  # minutes (ESTIMATE)
    },
    '3': {
        'name': 'Three Subjects', 
        'subjects': [1, 2, 3],
        'description': 'Standard test (~2 minutes)',
        'time_estimate_cold': 2.0,
        'time_estimate_cached': 0.5
    },
    '10': {
        'name': 'Ten Subjects',
        'subjects': list(range(1, 11)),
        'description': 'Pilot analysis (~7 minutes)',
        'time_estimate_cold': 7.0,
        'time_estimate_cached': 1.5
    },
    '128': {
        'name': 'All Subjects',
        'subjects': list(range(1, 129)),
        'description': 'Full dataset (~90 minutes)',
        'time_estimate_cold': 90.0,
        'time_estimate_cached': 15.0
    }
}

# Model info
MODEL_INFO = {
    'xgboost': {
        'name': 'XGBoost',
        'status': '✓ IMPLEMENTED',
        'description': 'Gradient boosting - fast, accurate',
        'implemented': True
    },
    'random_forest': {
        'name': 'Random Forest',
        'status': '✓ IMPLEMENTED',
        'description': 'Ensemble trees - robust baseline',
        'implemented': True
    },
    'fnn': {
        'name': 'Feedforward Neural Network',
        'status': '⚠ TODO',
        'description': 'Deep learning - may have ±0.5% nondeterminism',
        'implemented': False,
        'note': 'FNN nondeterminism expected due to GPU/parallel ops'
    }
}


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print application header."""
    print("=" * 70)
    print("FINGERPRINT-BASED ML CACHING RESEARCH PIPELINE")
    print("=" * 70)
    print("Bachelor Thesis Research - Sleep Stage Classification")
    print("")
    print("Research Focus: Intelligent caching for reproducible ML experiments")
    print("Dataset: BOAS (128 subjects, 6 EEG channels, 149 features)")
    print("")


def print_research_context():
    """Print thesis research questions (SQ1-SQ4) to the console."""
    print("-" * 70)
    print("THESIS RESEARCH QUESTIONS")
    print("-" * 70)
    print("Main: How can fingerprint-based caching reduce computational costs")
    print("      while ensuring reproducibility?")
    print("")
    print("Sub-questions:")
    print("  SQ1: Efficiency - Real time savings from caching")
    print("  SQ2: Reproducibility - Deterministic cached reuse")
    print("  SQ3: Scalability - Performance across dataset sizes")
    print("  SQ4: Trade-offs - Storage vs compute cost")
    print("-" * 70)


def print_cache_targets():
    """Print estimated cache timing targets (RAM hit, disk hit, cold miss)."""
    print("\n" + "-" * 70)
    print("CACHE TIMING TARGETS (ESTIMATES - require validation)")
    print("-" * 70)
    print(f"  {'Event Type':<20} {'Target':<15} {'Status':<15}")
    print("-" * 70)
    print(f"  {'RAM Cache Hit':<20} {'≈ 0.1 seconds':<15} {'ESTIMATE':<15}")
    print(f"  {'Disk Cache Hit':<20} {'≈ 1.0 seconds':<15} {'ESTIMATE':<15}")
    print(f"  {'Cold Miss (extract)':<20} {'≈ 15-30 sec':<15} {'ESTIMATE':<15}")
    print("-" * 70)
    print("Note: These are educated guesses. Chapter 5 will validate experimentally.")
    print("-" * 70 + "\n")


def print_leaderboard_summary():
    """Print one-line cache leaderboard stats (experiments, hit rate, time saved)."""
    leaderboard = get_leaderboard()
    
    if leaderboard.total_cache_hits + leaderboard.total_cache_misses == 0:
        print("No experiments recorded yet. Run an experiment to start tracking!")
    else:
        print(f"Experiments: {len(leaderboard.runs)} | "
              f"Hit Rate: {leaderboard.overall_hit_rate():.1f}% | "
              f"Time Saved: {leaderboard.total_time_saved/60:.1f} min")


class InteractiveMenu:
    """Interactive menu system for configuring and launching ML caching experiments."""

    def __init__(self, data_path: str = r"C:\Users\DerHo\Desktop\Data"):
        self.data_path = data_path
        self.selected_subjects: List[int] = []
        self.selected_models: List[str] = []
        self.selected_feature_strategy: str = 'quick'
        self.cv_method: str = 'loso'
        
    def run(self):
        """Run the main menu loop. Returns an ExperimentConfig or exits."""
        while True:
            clear_screen()
            print_header()
            print_leaderboard_summary()
            print_clinical_targets()
            
            print("\n" + "-" * 70)
            print("MAIN MENU")
            print("-" * 70)
            print("1. Run Experiment (Configure & Execute)")
            print("2. Quick Test (3 subjects, XGBoost, cached)")
            print("3. View Cache Leaderboard")
            print("4. View Cache Timing Targets")
            print("5. View Research Context")
            print("6. System Status")
            print("7. Exit")
            print("-" * 70)
            
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == '1':
                config = self.configure_experiment()
                if config is not None:
                    return config
            elif choice == '2':
                return self.quick_test_config()
            elif choice == '3':
                self.view_leaderboard()
            elif choice == '4':
                print_cache_targets()
                input("\nPress Enter to continue...")
            elif choice == '5':
                print_research_context()
                input("\nPress Enter to continue...")
            elif choice == '6':
                self.system_status()
            elif choice == '7':
                print("\nThank you for using the ML Caching Research Pipeline!")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1-7.")
                input("Press Enter to continue...")
    
    def configure_experiment(self) -> Optional[ExperimentConfig]:
        """Step-by-step wizard: subjects -> models -> features -> confirm. Returns config or None."""
        clear_screen()
        print("=" * 70)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 70)
        print("Let's configure your experiment step by step...\n")
        
        # Step 1: Subject Selection
        subjects = self.select_subjects()
        if subjects is None:
            return None
        
        # Step 2: Model Selection
        models = self.select_models()
        if models is None:
            return None
        
        # Step 3: Feature Strategy
        feature_strategy = self.select_feature_strategy()
        if feature_strategy is None:
            return None
        
        # Step 4: Show configuration and time estimate
        config = self.build_config(subjects, models, feature_strategy)
        
        if self.confirm_configuration(config, subjects, models, feature_strategy):
            return config
        
        return None
    
    def select_subjects(self) -> Optional[List[int]]:
        """Prompt user to select subjects from presets or custom input. Returns list or None."""
        print("=" * 50)
        print("STEP 1: SUBJECT SELECTION")
        print("=" * 50)
        print(f"Available subjects: 128 (BOAS dataset)\n")
        
        print("SUBJECT PRESETS:")
        for key, preset in SUBJECT_PRESETS.items():
            n = len(preset['subjects'])
            print(f"  [{key}] {preset['name']} ({n} subject{'s' if n > 1 else ''}) - {preset['description']}")
        print("  [C] Custom selection")
        
        choice = input("\nEnter choice: ").strip().upper()
        
        if choice in SUBJECT_PRESETS:
            return SUBJECT_PRESETS[choice]['subjects']
        elif choice == 'C':
            return self.custom_subject_selection()
        else:
            print("Invalid choice.")
            input("Press Enter to continue...")
            return None
    
    def custom_subject_selection(self) -> Optional[List[int]]:
        """Parse comma-separated or range input (e.g. '1-10,15') into subject list."""
        print("\nCustom subject selection:")
        print("Enter subject numbers separated by commas (e.g., 1,5,10,15)")
        print("Or enter a range like 1-10")
        
        selection = input("Subjects: ").strip()
        
        try:
            subjects = []
            for part in selection.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    subjects.extend(range(start, end + 1))
                else:
                    subjects.append(int(part))
            
            # Validate
            subjects = sorted(set(s for s in subjects if 1 <= s <= 128))
            if subjects:
                print(f"Selected {len(subjects)} subjects: {subjects[:5]}{'...' if len(subjects) > 5 else ''}")
                return subjects
            else:
                print("No valid subjects selected.")
                return None
                
        except ValueError:
            print("Invalid input format.")
            return None
    
    def select_models(self) -> Optional[List[str]]:
        """Prompt user to select ML models from presets or custom input."""
        print("\n" + "=" * 50)
        print("STEP 2: MODEL SELECTION")
        print("=" * 50)
        
        print("\nAVAILABLE MODELS:")
        for key, info in MODEL_INFO.items():
            print(f"  [{key}] {info['name']:<25} {info['status']}")
            print(f"         {info['description']}")
            if 'note' in info:
                print(f"         Note: {info['note']}")
        
        print("\nPRESETS:")
        print("  [1] XGBoost only (recommended for initial tests)")
        print("  [2] XGBoost + Random Forest")
        print("  [3] All models (XGBoost + RF + FNN)")
        print("  [C] Custom selection")
        
        choice = input("\nEnter choice: ").strip().upper()
        
        if choice == '1':
            return ['xgboost']
        elif choice == '2':
            return ['xgboost', 'random_forest']
        elif choice == '3':
            return ['xgboost', 'random_forest', 'fnn']
        elif choice == 'C':
            return self.custom_model_selection()
        else:
            print("Invalid choice.")
            return None
    
    def custom_model_selection(self) -> Optional[List[str]]:
        """Parse comma-separated model names from user input."""
        print("\nEnter model names separated by commas:")
        print(f"Available: {', '.join(MODEL_INFO.keys())}")
        
        selection = input("Models: ").strip().lower()
        models = [m.strip() for m in selection.split(',')]
        
        valid_models = [m for m in models if m in MODEL_INFO]
        if valid_models:
            return valid_models
        return None
    
    def select_feature_strategy(self) -> Optional[str]:
        """Prompt user to choose a feature selection strategy (quick/comprehensive/full/custom)."""
        print("\n" + "=" * 50)
        print("STEP 3: FEATURE SELECTION STRATEGY")
        print("=" * 50)
        print("\nConfiguration Grid: Correlation Threshold × Top-K Features")
        print("(Part of the 2×3×3 = 18 total configurations)")
        
        print("\nFEATURE STRATEGIES:")
        for key, strategy in FEATURE_STRATEGIES.items():
            n_configs = len(strategy['configs'])
            print(f"  [{key}] {strategy['name']} ({n_configs} config{'s' if n_configs > 1 else ''})")
            print(f"         {strategy['description']}")
        
        print("\n  [C] Custom (pick your own correlation/top-k)")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice in FEATURE_STRATEGIES:
            return choice
        elif choice == 'c':
            return 'custom'
        else:
            print("Invalid choice.")
            return None
    
    def build_config(
        self,
        subjects: List[int],
        models: List[str],
        feature_strategy: str
    ) -> ExperimentConfig:
        """Build an ExperimentConfig from the user's subject, model, and feature selections."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_subjects = len(subjects)
        
        # For now, use first model (multi-model support is config expansion)
        primary_model = models[0]
        
        experiment_name = f"Exp_S{n_subjects}_{primary_model}_{timestamp}"
        
        # Get feature config from strategy
        if feature_strategy in FEATURE_STRATEGIES:
            first_config = FEATURE_STRATEGIES[feature_strategy]['configs'][0]
            corr_threshold, top_k = first_config
        else:
            corr_threshold, top_k = 0.95, None
        
        # Get all feature configs from the strategy
        if feature_strategy in FEATURE_STRATEGIES:
            feature_configs = FEATURE_STRATEGIES[feature_strategy]['configs']
        else:
            feature_configs = [(corr_threshold, top_k)]

        config = ExperimentConfig(
            experiment_name=experiment_name,
            data=DataConfig(
                base_path=self.data_path,
                subjects=[str(s) for s in subjects]
            ),
            preprocessing=PreprocessingConfig(),
            features=FeatureConfig(correlation_threshold=corr_threshold),
            model=ModelConfig(model_type=primary_model),
            cross_validation=CrossValidationConfig(method='loso'),
            training_models=models,
            training_feature_configs=feature_configs,
        )

        return config
    
    def confirm_configuration(
        self,
        config: ExperimentConfig,
        subjects: List[int],
        models: List[str],
        feature_strategy: str
    ) -> bool:
        """Display config summary with time estimate and ask user to confirm. Returns True if confirmed."""
        
        clear_screen()
        print("=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        
        print(f"\nExperiment: {config.experiment_name}")
        print(f"Subjects: {len(subjects)} ({subjects[:3]}{'...' if len(subjects) > 3 else ''})")
        print(f"Models: {', '.join(models)}")
        print(f"Feature Strategy: {feature_strategy}")
        print(f"CV Method: LOSO (Leave-One-Subject-Out)")
        
        # Time estimate
        print("\n" + "-" * 70)
        n_feature_configs = len(FEATURE_STRATEGIES.get(feature_strategy, {}).get('configs', [(None, None)]))
        estimate = estimate_experiment_time(
            n_subjects=len(subjects),
            n_models=len(models),
            n_feature_configs=n_feature_configs,
            cache_hit_rate=0.8  # Assume 80% hit rate for estimate
        )
        print(format_time_estimate(estimate))
        print("-" * 70)
        
        # Model implementation status
        print("\nModel Status:")
        for model in models:
            info = MODEL_INFO.get(model, {})
            print(f"  {model}: {info.get('status', 'UNKNOWN')}")
        
        # Warning for unimplemented models
        unimplemented = [m for m in models if not MODEL_INFO.get(m, {}).get('implemented', False)]
        if unimplemented:
            print(f"\n⚠ WARNING: {', '.join(unimplemented)} not yet implemented!")
            print("  These will be skipped. Implement them first or remove from selection.")
        
        print("\n" + "-" * 70)
        response = input("Proceed with this configuration? (y/n/edit): ").strip().lower()
        
        return response == 'y'
    
    def quick_test_config(self) -> ExperimentConfig:
        """Create a quick test config: 3 subjects, XGBoost, LOSO CV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        config = ExperimentConfig(
            experiment_name=f"quick_test_{timestamp}",
            data=DataConfig(
                base_path=self.data_path,
                subjects=['1', '2', '3']
            ),
            preprocessing=PreprocessingConfig(),
            features=FeatureConfig(),
            model=ModelConfig(model_type='xgboost'),
            cross_validation=CrossValidationConfig(method='loso')
        )
        
        return config
    
    def view_leaderboard(self):
        """Display the full cache performance leaderboard."""
        clear_screen()
        leaderboard = get_leaderboard()
        leaderboard.print_summary()
        input("\nPress Enter to continue...")
    
    def system_status(self):
        """Show data path, cache status, model implementation status, and thesis grid info."""
        clear_screen()
        print("=" * 70)
        print("SYSTEM STATUS")
        print("=" * 70)
        
        # Check paths
        data_path = Path(self.data_path)
        cache_path = Path("results/features_cache_global")
        leaderboard_path = Path("results/cache_leaderboard.json")
        
        print(f"\nData Path: {data_path}")
        print(f"  Status: {'✓ EXISTS' if data_path.exists() else '✗ NOT FOUND'}")
        
        print(f"\nGlobal Cache: {cache_path}")
        if cache_path.exists():
            cache_files = list(cache_path.glob("*.npz"))
            total_size = sum(f.stat().st_size for f in cache_files) / (1024*1024)
            print(f"  Status: ✓ EXISTS ({len(cache_files)} files, {total_size:.1f} MB)")
        else:
            print("  Status: ✗ NOT INITIALIZED")
        
        print(f"\nLeaderboard: {leaderboard_path}")
        print(f"  Status: {'✓ EXISTS' if leaderboard_path.exists() else '○ Will be created'}")
        
        # Model status
        print("\nModel Implementation Status:")
        for model, info in MODEL_INFO.items():
            print(f"  {model:<15} {info['status']}")
        
        # Thesis configuration
        print("\nThesis Configuration Grid:")
        print(f"  Correlation thresholds: {CORRELATION_THRESHOLDS}")
        print(f"  Top-K features: {TOP_K_FEATURES}")
        print(f"  Models: {MODELS}")
        print(f"  Total: 2 × 3 × 3 = 18 configurations")
        print(f"  With 128 subjects LOSO: 18 × 128 = 2,304 model trainings")
        
        input("\nPress Enter to continue...")


def run_interactive_menu(data_path: str = None) -> Optional[ExperimentConfig]:
    """
    Run the interactive menu and return the configuration.
    
    Args:
        data_path: Path to BOAS dataset
        
    Returns:
        ExperimentConfig if user confirms, None if cancelled
    """
    if data_path is None:
        data_path = r"C:\Users\DerHo\Desktop\Data"
    
    menu = InteractiveMenu(data_path=data_path)
    return menu.run()


if __name__ == "__main__":
    # Test the menu
    config = run_interactive_menu()
    if config:
        print(f"\nConfiguration created: {config.experiment_name}")
