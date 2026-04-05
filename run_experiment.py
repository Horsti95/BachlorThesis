"""
Command-Line Interface for ML Caching Experiments
==================================================

Main entry point for running experiments.

Usage:
    python run_experiment.py                # Interactive menu
    python run_experiment.py --interactive  # Interactive menu (explicit)
    python run_experiment.py --config configs/experiment.yaml
    python run_experiment.py --quick-test   # Run on 3 subjects
    python run_experiment.py --pilot        # Run on 10 subjects

Author: Lennart Gorzel
Date: December 2025
"""

import argparse
from pathlib import Path
import sys
import logging

from config import ConfigManager, ExperimentConfig
from pipeline import DataPipeline, load_processed_dataset
from utils import setup_logging, get_timestamp
from leaderboard import get_leaderboard, print_clinical_targets

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ML Experiment with Intelligent Caching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    # Interactive mode
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run interactive menu (default if no args)'
    )
    
    # Quick modes
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (3 subjects)'
    )
    
    parser.add_argument(
        '--pilot',
        action='store_true',
        help='Pilot mode (10 subjects)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full dataset (128 subjects)'
    )
    
    parser.add_argument(
        '--subjects',
        type=int,
        help='Number of subjects to process (1-128)'
    )
    
    # Override options
    parser.add_argument(
        '--data-path',
        type=str,
        help='Override data path'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Override experiment name'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['xgboost', 'random_forest', 'fnn'],
        help='Override model type'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log to file (in addition to console)'
    )
    
    # Processing options
    parser.add_argument(
        '--no-save-intermediate',
        action='store_true',
        help='Don\'t save per-subject data (saves disk space)'
    )
    
    # Leaderboard
    parser.add_argument(
        '--show-leaderboard',
        action='store_true',
        help='Show cache performance leaderboard and exit'
    )
    
    parser.add_argument(
        '--show-targets',
        action='store_true',
        help='Show clinical targets and exit'
    )
    
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompt (auto-confirm)'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> ExperimentConfig:
    """
    Create configuration from command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        ExperimentConfig
    """
    # Load from file if provided
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config = ConfigManager.from_yaml(args.config)
    else:
        # Create default config
        data_path = args.data_path or r"C:\Users\DerHo\Desktop\Data"
        experiment_name = args.experiment_name or f"experiment_{get_timestamp()}"
        model_type = args.model or 'xgboost'
        
        config = ConfigManager.create_default_config(
            experiment_name=experiment_name,
            data_path=data_path,
            model_type=model_type
        )
    
    # Apply overrides
    if args.data_path:
        config.data.base_path = args.data_path
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    if args.model:
        config.model.model_type = args.model
    
    config.log_level = args.log_level
    
    # Handle mode selections
    if args.quick_test:
        logger.info("Quick test mode: 3 subjects")
        config.data.subjects = ['1', '2', '3']
        config.experiment_name += "_quick"
    elif args.pilot:
        logger.info("Pilot mode: 10 subjects")
        # First 10 subjects
        all_subjects = range(1, 129)
        config.data.subjects = [str(i) for i in all_subjects[:10]]
        config.experiment_name += "_pilot"
    elif args.full:
        logger.info("Full dataset mode: all 128 subjects")
        config.data.subjects = None  # Use all
        config.experiment_name += "_full"
    elif args.subjects:
        n = min(max(1, args.subjects), 128)  # Clamp to 1-128
        logger.info(f"Custom subject count: {n} subjects")
        config.data.subjects = [str(i) for i in range(1, n + 1)]
        config.experiment_name += f"_{n}subj"
    
    return config


def print_experiment_summary(config: ExperimentConfig):
    """Print experiment configuration summary."""
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Name: {config.experiment_name}")
    print(f"Output: {config.get_output_dir()}")
    print(f"\nData:")
    print(f"  Path: {config.data.base_path}")
    print(f"  Subjects: {len(config.data.subjects) if config.data.subjects else 'all (128)'}")
    
    # Get channels using the method
    channels = config.data.get_channels()
    if channels:
        print(f"  Channel preset: {config.data.channel_preset}")
        print(f"  Channels: {len(channels)} ({', '.join(channels[:3])}...)")
        print(f"  Expected features: {config.data.get_expected_features()}")
    else:
        print(f"  Channels: all available")
    
    print(f"  Labels: {'Human consensus' if config.data.use_human_labels else 'AI predictions'}")
    print(f"\nPreprocessing:")
    print(f"  Bandpass: {config.preprocessing.bandpass_low}-{config.preprocessing.bandpass_high} Hz")
    print(f"  Notch: {config.preprocessing.notch_frequency} Hz")
    print(f"  Sampling: {config.preprocessing.original_sfreq} → {config.preprocessing.target_sfreq} Hz")
    print(f"  Epoch duration: {config.preprocessing.epoch_duration} s")
    print(f"\nFeatures:")
    expected_features = config.data.get_expected_features()
    print(f"  Expected count: {expected_features if expected_features else 'dynamic (depends on channels)'}")
    print(f"  Correlation filter: {config.features.correlation_threshold or 'None'}")
    print(f"\nModel:")
    print(f"  Type: {config.model.model_type}")
    print(f"  Random seed: {config.model.random_seed}")
    print(f"\nCross-Validation:")
    print(f"  Method: {config.cross_validation.method.upper()}")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Handle info commands first
    if args.show_leaderboard:
        leaderboard = get_leaderboard()
        leaderboard.print_summary()
        return
    
    if args.show_targets:
        print_clinical_targets()
        return
    
    # Setup logging
    log_file = args.log_file
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    setup_logging(args.log_level, log_file)
    
    # Check if interactive mode (no specific mode selected)
    use_interactive = (
        args.interactive or 
        (not args.config and not args.quick_test and not args.pilot and not args.full and not args.subjects)
    )
    
    if use_interactive:
        # Import here to avoid circular imports
        from interactive_menu import run_interactive_menu

        logger.info("Starting interactive menu...")
        config = run_interactive_menu(args.data_path or r"C:\Users\DerHo\Desktop\Data")

        if config is None:
            logger.info("Menu cancelled. Exiting.")
            return

        # Config is already fully configured by the interactive menu
        
    else:
        logger.info("="*60)
        logger.info("ML EXPERIMENT WITH INTELLIGENT CACHING")
        logger.info("="*60)
        
        # Create configuration from args
        config = create_config_from_args(args)
    
    try:
        
        # Print summary
        print_experiment_summary(config)
        
        # Confirm with user (if not in batch mode and not from interactive menu)
        if sys.stdout.isatty() and not use_interactive and not args.yes:
            response = input("Proceed with experiment? [y/N]: ")
            if response.lower() != 'y':
                logger.info("Experiment cancelled by user")
                return
        
        # Initialize leaderboard tracking
        leaderboard = get_leaderboard()
        run = leaderboard.start_run(
            config.experiment_name,
            {'subjects': len(config.data.subjects) if config.data.subjects else 128,
             'model': config.model.model_type}
        )
        
        # Create and run pipeline
        logger.info("Initializing pipeline...")
        pipeline = DataPipeline(config)
        
        # Run
        save_intermediate = not args.no_save_intermediate
        stats = pipeline.run(save_intermediate=save_intermediate)
        
        # Update leaderboard
        leaderboard.finalize_run(
            run,
            cache_hits=stats.get('cache_hits', 0),
            cache_misses=stats.get('cache_misses', 0),
            total_time=stats.get('elapsed_time_seconds', 0)
        )
        
        # Print results
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("="*60)
        print(f"Subjects processed: {stats['n_subjects_processed']}")
        print(f"Total epochs: {stats['n_total_epochs']}")
        print(f"Features extracted: {stats['n_features']}")
        print(f"Time elapsed: {stats['elapsed_time_seconds']:.1f} seconds")
        print(f"               ({stats['elapsed_time_seconds']/60:.1f} minutes)")
        
        # Cache stats
        if 'cache_hits' in stats:
            total = stats.get('cache_hits', 0) + stats.get('cache_misses', 0)
            hit_rate = (stats['cache_hits'] / total * 100) if total > 0 else 0
            print(f"\nCache Performance:")
            print(f"  Hits: {stats.get('cache_hits', 0)} | Misses: {stats.get('cache_misses', 0)}")
            print(f"  Hit Rate: {hit_rate:.1f}%")
        
        print(f"\nResults saved to: {stats['output_directory']}")
        print("="*60 + "\n")
        
        logger.info("Experiment completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("\nExperiment interrupted by user (Ctrl+C)")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()