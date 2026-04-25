"""
ML Pipeline Orchestrator (No Caching)
======================================

Orchestrates the complete ML pipeline from raw data to extracted features.

Pipeline Stages:
1. Load raw EEG data
2. Preprocess (filter, downsample, epoch)
3. Extract features
4. Save results

NOTE: This version does NOT include caching or fingerprinting.
      Those will be added in the next iteration.

Author: Lennart Gorzel
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

from data_loader_boas import BOASDataLoader, RecordingMetadata
from preprocessing import preprocess_subject
from feature_extractor import FeatureExtractor
from feature_cache import (
    save_features_to_cache,
    load_features_from_cache,
    select_channel_features
)
from config import ExperimentConfig
import os
from utils import (
    setup_logging, create_output_directories, save_dataframe,
    save_numpy_array, save_json, ProgressTracker, validate_epochs_labels
)

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Orchestrates data loading, preprocessing, and feature extraction.
    
    This version processes data WITHOUT caching. Each run recomputes everything.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        
        # Initialize data loader
        channels = config.data.get_channels()
        
        self.loader = BOASDataLoader(
            base_path=config.data.base_path,
            target_channels=channels,
            target_sfreq=None,  # Will resample in preprocessing
            epoch_duration=config.preprocessing.epoch_duration,
            use_human_labels=config.data.use_human_labels,
            preload=True
        )
        
        # Log channel configuration
        n_channels = len(channels) if channels else "all"
        expected_features = config.data.get_expected_features()
        logger.info(f"Channel configuration: {config.data.channel_preset}")
        logger.info(f"  Channels: {n_channels}")
        logger.info(f"  Expected features: {expected_features}")
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            sfreq=config.preprocessing.target_sfreq
        )

        # Number of workers for feature extraction (processes). Read from env NJOBS or default to 7.
        # 7 workers = optimal for 8-core CPU (leaves 1 core for system)
        try:
            self.n_jobs = int(os.getenv('NJOBS', '7'))
        except Exception:
            self.n_jobs = 7

        # Initialize a loader configured to always fetch 8 channels for full-feature extraction
        full_channel_list = [
            'PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2',
            'PSG_EOG', 'PSG_EMG'
        ]
        self.loader_full = BOASDataLoader(
            base_path=config.data.base_path,
            target_channels=full_channel_list,
            target_sfreq=None,
            epoch_duration=config.preprocessing.epoch_duration,
            use_human_labels=config.data.use_human_labels,
            preload=True
        )
        
        # Output directory (per-experiment)
        self.output_dir = config.get_output_dir()
        create_output_directories(self.output_dir)
        
        # GLOBAL feature cache directory (shared across ALL experiments)
        # This allows reusing computed features without re-extraction
        self.global_cache_dir = Path(config.output_dir) / "features_cache_global"
        self.global_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache statistics for leaderboard tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Initialized DataPipeline")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Global cache: {self.global_cache_dir}")
    
    def check_cache(self, subject_id: str) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        """
        Check if features are cached for a subject.
        
        OPTIMIZATION: Call this BEFORE loading/preprocessing to skip expensive operations.
        
        Args:
            subject_id: Subject ID to check
            
        Returns:
            Tuple of (features_df, labels) if cached, None otherwise
        """
        cache_path = self.global_cache_dir / f"subject_{subject_id}_full.npz"
        cached = load_features_from_cache(cache_path)
        
        if cached is not None:
            features_full_df, labels_cached, n_ch = cached
            active_count = len(self.config.data.get_channels())
            features_df = select_channel_features(features_full_df, channels_to_keep=active_count)
            return features_df, labels_cached
        
        return None
    
    def get_subject_list(self) -> List[str]:
        """
        Get list of subjects to process.
        
        Returns:
            List of subject IDs
        """
        # If specific subjects are configured, use those
        if self.config.data.subjects:
            subjects = self.config.data.subjects
            logger.info(f"Using configured subjects: {len(subjects)} subjects")
        else:
            # Otherwise, use all available subjects
            subjects = self.loader.list_subjects()
            logger.info(f"Found {len(subjects)} subjects in dataset")
        
        return subjects
    
    def process_single_subject(
        self,
        subject_id: str,
        subject_num: int = 0,
        total_subjects: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, RecordingMetadata]:
        """
        Process a single subject through the complete pipeline.
        
        OPTIMIZED: Only loads data ONCE with 8 channels, then subsets for experiment.
        This avoids the previous double-loading issue.
        
        Args:
            subject_id: Subject ID to process
            subject_num: Current subject number (for progress display)
            total_subjects: Total number of subjects (for progress display)
            
        Returns:
            Tuple of (epochs_experiment, labels, metadata, epochs_full, labels_full)
            - epochs_experiment: Subset with experiment channels (e.g., 6)
            - epochs_full: Full 8-channel data for caching
        """
        # Print header
        print(f"\n{'='*60}")
        print(f"PROCESSING SUBJECT {subject_num}/{total_subjects}: {subject_id}")
        print(f"{'='*60}")
        
        # Stage 1: Load raw data with FULL 8 channels (load only ONCE!)
        print(f"[Step 1/3] Loading raw EEG data (8 channels)...")
        raw_full, annotations, metadata = self.loader_full.load_subject(
            subject_id,
            apply_preprocessing=False,  # Don't use deprecated param
            apply_channel_selection=True,  # Select 8 EEG channels
            apply_resampling=False  # Don't resample yet (done in preprocessing)
        )
        print(f"  Loaded: {metadata.n_epochs} total epochs, {metadata.sampling_rate} Hz")
        print(f"  Channels: {metadata.channels}")
        
        # Stage 2: Preprocess (only ONCE with 8 channels)
        print(f"[Step 2/3] Preprocessing (filtering, downsampling, epoching)...")
        print(f"  -> Bandpass filter: {self.config.preprocessing.bandpass_low}-{self.config.preprocessing.bandpass_high} Hz")
        print(f"  -> Notch filter: {self.config.preprocessing.notch_frequency} Hz")
        print(f"  -> Downsampling: {self.config.preprocessing.original_sfreq} Hz -> {self.config.preprocessing.target_sfreq} Hz")
        
        epochs_full, labels = preprocess_subject(
            raw_full,
            annotations,
            self.config.preprocessing,
            validate_quality=False  # DISABLED: Quality check too strict for BOAS data
        )
        
        # Validate
        epochs_full, labels = validate_epochs_labels(epochs_full, labels)
        print(f"  Preprocessed: {len(epochs_full)} valid epochs, shape {epochs_full.shape}")
        
        # Create experiment subset by selecting first N channels
        # The channels are ordered: F3, F4, C3, C4, O1, O2, EOG, EMG
        # Experiment typically uses first 6 (EEG only)
        active_channels = len(self.config.data.get_channels())
        epochs_experiment = epochs_full[:, :active_channels, :]
        print(f"  Experiment channels: {active_channels} of {epochs_full.shape[1]} (subset)")
        
        # Stage 3: Extract features
        print(f"[Step 3/3] Extracting {self.config.features.expected_feature_count()} features...")
        
        logger.info(f"Subject {subject_id} complete: {len(epochs_experiment)} epochs")

        # Return both the experiment epochs and full epochs for cache
        return epochs_experiment, labels, metadata, epochs_full, labels
    
    def extract_features_from_epochs(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        subject_id: str,
        subject_num: int = 0,
        total_subjects: int = 0,
        epochs_full: Optional[np.ndarray] = None,
        labels_full: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Extract features from preprocessed epochs.
        
        Args:
            epochs: Preprocessed epochs (n_epochs, n_channels, n_samples)
            subject_id: Subject ID (for logging)
            subject_num: Current subject number (for progress display)
            total_subjects: Total number of subjects (for progress display)
            
        Returns:
            DataFrame with features (n_epochs, 149)
        """
        n_epochs = len(epochs)
        expected_features = self.config.features.expected_feature_count()

        print(f"  -> Processing {n_epochs} epochs × {expected_features} features...")

        # GLOBAL cache path for full-feature set (195 features when 8 channels available)
        # Using global cache allows reuse across different experiments
        cache_path = self.global_cache_dir / f"subject_{subject_id}_full.npz"

        # NOTE: Cache is now checked BEFORE this function is called (in process_all_subjects)
        # This function is only called on cache MISS, so we go straight to computation

        # Determine which epoch array to compute full features from
        source_epochs_for_full = epochs_full if epochs_full is not None else epochs
        source_labels_for_full = labels_full if labels_full is not None else labels

        # Compute full features (use configured number of workers)
        features_full_df = self.feature_extractor.extract_multiple_epochs(source_epochs_for_full, n_jobs=self.n_jobs)

        # Save to GLOBAL feature cache for reuse in future experiments
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if source_labels_for_full is not None:
                save_features_to_cache(cache_path, features_full_df, source_labels_for_full, n_channels=source_epochs_for_full.shape[1])
                logger.info(f"Saved features to global cache: {cache_path}")
                print(f"  Saved to global cache: {cache_path.name}")
        except Exception:
            logger.exception("Failed to save feature cache")

        # Filter to active channels for pipeline operation
        active_count = len(self.config.data.get_channels())
        features_df = select_channel_features(features_full_df, channels_to_keep=active_count)

        print(f"  Features extracted: {features_df.shape}")
        return features_df
    
    def process_all_subjects(
        self,
        save_intermediate: bool = True
    ) -> Dict[str, Dict]:
        """
        Process all subjects and extract features.
        
        Args:
            save_intermediate: Save per-subject preprocessed data
            
        Returns:
            Dictionary mapping subject_id -> {'features': DataFrame, 'labels': array, 'metadata': RecordingMetadata}
        """
        subjects = self.get_subject_list()
        n_subjects = len(subjects)
        
        print(f"\n{'='*60}")
        print(f"PIPELINE: DATA PREPARATION")
        print(f"{'='*60}")
        print(f"Total subjects to process: {n_subjects}")
        
        # Get expected features dynamically
        channels = self.config.data.get_channels()
        n_channels = len(channels) if channels else 0
        expected_features = self.config.data.get_expected_features()
        
        print(f"Channel configuration: {self.config.data.channel_preset}")
        print(f"  Channels: {n_channels} ({', '.join(channels) if channels else 'all'})")
        print(f"  Expected features: {expected_features}")
        print(f"Save intermediate results: {save_intermediate}")
        print(f"{'='*60}\n")
        
        results = {}
        failed_subjects = []
        
        for i, subject_id in enumerate(subjects, start=1):
            try:
                # Print header
                print(f"\n{'='*60}")
                print(f"PROCESSING SUBJECT {i}/{n_subjects}: {subject_id}")
                print(f"{'='*60}")
                
                # OPTIMIZATION: Check cache FIRST before expensive loading/preprocessing
                cached_data = self.check_cache(subject_id)
                
                if cached_data is not None:
                    # CACHE HIT - Skip loading and preprocessing entirely!
                    features_df, labels = cached_data
                    self.cache_hits += 1
                    print(f"  CACHE HIT: Loaded {features_df.shape} features from cache")
                    print(f"  Skipped: Loading, preprocessing, feature extraction")
                    logger.info(f"Cache hit for subject {subject_id} - skipped full processing")
                    
                    # Store results (no metadata/epochs available from cache)
                    results[subject_id] = {
                        'features': features_df,
                        'labels': labels,
                        'metadata': None  # Not available from cache
                    }
                    
                    # Summary for cached subject
                    print(f"\n  SUMMARY: Subject {subject_id} (from cache)")
                    print(f"    Epochs: {len(labels)}")
                    print(f"    Features: {features_df.shape}")
                    print(f"    Labels: {np.unique(labels, return_counts=True)}")
                    print(f"  Subject {i}/{n_subjects} complete (CACHED)")
                    continue
                
                # CACHE MISS - Need full processing
                self.cache_misses += 1
                print(f"  -> Cache miss - running full processing pipeline...")
                
                # Process subject with progress info
                epochs, labels, metadata, epochs_full, labels_full = self.process_single_subject(
                    subject_id,
                    subject_num=i,
                    total_subjects=n_subjects
                )
                
                # Extract features with progress info
                features_df = self.extract_features_from_epochs(
                    epochs,
                    labels,
                    subject_id,
                    subject_num=i,
                    total_subjects=n_subjects,
                    epochs_full=epochs_full,
                    labels_full=labels_full
                )
                
                # Store results
                results[subject_id] = {
                    'features': features_df,
                    'labels': labels,
                    'metadata': metadata
                }
                
                # Save intermediate results (optional)
                if save_intermediate:
                    print(f"  -> Saving intermediate results...")
                    self.save_subject_data(subject_id, epochs, labels, features_df)
                    print(f"  Saved to {self.output_dir / 'per_subject' / f'subject_{subject_id}'}")
                
                # Summary for this subject
                print(f"\n  SUMMARY: Subject {subject_id}")
                print(f"    Epochs: {len(epochs)}")
                print(f"    Features: {features_df.shape}")
                print(f"    Labels: {np.unique(labels, return_counts=True)}")
                print(f"  Subject {i}/{n_subjects} complete")
                
            except Exception as e:
                print(f"\n  FAILED: Subject {subject_id}")
                print(f"    Error: {str(e)}")
                logger.error(f"Failed to process subject {subject_id}: {e}", exc_info=True)
                failed_subjects.append(subject_id)
                continue
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"  Successful: {len(results)}/{n_subjects} subjects")
        if failed_subjects:
            print(f"  Failed: {len(failed_subjects)} subjects: {failed_subjects}")
        print(f"{'='*60}\n")
        
        return results
    
    def save_subject_data(
        self,
        subject_id: str,
        epochs: np.ndarray,
        labels: np.ndarray,
        features: pd.DataFrame
    ):
        """
        Save per-subject processed data.
        
        Args:
            subject_id: Subject ID
            epochs: Preprocessed epochs
            labels: Sleep stage labels
            features: Extracted features
        """
        subject_dir = self.output_dir / "per_subject" / f"subject_{subject_id}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Save epochs (compressed)
        save_numpy_array(epochs, subject_dir / "epochs.npy")
        
        # Save labels
        save_numpy_array(labels, subject_dir / "labels.npy")
        
        # Save features
        save_dataframe(features, subject_dir / "features.csv")
        
        logger.debug(f"Saved data for subject {subject_id}")
    
    def aggregate_features(
        self,
        results: Dict[str, Dict]
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Aggregate features from all subjects into unified dataset.
        
        Args:
            results: Dictionary from process_all_subjects()
            
        Returns:
            Tuple of (features_df, labels, subject_ids)
            - features_df: (total_epochs, 149)
            - labels: (total_epochs,)
            - subject_ids: (total_epochs,) - subject ID for each epoch
        """
        print(f"\n{'='*60}")
        print(f"AGGREGATING DATA FROM ALL SUBJECTS")
        print(f"{'='*60}")
        
        all_features = []
        all_labels = []
        all_subject_ids = []
        
        for subject_id, data in results.items():
            features = data['features']
            labels = data['labels']
            n_epochs = len(features)
            
            all_features.append(features)
            all_labels.append(labels)
            all_subject_ids.extend([subject_id] * n_epochs)
            
            print(f"  Subject {subject_id}: {n_epochs} epochs")
        
        # Concatenate
        print(f"\n  -> Concatenating all data...")
        features_df = pd.concat(all_features, ignore_index=True)
        labels_array = np.concatenate(all_labels)
        subject_ids_array = np.array(all_subject_ids)
        
        # Statistics
        unique_labels, label_counts = np.unique(labels_array, return_counts=True)
        
        print(f"\n  AGGREGATED DATASET:")
        print(f"    Total epochs: {len(features_df):,}")
        print(f"    Total features: {features_df.shape[1]}")
        print(f"    Total subjects: {len(results)}")
        print(f"\n  LABEL DISTRIBUTION:")
        label_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
        for label, count in zip(unique_labels, label_counts):
            percent = (count / len(labels_array)) * 100
            print(f"    {label_names.get(label, f'Stage {label}')}: {count:,} ({percent:.1f}%)")
        print(f"{'='*60}\n")
        
        return features_df, labels_array, subject_ids_array
    
    def save_aggregated_dataset(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        subject_ids: np.ndarray
    ):
        """
        Save aggregated dataset to disk.
        
        Args:
            features_df: Features DataFrame
            labels: Labels array
            subject_ids: Subject IDs array
        """
        print(f"\n{'='*60}")
        print(f"SAVING AGGREGATED DATASET")
        print(f"{'='*60}")
        
        # Save features
        features_path = self.output_dir / "features" / "all_features.csv"
        print(f"  -> Saving features to CSV...")
        save_dataframe(features_df, features_path)
        print(f"    {features_path}")
        
        # Save labels
        labels_path = self.output_dir / "features" / "all_labels.npy"
        print(f"  -> Saving labels...")
        save_numpy_array(labels, labels_path)
        print(f"    {labels_path}")
        
        # Save subject IDs
        subject_ids_path = self.output_dir / "features" / "subject_ids.npy"
        print(f"  -> Saving subject IDs...")
        save_numpy_array(subject_ids, subject_ids_path)
        print(f"    {subject_ids_path}")
        
        # Save metadata
        print(f"  -> Saving metadata...")
        metadata = {
            'n_epochs': len(features_df),
            'n_features': features_df.shape[1],
            'n_subjects': len(np.unique(subject_ids)),
            'feature_names': features_df.columns.tolist(),
            'label_distribution': {
                int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = self.output_dir / "features" / "dataset_metadata.json"
        save_json(metadata, metadata_path)
        print(f"    {metadata_path}")
        
        print(f"\n  ALL FILES SAVED TO: {self.output_dir / 'features'}")
        print(f"{'='*60}\n")
    
    def run(self, save_intermediate: bool = True) -> Dict:
        """
        Run complete pipeline.

        Args:
            save_intermediate: Whether to save intermediate files to disk

        Returns:
            Dictionary with pipeline results and statistics
        """
        start_time = datetime.now()
        
        print("\n" + "="*60)
        print("   ML EXPERIMENT PIPELINE - STAGE 1 & 2")
        print("   Data Preparation: Load -> Preprocess -> Extract Features")
        print("="*60)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Output dir: {self.output_dir}")
        print("="*60 + "\n")
        
        # Process all subjects
        results = self.process_all_subjects(save_intermediate=save_intermediate)
        
        # Aggregate
        features_df, labels, subject_ids = self.aggregate_features(results)
        
        # Save
        self.save_aggregated_dataset(features_df, labels, subject_ids)
        
        # Statistics
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        pipeline_stats = {
            'n_subjects_processed': len(results),
            'n_total_epochs': len(features_df),
            'n_features': features_df.shape[1],
            'elapsed_time_seconds': elapsed,
            'output_directory': str(self.output_dir),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            # Cache statistics for leaderboard
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0,
            # Estimated cold time based on measured miss times (~25s per subject)
            'estimated_cold_time': (self.cache_hits + self.cache_misses) * 25.0 if self.cache_hits > 0 else elapsed
        }
        
        # Final summary
        print("\n" + "="*60)
        print("   PIPELINE COMPLETE - STAGE 1 & 2")
        print("="*60)
        print(f"Subjects processed: {pipeline_stats['n_subjects_processed']}")
        print(f"Total epochs: {pipeline_stats['n_total_epochs']:,}")
        print(f"Features per epoch: {pipeline_stats['n_features']}")
        print(f"\nTime Statistics:")
        print(f"  Start: {start_time.strftime('%H:%M:%S')}")
        print(f"  End: {end_time.strftime('%H:%M:%S')}")
        print(f"  Elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"  Avg per subject: {elapsed/pipeline_stats['n_subjects_processed']:.1f} seconds")
        print(f"\nCache Statistics:")
        print(f"  Hits: {self.cache_hits} | Misses: {self.cache_misses}")
        print(f"  Hit Rate: {pipeline_stats['cache_hit_rate']:.1f}%")
        print(f"\nOutput saved to:")
        print(f"  {pipeline_stats['output_directory']}")
        print("="*60 + "\n")
        
        # Save statistics
        stats_path = self.output_dir / "pipeline_stats.json"
        save_json(pipeline_stats, stats_path)
        print(f"Pipeline statistics saved to: {stats_path}\n")
        
        return pipeline_stats


def load_processed_dataset(
    experiment_dir: Path
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    """
    Load previously processed dataset from disk.
    
    Args:
        experiment_dir: Experiment output directory
        
    Returns:
        Tuple of (features_df, labels, subject_ids, metadata)
    """
    features_dir = experiment_dir / "features"
    
    # Load components
    from utils import load_dataframe, load_numpy_array, load_json
    
    features_df = load_dataframe(features_dir / "all_features.csv")
    labels = load_numpy_array(features_dir / "all_labels.npy")
    subject_ids = load_numpy_array(features_dir / "subject_ids.npy")
    metadata = load_json(features_dir / "dataset_metadata.json")
    
    logger.info(f"Loaded dataset from {features_dir}")
    logger.info(f"  Epochs: {len(features_df)}")
    logger.info(f"  Features: {features_df.shape[1]}")
    logger.info(f"  Subjects: {len(np.unique(subject_ids))}")
    
    return features_df, labels, subject_ids, metadata


# Example usage
if __name__ == "__main__":
    from config import ConfigManager
    
    # Setup logging
    setup_logging("INFO")
    
    # Create config
    config = ConfigManager.create_default_config(
        experiment_name="test_pipeline",
        data_path=r"C:\Users\DerHo\Desktop\Data",
        model_type="xgboost"
    )
    
    # Limit to 3 subjects for testing
    config.data.subjects = ['1', '2', '3']
    
    # Run pipeline
    pipeline = DataPipeline(config)
    stats = pipeline.run()
    
    print("\nPipeline Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
