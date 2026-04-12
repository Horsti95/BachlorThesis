"""
Data Loader for Bitbrain Open Access Sleep (BOAS) Dataset
===========================================================

Handles loading and validation of EEG recordings and sleep stage annotations
from the BOAS dataset.

BOAS Structure:
- EEG Channels: PSG_F3, PSG_F4, PSG_C3, PSG_C4, PSG_O1, PSG_O2 (6 channels)
- Original Sampling Rate: 256 Hz
- Sleep Stages: 0=W, 1=N1, 2=N2, 3=N3, 4=REM, 8=Disconnection (filter)
- Epoch Duration: 30 seconds
- Labels: Human consensus (stage_hum) and AI predictions (stage_ai)

Author: Lennart Gorzel
Date: December 2025
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import mne
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RecordingMetadata:
    """Metadata for a single BOAS recording."""
    subject_id: str
    filepath: Path
    annotations_filepath: Path
    sampling_rate: float
    channels: List[str]
    duration_seconds: float
    n_epochs: int
    n_valid_epochs: int  # After filtering disconnections
    available_stages: List[int]
    human_labels: bool  # True if using stage_hum
    ai_labels: bool  # True if using stage_ai


class BOASSleepStageMapper:
    """Maps BOAS integer sleep stage codes to/from human-readable labels.

    Valid stages: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM.
    Invalid stages (8=disconnection, -2=artifact) are mapped to -1.
    """
    
    # BOAS standard mapping
    STAGE_TO_INT = {
        0: 0,  # Wake
        1: 1,  # N1
        2: 2,  # N2
        3: 3,  # N3
        4: 4,  # REM
        8: -1,  # Disconnection → filter out
        -2: -1,  # Artifact (AI only) → filter out
    }
    
    INT_TO_STAGE = {
        0: 'Wake',
        1: 'N1',
        2: 'N2',
        3: 'N3',
        4: 'REM',
        -1: 'Invalid'
    }
    
    @classmethod
    def is_valid(cls, stage: int) -> bool:
        """Check if stage should be kept (not disconnection/artifact)."""
        return stage in [0, 1, 2, 3, 4]
    
    @classmethod
    def to_string(cls, stage_int: int) -> str:
        """Convert integer to string label."""
        return cls.INT_TO_STAGE.get(stage_int, 'Unknown')


class BOASDataLoader:
    """
    Loader for Bitbrain Open Access Sleep (BOAS) dataset.
    
    Handles:
    - EDF file reading with 6 EEG channels
    - Annotation file parsing (TSV with stage_hum and stage_ai)
    - Disconnection/artifact filtering
    - Sampling rate validation
    - Quality checks
    """
    
    # Standard BOAS PSG EEG channels
    BOAS_PSG_CHANNELS = ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2']
    
    def __init__(
        self,
        base_path: str,
        target_channels: Optional[List[str]] = None,
        target_sfreq: Optional[float] = 128.0,
        epoch_duration: float = 30.0,
        use_human_labels: bool = True,
        preload: bool = False
    ):
        """
        Initialize BOAS data loader.
        
        Args:
            base_path: Root directory containing sub-X folders
            target_channels: Desired channels (default: 6 PSG EEG channels)
            target_sfreq: Target sampling rate (128 Hz recommended, None = keep 256 Hz)
            epoch_duration: Duration of each epoch in seconds (30s standard)
            use_human_labels: Use stage_hum (True) or stage_ai (False)
            preload: Load data into memory immediately
        """
        self.base_path = Path(base_path)
        self.target_channels = target_channels or self.BOAS_PSG_CHANNELS
        self.target_sfreq = target_sfreq
        self.epoch_duration = epoch_duration
        self.use_human_labels = use_human_labels
        self.preload = preload
        
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path not found: {self.base_path}")
        
        logger.info(f"Initialized BOASDataLoader: {self.base_path}")
        logger.info(f"Target channels: {self.target_channels}")
        logger.info(f"Target sampling rate: {self.target_sfreq} Hz")
        logger.info(f"Using {'human' if use_human_labels else 'AI'} labels")
    
    def list_subjects(self) -> List[str]:
        """Return sorted list of subject IDs found in base_path (sub-* directories)."""
        subject_dirs = sorted([d for d in self.base_path.iterdir() 
                              if d.is_dir() and d.name.startswith('sub-')])
        subject_ids = [d.name.replace('sub-', '') for d in subject_dirs]
        logger.info(f"Found {len(subject_ids)} subjects")
        return subject_ids
    
    def get_subject_files(self, subject_id: str) -> Tuple[Path, Path]:
        """
        Get PSG EDF and annotation file paths for a subject.
        
        Args:
            subject_id: Subject ID (e.g., '1', '001', 'sub-1')
        
        Returns:
            Tuple of (psg_edf_path, annotations_path)
        """
        # Normalize subject ID
        if not subject_id.startswith('sub-'):
            subject_id = f"sub-{subject_id}"
        
        subject_dir = self.base_path / subject_id / "eeg"
        
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
        
        # Find PSG EDF file (not headband)
        psg_files = list(subject_dir.glob("*_acq-psg_eeg.edf"))
        if not psg_files:
            raise FileNotFoundError(f"No PSG EDF file found in {subject_dir}")
        if len(psg_files) > 1:
            logger.warning(f"Multiple PSG EDF files found for {subject_id}, using first")
        psg_path = psg_files[0]
        
        # Find annotation file (shared by PSG and headband)
        annotation_files = list(subject_dir.glob("*_events.txt"))
        if not annotation_files:
            raise FileNotFoundError(f"No annotation file found in {subject_dir}")
        annotation_path = annotation_files[0]
        
        return psg_path, annotation_path
    
    def print_subject_info(
        self, 
        subject_id: str, 
        raw: mne.io.Raw, 
        annotations: pd.DataFrame,
        psg_path: Path,
        annotation_path: Path
    ) -> None:
        """
        Print detailed subject information for debugging/verification.
        
        Similar to verbose output in research pipelines for transparency.
        """
        print(f"\n{'='*60}")
        print(f"SUBJECT: {subject_id}")
        print(f"{'='*60}")
        
        # File paths
        print(f"\n[*] File Paths:")
        print(f"   EDF:    {psg_path}")
        print(f"   Events: {annotation_path}")
        
        # EDF metadata
        print(f"\n[*] EDF Metadata:")
        print(f"   Channels:    {len(raw.ch_names)} ({', '.join(raw.ch_names[:6])}{'...' if len(raw.ch_names) > 6 else ''})")
        print(f"   Sfreq:       {raw.info['sfreq']} Hz")
        print(f"   Duration:    {raw.times[-1]:.1f}s ({raw.times[-1]/3600:.2f}h)")
        print(f"   Samples:     {raw.n_times:,}")
        if raw.info.get('meas_date'):
            print(f"   Recorded:    {raw.info['meas_date']}")
        print(f"   Highpass:    {raw.info.get('highpass', 'N/A')} Hz")
        print(f"   Lowpass:     {raw.info.get('lowpass', 'N/A')} Hz")
        
        # Annotations preview
        print(f"\n[*] Annotations Preview (first 5 epochs):")
        print(annotations.head().to_string(index=False))
        
        # Stage distribution
        stage_counts = annotations['stage'].value_counts().sort_index()
        print(f"\n[*] Stage Distribution:")
        for stage, count in stage_counts.items():
            stage_name = BOASSleepStageMapper.to_string(stage)
            pct = 100 * count / len(annotations)
            print(f"   {stage} ({stage_name:5}): {count:4} epochs ({pct:5.1f}%)")
        
        # Human vs AI agreement (if both available)
        if 'stage_hum' in annotations.columns and 'stage_ai' in annotations.columns:
            agreement = (annotations['stage_hum'] == annotations['stage_ai']).mean()
            print(f"\n[AI] Human vs AI Label Agreement: {agreement:.1%}")
        
        print(f"{'='*60}\n")
    
    def load_annotations(self, annotation_path: Path) -> pd.DataFrame:
        """
        Load sleep stage annotations from BOAS events file.
        
        Expected columns: onset, duration, begsample, endsample, offset, stage_hum, stage_ai
        
        Args:
            annotation_path: Path to events.txt file
        
        Returns:
            DataFrame with columns: onset, duration, stage_hum, stage_ai, is_valid
        """
        try:
            # BOAS uses tab-separated format
            df = pd.read_csv(annotation_path, sep='\t')
        except Exception as e:
            raise ValueError(f"Failed to parse annotation file {annotation_path}: {e}")
        
        # Verify required columns
        required_cols = ['onset', 'duration', 'stage_hum', 'stage_ai']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create clean dataframe
        annotations = pd.DataFrame({
            'onset': df['onset'].astype(float),
            'duration': df['duration'].astype(float),
            'stage_hum': df['stage_hum'].astype(int),
            'stage_ai': df['stage_ai'].astype(int),
        })
        
        # Determine which label column to use
        label_col = 'stage_hum' if self.use_human_labels else 'stage_ai'
        annotations['stage'] = annotations[label_col]
        
        # Mark valid epochs (exclude disconnections=8 and artifacts=-2)
        annotations['is_valid'] = annotations['stage'].apply(BOASSleepStageMapper.is_valid)
        
        # Filter to valid epochs
        valid_annotations = annotations[annotations['is_valid']].copy()
        n_filtered = len(annotations) - len(valid_annotations)
        
        if n_filtered > 0:
            logger.info(f"Filtered {n_filtered} invalid epochs "
                       f"(disconnections/artifacts)")
        
        # Log stage distribution
        stage_counts = valid_annotations['stage'].value_counts().sort_index()
        logger.debug(f"Stage distribution: {dict(stage_counts)}")
        
        return valid_annotations.reset_index(drop=True)
    
    def load_raw(self, edf_path: Path, preload: Optional[bool] = None) -> mne.io.Raw:
        """
        Load raw PSG EEG data from EDF file.
        
        Args:
            edf_path: Path to PSG EDF file
            preload: Override class preload setting
        
        Returns:
            MNE Raw object
        """
        if preload is None:
            preload = self.preload
        
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose='ERROR')
            
            # Verify sampling rate
            actual_sfreq = raw.info['sfreq']
            if abs(actual_sfreq - 256.0) > 1.0:
                logger.warning(f"Unexpected sampling rate: {actual_sfreq} Hz "
                             f"(expected 256 Hz)")
            
            logger.info(f"Loaded {edf_path.name}: {raw.n_times} samples, "
                       f"{actual_sfreq} Hz, {len(raw.ch_names)} channels")
            return raw
        except Exception as e:
            logger.error(f"Failed to load EDF file {edf_path}: {e}")
            raise
    
    def select_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Select PSG channels from raw data (6 EEG + EOG + EMG = 8 channels).
        
        Handles channel naming variations in BOAS dataset:
        - Standard (subjects 1-102): PSG_EOG, PSG_EMG
        - Variant (subjects 103-128): PSG_EOGL/PSG_EOGR, PSG_EMG
        
        For EOG variants (EOGL/EOGR), we use EOGL as the single EOG channel
        to maintain consistent 8-channel output across all subjects.
        
        Args:
            raw: MNE Raw object
        
        Returns:
            Raw object with 8 selected channels (F3, F4, C3, C4, O1, O2, EOG, EMG)
        """
        available_channels = raw.ch_names
        logger.debug(f"Available channels: {available_channels}")
        
        # Define channel aliases for BOAS dataset variants
        # Key: target channel name, Value: list of possible names in order of preference
        channel_aliases = {
            'PSG_F3': ['PSG_F3'],
            'PSG_F4': ['PSG_F4'],
            'PSG_C3': ['PSG_C3'],
            'PSG_C4': ['PSG_C4'],
            'PSG_O1': ['PSG_O1'],
            'PSG_O2': ['PSG_O2'],
            'PSG_EOG': ['PSG_EOG', 'PSG_EOGL', 'PSG_EOGR'],  # EOG variants
            'PSG_EMG': ['PSG_EMG'],
        }
        
        found_channels = []
        channel_mapping = {}  # Maps found channel name to standard name
        
        for target in self.target_channels:
            # Get aliases for this target (or just the target itself)
            aliases = channel_aliases.get(target, [target])
            
            matched = False
            for alias in aliases:
                if alias in available_channels and alias not in found_channels:
                    found_channels.append(alias)
                    channel_mapping[alias] = target
                    if alias != target:
                        logger.info(f"Matched '{target}' to '{alias}' (alias)")
                    matched = True
                    break
            
            # If no alias matched, try fuzzy matching
            if not matched:
                target_clean = target.lower().replace('_', '').replace(' ', '')
                for avail in available_channels:
                    if avail in found_channels:
                        continue
                    avail_clean = avail.lower().replace('_', '').replace(' ', '')
                    if target_clean == avail_clean:
                        found_channels.append(avail)
                        channel_mapping[avail] = target
                        logger.info(f"Matched '{target}' to '{avail}' (fuzzy)")
                        matched = True
                        break
        
        if not found_channels:
            raise ValueError(f"None of target channels {self.target_channels} "
                           f"found in {available_channels}")
        
        if len(found_channels) < len(self.target_channels):
            missing = [t for t in self.target_channels 
                      if t not in channel_mapping.values()]
            logger.warning(f"Only found {len(found_channels)}/{len(self.target_channels)} "
                         f"target channels. Missing: {missing}")
        
        # Pick channels
        raw_picked = raw.copy().pick_channels(found_channels, ordered=False)
        
        # Rename channels to standard names (e.g., PSG_EOGL -> PSG_EOG)
        rename_map = {actual: standard for actual, standard in channel_mapping.items() 
                     if actual != standard}
        if rename_map:
            raw_picked.rename_channels(rename_map)
            logger.info(f"Renamed channels: {rename_map}")
        
        logger.info(f"Selected {len(found_channels)} channels: {raw_picked.ch_names}")
        
        return raw_picked
    
    def resample_if_needed(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Resample to target_sfreq if it differs from current rate. Returns raw unchanged if already at target."""
        if self.target_sfreq is None:
            return raw
        
        current_sfreq = raw.info['sfreq']
        if abs(current_sfreq - self.target_sfreq) < 0.01:
            logger.info(f"Already at target sampling rate: {current_sfreq} Hz")
            return raw
        
        logger.info(f"Resampling from {current_sfreq} Hz to {self.target_sfreq} Hz")
        raw_resampled = raw.copy().resample(self.target_sfreq)
        return raw_resampled
    
    def load_subject(
        self,
        subject_id: str,
        apply_preprocessing: bool = True,
        apply_channel_selection: bool = True,
        apply_resampling: bool = False,
        verbose: bool = False
    ) -> Tuple[mne.io.Raw, pd.DataFrame, RecordingMetadata]:
        """
        Load complete subject data: PSG EEG + annotations + metadata.
        
        Args:
            subject_id: Subject ID
            apply_preprocessing: Deprecated - use apply_channel_selection/apply_resampling
            apply_channel_selection: Whether to select only 6 EEG channels
            apply_resampling: Whether to resample to target frequency
            verbose: Print detailed subject information (paths, metadata, stage distribution)
        
        Returns:
            Tuple of (raw_data, annotations, metadata)
        """
        # Get file paths
        psg_path, annotation_path = self.get_subject_files(subject_id)
        
        # Load raw data
        raw = self.load_raw(psg_path)
        
        # Load annotations (before channel selection for verbose output)
        annotations = self.load_annotations(annotation_path)
        
        # Verbose output before any processing
        if verbose:
            self.print_subject_info(subject_id, raw, annotations, psg_path, annotation_path)
        
        # Backward compatibility: only override if apply_preprocessing was explicitly set
        # and individual flags were not explicitly provided by the caller
        if apply_preprocessing and not apply_channel_selection:
            apply_channel_selection = True
        if apply_preprocessing and not apply_resampling:
            apply_resampling = True
        
        # Apply channel selection if requested
        if apply_channel_selection:
            raw = self.select_channels(raw)
        
        # Apply resampling if requested
        if apply_resampling:
            raw = self.resample_if_needed(raw)
        
        # Create metadata
        metadata = RecordingMetadata(
            subject_id=subject_id.replace('sub-', ''),
            filepath=psg_path,
            annotations_filepath=annotation_path,
            sampling_rate=raw.info['sfreq'],
            channels=raw.ch_names,
            duration_seconds=raw.times[-1],
            n_epochs=len(pd.read_csv(annotation_path, sep='\t')),  # Total before filtering
            n_valid_epochs=len(annotations),  # After filtering
            available_stages=sorted(annotations['stage'].unique().tolist()),
            human_labels=self.use_human_labels,
            ai_labels=not self.use_human_labels
        )
        
        logger.info(f"Loaded subject {metadata.subject_id}: "
                   f"{metadata.n_valid_epochs}/{metadata.n_epochs} valid epochs, "
                   f"{metadata.duration_seconds:.1f}s duration")
        
        return raw, annotations, metadata
    
    def validate_subject_data(
        self,
        raw: mne.io.Raw,
        annotations: pd.DataFrame,
        metadata: RecordingMetadata,
        check_signal_integrity: bool = True
    ) -> Dict[str, bool]:
        """
        Comprehensive validation of loaded data for common issues.
        
        Args:
            raw: MNE Raw object
            annotations: DataFrame with epoch annotations
            metadata: Recording metadata
            check_signal_integrity: If True, loads data and checks for NaN/flat channels (slower)
        
        Returns:
            Dictionary of validation results with detailed info
        """
        results = {}
        issues = []
        
        # ============================================================
        # ANNOTATION CHECKS
        # ============================================================
        
        # Check 1: Minimum number of valid epochs (after filtering)
        results['sufficient_epochs'] = metadata.n_valid_epochs >= 100
        if not results['sufficient_epochs']:
            issues.append(f"Only {metadata.n_valid_epochs} valid epochs (min: 100)")
        
        # Check 2: All 5 sleep stages present
        unique_stages = set(annotations['stage'].unique())
        results['all_stages_present'] = {0, 1, 2, 3, 4}.issubset(unique_stages)
        if not results['all_stages_present']:
            missing = {0, 1, 2, 3, 4} - unique_stages
            issues.append(f"Missing sleep stages: {missing}")
        
        # Check 3: Reasonable recording duration (at least 2 hours)
        results['sufficient_duration'] = metadata.duration_seconds >= 7200
        if not results['sufficient_duration']:
            issues.append(f"Recording only {metadata.duration_seconds/3600:.1f}h (min: 2h)")
        
        # Check 4: Not too many epochs filtered (>30% is suspicious)
        filter_rate = 1 - (metadata.n_valid_epochs / metadata.n_epochs)
        results['acceptable_filter_rate'] = filter_rate < 0.3
        if not results['acceptable_filter_rate']:
            issues.append(f"High filter rate: {filter_rate:.1%} epochs removed")
        
        # Check 5: Correct number of channels
        results['correct_channels'] = len(metadata.channels) >= 6
        if not results['correct_channels']:
            issues.append(f"Only {len(metadata.channels)} channels (expected ≥6)")
        
        # Check 6: Correct sampling rate
        expected_sfreq = self.target_sfreq or 256.0
        results['correct_sfreq'] = abs(metadata.sampling_rate - expected_sfreq) < 1.0
        if not results['correct_sfreq']:
            issues.append(f"Unexpected sfreq: {metadata.sampling_rate} Hz (expected {expected_sfreq})")
        
        # Check 7: Temporal continuity (no large gaps in annotations)
        if len(annotations) > 1:
            time_diffs = annotations['onset'].diff()[1:]
            expected_diff = self.epoch_duration
            large_gaps = (time_diffs > expected_diff * 1.5).sum()
            results['temporal_continuity'] = large_gaps == 0
            if not results['temporal_continuity']:
                issues.append(f"Found {large_gaps} gaps in annotation timeline")
        else:
            results['temporal_continuity'] = True
        
        # ============================================================
        # EDF-ANNOTATION ALIGNMENT CHECKS
        # ============================================================
        
        # Check 8: Annotations don't exceed EDF duration
        max_annotation_end = annotations['onset'].max() + self.epoch_duration
        results['annotations_within_edf'] = max_annotation_end <= metadata.duration_seconds + 1.0
        if not results['annotations_within_edf']:
            issues.append(f"Annotations extend beyond EDF: {max_annotation_end:.1f}s > {metadata.duration_seconds:.1f}s")
        
        # Check 9: Epoch boundaries are valid sample indices
        max_sample_in_edf = raw.n_times
        if 'begsample' in annotations.columns and 'endsample' in annotations.columns:
            max_annotation_sample = annotations['endsample'].max()
            results['sample_bounds_valid'] = max_annotation_sample <= max_sample_in_edf
            if not results['sample_bounds_valid']:
                issues.append(f"Annotation samples exceed EDF: {max_annotation_sample} > {max_sample_in_edf}")
        else:
            # Calculate expected sample bounds from onset/duration
            last_epoch_end_sample = int((annotations['onset'].max() + self.epoch_duration) * metadata.sampling_rate)
            results['sample_bounds_valid'] = last_epoch_end_sample <= max_sample_in_edf
            if not results['sample_bounds_valid']:
                issues.append(f"Calculated epoch end exceeds EDF samples")
        
        # ============================================================
        # SIGNAL INTEGRITY CHECKS (optional, requires loading data)
        # ============================================================
        
        if check_signal_integrity:
            # Load a sample of data for integrity checks
            if not raw.preload:
                # Load just first and last 30 seconds for quick check
                try:
                    data_start = raw.get_data(start=0, stop=int(30 * metadata.sampling_rate))
                    data_end = raw.get_data(start=max(0, raw.n_times - int(30 * metadata.sampling_rate)))
                    sample_data = np.concatenate([data_start, data_end], axis=1)
                except Exception:
                    sample_data = raw.get_data(stop=min(raw.n_times, int(60 * metadata.sampling_rate)))
            else:
                sample_data = raw.get_data()
            
            # Check 10: No NaN or Inf values
            has_nan = np.any(np.isnan(sample_data))
            has_inf = np.any(np.isinf(sample_data))
            results['no_invalid_values'] = not (has_nan or has_inf)
            if has_nan:
                issues.append("Signal contains NaN values")
            if has_inf:
                issues.append("Signal contains Inf values")
            
            # Check 11: No flat/dead channels (std ≈ 0)
            channel_stds = np.std(sample_data, axis=1)
            flat_channels = np.where(channel_stds < 1e-10)[0]
            results['no_flat_channels'] = len(flat_channels) == 0
            if len(flat_channels) > 0:
                flat_names = [raw.ch_names[i] for i in flat_channels]
                issues.append(f"Flat/dead channels detected: {flat_names}")
            
            # Check 12: Signal amplitude sanity (not extremely large values)
            max_amplitude = np.max(np.abs(sample_data))
            # EEG typically < 500µV, but raw EDF might be in different units
            results['reasonable_amplitude'] = max_amplitude < 1e6  # Very permissive
            if not results['reasonable_amplitude']:
                issues.append(f"Extreme signal amplitude: {max_amplitude:.2e}")
        
        # ============================================================
        # SUMMARY
        # ============================================================
        
        passed = sum(results.values())
        total = len(results)
        all_passed = passed == total
        
        # Log results
        if all_passed:
            logger.info(f"[OK] Data validation PASSED: {passed}/{total} checks")
        else:
            logger.warning(f"[!] Data validation: {passed}/{total} checks passed")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        # Add summary to results
        results['_summary'] = {
            'passed': passed,
            'total': total,
            'all_passed': all_passed,
            'issues': issues
        }
        
        return results


def main():
    """Example usage."""
    # Configuration
    BASE_PATH = r"C:\Users\DerHo\Desktop\Data"
    
    # Initialize loader
    loader = BOASDataLoader(
        base_path=BASE_PATH,
        target_channels=['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2'],
        target_sfreq=128.0,  # Downsample from 256 to 128 Hz
        epoch_duration=30.0,
        use_human_labels=True  # Use human consensus labels
    )
    
    # List available subjects
    subjects = loader.list_subjects()
    print(f"\nFound {len(subjects)} subjects")
    print(f"First 10: {subjects[:10]}")
    
    # Load one subject as example
    if subjects:
        test_subject = subjects[0]
        print(f"\n{'='*60}")
        print(f"Loading subject: {test_subject}")
        print(f"{'='*60}")
        
        raw, annotations, metadata = loader.load_subject(
            test_subject,
            apply_preprocessing=True
        )
        
        print(f"\nMetadata:")
        print(f"  Subject ID: {metadata.subject_id}")
        print(f"  Channels: {metadata.channels}")
        print(f"  Sampling Rate: {metadata.sampling_rate} Hz")
        print(f"  Duration: {metadata.duration_seconds:.1f}s")
        print(f"  Total Epochs: {metadata.n_epochs}")
        print(f"  Valid Epochs: {metadata.n_valid_epochs}")
        print(f"  Filter Rate: {(1 - metadata.n_valid_epochs/metadata.n_epochs)*100:.1f}%")
        print(f"  Available Stages: {[BOASSleepStageMapper.to_string(s) for s in metadata.available_stages]}")
        print(f"  Label Source: {'Human consensus' if metadata.human_labels else 'AI'}")
        
        print(f"\nAnnotations (first 10):")
        print(annotations[['onset', 'duration', 'stage_hum', 'stage_ai', 'stage']].head(10))
        
        print(f"\nStage distribution:")
        stage_dist = annotations['stage'].value_counts().sort_index()
        for stage, count in stage_dist.items():
            stage_name = BOASSleepStageMapper.to_string(stage)
            print(f"  {stage_name} ({stage}): {count} epochs ({count/len(annotations)*100:.1f}%)")
        
        # Validate
        print(f"\nValidation Results:")
        validation = loader.validate_subject_data(raw, annotations, metadata)
        for check, passed in validation.items():
            status = "[OK]" if passed else "[X]"
            print(f"  {status} {check}")


if __name__ == "__main__":
    main()
