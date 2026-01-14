"""
Preprocessing Module for EEG Data
==================================

Handles signal filtering, resampling, and epoch extraction for BOAS dataset.

Pipeline:
1. Load raw EEG (256 Hz, 6 channels)
2. Apply bandpass filter (0.5-40 Hz)
3. Apply notch filter (50 Hz power line)
4. Downsample to target frequency (128 Hz)
5. Extract 30-second epochs
6. Align with annotations
7. Filter invalid epochs

Author: Lennart Gorzel
Date: December 2025
"""

import numpy as np
import pandas as pd
import mne
from typing import Tuple, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SignalPreprocessor:
    """
    Handles EEG signal preprocessing.
    
    Operations:
    - Bandpass filtering
    - Notch filtering (power line noise)
    - Resampling/downsampling
    - Epoch extraction
    """
    
    def __init__(
        self,
        bandpass_low: float = 0.5,
        bandpass_high: float = 40.0,
        notch_freq: float = 50.0,
        target_sfreq: float = 128.0,
        epoch_duration: float = 30.0
    ):
        """
        Initialize preprocessor.
        
        Args:
            bandpass_low: Low cutoff frequency (Hz)
            bandpass_high: High cutoff frequency (Hz)
            notch_freq: Notch filter frequency (Hz) - 50 (EU) or 60 (US)
            target_sfreq: Target sampling rate (Hz)
            epoch_duration: Epoch duration (seconds)
        """
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.target_sfreq = target_sfreq
        self.epoch_duration = epoch_duration
        
        # Validation
        assert target_sfreq >= 2 * bandpass_high, (
            f"Nyquist violation: {target_sfreq} Hz < 2 × {bandpass_high} Hz"
        )
        
        logger.info(f"Initialized SignalPreprocessor:")
        logger.info(f"  Bandpass: {bandpass_low}-{bandpass_high} Hz")
        logger.info(f"  Notch: {notch_freq} Hz")
        logger.info(f"  Target sampling rate: {target_sfreq} Hz")
        logger.info(f"  Epoch duration: {epoch_duration} s")
    
    def apply_bandpass_filter(
        self,
        raw: mne.io.Raw,
        verbose: str = 'ERROR'
    ) -> mne.io.Raw:
        """
        Apply bandpass filter to remove DC drift and high-frequency noise.
        
        Args:
            raw: MNE Raw object
            verbose: Verbosity level
            
        Returns:
            Filtered Raw object
        """
        logger.info(f"Applying bandpass filter: {self.bandpass_low}-{self.bandpass_high} Hz")
        
        # MNE's filter function applies FIR filter
        raw_filtered = raw.copy().filter(
            l_freq=self.bandpass_low,
            h_freq=self.bandpass_high,
            picks='eeg',
            method='fir',
            fir_design='firwin',
            verbose=verbose
        )
        
        return raw_filtered
    
    def apply_notch_filter(
        self,
        raw: mne.io.Raw,
        verbose: str = 'ERROR'
    ) -> mne.io.Raw:
        """
        Apply notch filter to remove power line interference.
        
        Args:
            raw: MNE Raw object
            verbose: Verbosity level
            
        Returns:
            Notch-filtered Raw object
        """
        logger.info(f"Applying notch filter: {self.notch_freq} Hz")
        
        raw_notched = raw.copy().notch_filter(
            freqs=self.notch_freq,
            picks='eeg',
            method='fir',
            verbose=verbose
        )
        
        return raw_notched
    
    def resample(
        self,
        raw: mne.io.Raw,
        verbose: str = 'ERROR'
    ) -> mne.io.Raw:
        """
        Resample to target sampling rate.
        
        Args:
            raw: MNE Raw object
            verbose: Verbosity level
            
        Returns:
            Resampled Raw object
        """
        current_sfreq = raw.info['sfreq']
        
        if abs(current_sfreq - self.target_sfreq) < 0.01:
            logger.info(f"Already at target sampling rate: {current_sfreq} Hz")
            return raw
        
        logger.info(f"Resampling: {current_sfreq} Hz → {self.target_sfreq} Hz")
        
        raw_resampled = raw.copy().resample(
            sfreq=self.target_sfreq,
            npad='auto',
            verbose=verbose
        )
        
        return raw_resampled
    
    def preprocess(
        self,
        raw: mne.io.Raw,
        apply_filter: bool = True,
        apply_notch: bool = True,
        apply_resample: bool = True
    ) -> mne.io.Raw:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            raw: MNE Raw object
            apply_filter: Apply bandpass filter
            apply_notch: Apply notch filter
            apply_resample: Apply resampling
            
        Returns:
            Preprocessed Raw object
        """
        logger.info("Starting preprocessing pipeline")
        
        # Load data into memory if not already loaded
        if not raw.preload:
            logger.info("Loading data into memory")
            raw.load_data()
        
        processed = raw.copy()
        
        # Apply filters
        if apply_filter:
            processed = self.apply_bandpass_filter(processed)
        
        if apply_notch:
            processed = self.apply_notch_filter(processed)
        
        # Resample
        if apply_resample:
            processed = self.resample(processed)
        
        logger.info("Preprocessing complete")
        return processed


class EpochExtractor:
    """
    Extracts fixed-length epochs from continuous EEG data.
    
    Handles:
    - Epoch extraction at specified intervals
    - Alignment with annotations
    - Validation of epoch quality
    """
    
    def __init__(
        self,
        epoch_duration: float = 30.0,
        target_sfreq: float = 128.0
    ):
        """
        Initialize epoch extractor.
        
        Args:
            epoch_duration: Duration of each epoch (seconds)
            target_sfreq: Sampling rate of processed data (Hz)
        """
        self.epoch_duration = epoch_duration
        self.target_sfreq = target_sfreq
        self.samples_per_epoch = int(epoch_duration * target_sfreq)
        
        logger.info(f"Initialized EpochExtractor:")
        logger.info(f"  Epoch duration: {epoch_duration} s")
        logger.info(f"  Samples per epoch: {self.samples_per_epoch}")
    
    def extract_epochs(
        self,
        raw: mne.io.Raw,
        annotations: 'pd.DataFrame') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract epochs aligned with annotations.
        
        Args:
            raw: Preprocessed MNE Raw object
            annotations: DataFrame with columns ['onset', 'duration', 'stage']
            
        Returns:
            Tuple of (epochs_data, labels)
            - epochs_data: (n_epochs, n_channels, n_samples)
            - labels: (n_epochs,) sleep stage integers
        """
        logger.info(f"Extracting {len(annotations)} epochs")
        
        # Get data
        data = raw.get_data()  # (n_channels, n_samples)
        sfreq = raw.info['sfreq']
        n_channels = data.shape[0]
        
        epochs_list = []
        labels_list = []
        
        for idx, row in annotations.iterrows():
            onset = row['onset']  # seconds
            stage = row['stage']
            
            # Convert onset to sample index
            start_sample = int(onset * sfreq)
            end_sample = start_sample + self.samples_per_epoch
            
            # Check if epoch is within bounds
            if end_sample > data.shape[1]:
                logger.warning(f"Epoch {idx} exceeds signal length, skipping")
                continue
            
            # Extract epoch
            epoch_data = data[:, start_sample:end_sample]
            
            # Verify shape
            if epoch_data.shape[1] != self.samples_per_epoch:
                logger.warning(f"Epoch {idx} has incorrect length: "
                             f"{epoch_data.shape[1]} != {self.samples_per_epoch}, skipping")
                continue
            
            epochs_list.append(epoch_data)
            labels_list.append(stage)
        
        if not epochs_list:
            raise ValueError("No valid epochs extracted")
        
        # Stack into arrays
        epochs_data = np.stack(epochs_list, axis=0)  # (n_epochs, n_channels, n_samples)
        labels = np.array(labels_list)  # (n_epochs,)
        
        logger.info(f"Extracted {len(epochs_data)} valid epochs")
        logger.info(f"  Shape: {epochs_data.shape}")
        logger.info(f"  Label distribution: {np.bincount(labels)}")
        
        return epochs_data, labels
    
    def validate_epoch_quality(
        self,
        epochs: np.ndarray,
        max_amplitude: float = 1000.0,  # µV - increased for BOAS data
        min_amplitude: float = 0.1  # µV - decreased threshold
    ) -> np.ndarray:
        """
        Validate epoch quality based on amplitude criteria.
        
        Args:
            epochs: Epoch data (n_epochs, n_channels, n_samples)
            max_amplitude: Maximum allowed amplitude (µV)
            min_amplitude: Minimum required amplitude (µV)
            
        Returns:
            Boolean mask (n_epochs,) - True = valid
        """
        n_epochs = epochs.shape[0]
        valid_mask = np.ones(n_epochs, dtype=bool)
        
        for i in range(n_epochs):
            epoch = epochs[i]
            
            # Check for excessive amplitude (artifact)
            max_val = np.abs(epoch).max()
            if max_val > max_amplitude:
                logger.debug(f"Epoch {i} rejected: amplitude {max_val:.2f} > {max_amplitude}")
                valid_mask[i] = False
                continue
            
            # Check for flat signal (disconnection)
            peak_to_peak = epoch.max() - epoch.min()
            if peak_to_peak < min_amplitude:
                logger.debug(f"Epoch {i} rejected: peak-to-peak {peak_to_peak:.2f} < {min_amplitude}")
                valid_mask[i] = False
                continue
        
        n_valid = valid_mask.sum()
        n_rejected = n_epochs - n_valid
        
        if n_rejected > 0:
            logger.info(f"Quality validation: {n_valid}/{n_epochs} epochs passed "
                       f"({n_rejected} rejected)")
        
        return valid_mask


def preprocess_subject(
    raw: mne.io.Raw,
    annotations: 'pd.DataFrame',
    config: 'PreprocessingConfig',
    validate_quality: bool = False  # NEW: Optional quality check
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for a single subject.
    
    Args:
        raw: Raw EEG data
        annotations: Sleep stage annotations
        config: Preprocessing configuration
        validate_quality: Whether to apply quality validation (default: False)
        
    Returns:
        Tuple of (epochs_data, labels)
    """
    # Initialize preprocessor
    preprocessor = SignalPreprocessor(
        bandpass_low=config.bandpass_low,
        bandpass_high=config.bandpass_high,
        notch_freq=config.notch_frequency,
        target_sfreq=config.target_sfreq,
        epoch_duration=config.epoch_duration
    )
    
    # Preprocess signal
    raw_processed = preprocessor.preprocess(raw)
    
    # Extract epochs
    extractor = EpochExtractor(
        epoch_duration=config.epoch_duration,
        target_sfreq=config.target_sfreq
    )
    
    epochs_data, labels = extractor.extract_epochs(raw_processed, annotations)
    
    # Validate quality (OPTIONAL)
    if validate_quality:
        valid_mask = extractor.validate_epoch_quality(epochs_data)
        
        # Filter to valid epochs only
        epochs_data = epochs_data[valid_mask]
        labels = labels[valid_mask]
        
        logger.info(f"Quality filtering: {len(epochs_data)} epochs kept")
    else:
        logger.info(f"Quality validation SKIPPED - keeping all {len(epochs_data)} epochs")
    
    logger.info(f"Preprocessing complete: {len(epochs_data)} valid epochs")
    
    return epochs_data, labels


# Example usage
if __name__ == "__main__":
    from data_loader_boas import BOASDataLoader
    from config import PreprocessingConfig
    
    # Setup
    logging.basicConfig(level=logging.INFO)
    
    # Load one subject
    loader = BOASDataLoader(
        base_path=r"C:\Users\DerHo\Desktop\Data",
        target_sfreq=None,  # Don't resample yet
        preload=True
    )
    
    subjects = loader.list_subjects()
    test_subject = subjects[0]
    
    print(f"\nTesting preprocessing on subject: {test_subject}")
    raw, annotations, metadata = loader.load_subject(test_subject, apply_preprocessing=False)
    
    # Preprocess
    config = PreprocessingConfig()
    epochs_data, labels = preprocess_subject(raw, annotations, config)
    
    print(f"\nResults:")
    print(f"  Epochs shape: {epochs_data.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {np.unique(labels)}")
    print(f"  Expected samples per epoch: {int(config.target_sfreq * config.epoch_duration)}")
    print(f"  Actual samples per epoch: {epochs_data.shape[2]}")
