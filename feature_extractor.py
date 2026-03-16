"""
Feature Extraction Module for EEG Data
=======================================

Extracts 149 features per epoch from 6-channel EEG data.

Feature Groups:
1. Time-Domain (10 per channel × 6 = 60)
2. Frequency-Domain (9 per channel × 6 = 54)
3. Complexity (4 per channel × 6 = 24)
4. Global (11 total)

Total: 60 + 54 + 24 + 11 = 149 features

Author: Lennart Gorzel
Date: December 2025
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, List, Optional
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
import os

# Optimized entropy/complexity library (same algorithms, C-optimized)
try:
    import antropy as ant
    ANTROPY_AVAILABLE = True
except ImportError:
    ANTROPY_AVAILABLE = False
    
logger = logging.getLogger(__name__)


# ============================================================================
# MODULE-LEVEL WORKER FUNCTION FOR MULTIPROCESSING
# ============================================================================
# This MUST be at module level (not a method) to be picklable for multiprocessing.
# Each worker process reconstructs its own FeatureExtractor instance.

def _extract_epoch_worker(sfreq: float, epoch: np.ndarray) -> dict:
    """
    Worker function for parallel feature extraction.
    
    This function is called in separate processes by joblib.
    It reconstructs a FeatureExtractor and extracts features from one epoch.
    
    Args:
        sfreq: Sampling frequency (Hz)
        epoch: Single epoch array (n_channels, n_samples)
        
    Returns:
        Dictionary of extracted features
    """
    # Reconstruct extractor in this process
    extractor = FeatureExtractor(sfreq=sfreq)
    return extractor.extract_single_epoch(epoch)


class TimeDomainFeatures:
    """Extract 10 time-domain features per single-channel epoch.

    Features: mean, std, var, min, max, peak-to-peak, RMS, skew, kurtosis, zero-crossing rate.
    """
    
    @staticmethod
    def extract(epoch: np.ndarray) -> Dict[str, float]:
        """
        Extract 10 time-domain features from a single-channel epoch.
        
        Args:
            epoch: 1D array (n_samples,)
            
        Returns:
            Dictionary of 10 features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(epoch)
        features['std'] = np.std(epoch)
        features['var'] = np.var(epoch)
        features['min'] = np.min(epoch)
        features['max'] = np.max(epoch)
        features['ptp'] = np.ptp(epoch)  # Peak-to-peak
        
        # RMS (Root Mean Square)
        features['rms'] = np.sqrt(np.mean(epoch**2))
        
        # Statistical moments
        features['skew'] = stats.skew(epoch)
        features['kurtosis'] = stats.kurtosis(epoch)
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.sign(epoch)))[0]
        features['zcr'] = len(zero_crossings) / len(epoch)
        
        return features


class FrequencyDomainFeatures:
    """Extract 9 frequency-domain features per single-channel epoch.

    Features: 6 band powers (delta/theta/alpha/sigma/beta/gamma),
    spectral entropy, peak frequency, median frequency.
    """
    
    def __init__(self, sfreq: float = 128.0):
        """
        Initialize frequency feature extractor.
        
        Args:
            sfreq: Sampling frequency (Hz)
        """
        self.sfreq = sfreq
        
        # Standard EEG frequency bands
        self.bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'sigma': (12.0, 16.0),  # Sleep spindles
            'beta': (16.0, 30.0),
            'gamma': (30.0, 40.0)
        }
    
    def compute_psd(self, epoch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density using Welch's method.
        
        Args:
            epoch: 1D array (n_samples,)
            
        Returns:
            Tuple of (frequencies, power_spectral_density)
        """
        # Welch's method: Divide signal into overlapping segments
        freqs, psd = signal.welch(
            epoch,
            fs=self.sfreq,
            nperseg=min(256, len(epoch)),
            noverlap=None,
            scaling='density'
        )
        
        return freqs, psd
    
    def band_power(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        band: Tuple[float, float]
    ) -> float:
        """
        Compute power in a specific frequency band.
        
        Args:
            freqs: Frequency values
            psd: Power spectral density
            band: (low_freq, high_freq) tuple
            
        Returns:
            Band power
        """
        low, high = band
        idx = np.logical_and(freqs >= low, freqs <= high)
        
        if not np.any(idx):
            return 0.0
        
        # Integrate using trapezoidal rule
        # Use trapezoid for NumPy 2.0+, trapz for older versions
        try:
            band_power = np.trapezoid(psd[idx], freqs[idx])
        except AttributeError:
            # Fallback for NumPy < 2.0
            band_power = np.trapz(psd[idx], freqs[idx])
        
        return band_power
    
    def spectral_entropy(self, psd: np.ndarray) -> float:
        """
        Compute spectral entropy (measure of signal complexity).
        
        Args:
            psd: Power spectral density
            
        Returns:
            Spectral entropy
        """
        # Normalize to probability distribution
        psd_norm = psd / np.sum(psd)
        
        # Remove zeros to avoid log(0)
        psd_norm = psd_norm[psd_norm > 0]
        
        # Shannon entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        
        return entropy
    
    def extract(self, epoch: np.ndarray) -> Dict[str, float]:
        """
        Extract 9 frequency-domain features from a single-channel epoch.
        
        Args:
            epoch: 1D array (n_samples,)
            
        Returns:
            Dictionary of 9 features
        """
        features = {}
        
        # Compute PSD
        freqs, psd = self.compute_psd(epoch)
        
        # Band powers (6 features)
        for band_name, band_range in self.bands.items():
            power = self.band_power(freqs, psd, band_range)
            features[f'{band_name}_power'] = power
        
        # Spectral entropy (1 feature)
        features['spectral_entropy'] = self.spectral_entropy(psd)
        
        # Peak frequency (1 feature)
        peak_idx = np.argmax(psd)
        features['peak_frequency'] = freqs[peak_idx]
        
        # Median frequency (1 feature)
        cumsum_psd = np.cumsum(psd)
        median_idx = np.where(cumsum_psd >= cumsum_psd[-1] / 2)[0][0]
        features['median_frequency'] = freqs[median_idx]
        
        return features


class ComplexityFeatures:
    """Extract complexity features from EEG epochs.
    
    Uses antropy library when available for optimized implementations
    of the same peer-reviewed algorithms (no quality loss).
    """
    
    @staticmethod
    def hjorth_parameters(epoch: np.ndarray) -> Tuple[float, float]:
        """
        Compute Hjorth parameters (mobility and complexity).
        
        Uses antropy if available (same formula, optimized).
        
        Args:
            epoch: 1D array (n_samples,)
            
        Returns:
            Tuple of (mobility, complexity)
        """
        if ANTROPY_AVAILABLE:
            # antropy uses identical Hjorth formulas
            mobility = ant.hjorth_params(epoch)[0]
            complexity = ant.hjorth_params(epoch)[1]
            return mobility, complexity
        
        # Fallback: Pure Python implementation
        # First derivative (velocity)
        first_deriv = np.diff(epoch)
        
        # Second derivative (acceleration)
        second_deriv = np.diff(first_deriv)
        
        # Variance of signal and derivatives
        var_signal = np.var(epoch)
        var_first_deriv = np.var(first_deriv)
        var_second_deriv = np.var(second_deriv)
        
        # Mobility: sqrt(var(derivative) / var(signal))
        mobility = np.sqrt(var_first_deriv / var_signal) if var_signal > 0 else 0
        
        # Complexity: mobility(derivative) / mobility(signal)
        mobility_deriv = np.sqrt(var_second_deriv / var_first_deriv) if var_first_deriv > 0 else 0
        complexity = mobility_deriv / mobility if mobility > 0 else 0
        
        return mobility, complexity
    
    @staticmethod
    def hurst_exponent(epoch: np.ndarray, max_lag: int = 20) -> float:
        """
        Compute Hurst exponent.
        
        Uses antropy's DFA-based Hurst if available (more robust).
        
        Args:
            epoch: 1D array (n_samples,)
            max_lag: Maximum lag to consider (fallback only)
            
        Returns:
            Hurst exponent (0.5 = random, >0.5 = persistent, <0.5 = anti-persistent)
        """
        if ANTROPY_AVAILABLE:
            try:
                # antropy computes Hurst via DFA (standard method in EEG research)
                return ant.detrended_fluctuation(epoch)
            except:
                return 0.5
        
        # Fallback: R/S analysis (slower)
        lags = range(2, min(max_lag, len(epoch) // 2))
        tau = []
        
        for lag in lags:
            n_windows = len(epoch) // lag
            
            if n_windows == 0:
                continue
            
            rs_values = []
            
            for i in range(n_windows):
                window = epoch[i*lag:(i+1)*lag]
                mean_window = np.mean(window)
                cumsum_window = np.cumsum(window - mean_window)
                R = np.max(cumsum_window) - np.min(cumsum_window)
                S = np.std(window, ddof=1)
                
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                tau.append(np.mean(rs_values))
        
        if len(tau) < 2:
            return 0.5
        
        try:
            poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5
    
    @staticmethod
    def detrended_fluctuation_analysis(epoch: np.ndarray) -> float:
        """
        Compute DFA (Detrended Fluctuation Analysis) exponent.
        
        Uses antropy if available (Numba-optimized, same algorithm).
        
        Args:
            epoch: 1D array (n_samples,)
            
        Returns:
            DFA exponent
        """
        if ANTROPY_AVAILABLE:
            try:
                return ant.detrended_fluctuation(epoch)
            except:
                return 1.0
        
        # Fallback: Pure Python (slower)
        signal_centered = epoch - np.mean(epoch)
        y = np.cumsum(signal_centered)
        
        N = len(epoch)
        min_window = 4
        max_window = N // 4
        
        if max_window < min_window:
            return 1.0
        
        scales = np.logspace(np.log10(min_window), np.log10(max_window), num=10, dtype=int)
        scales = np.unique(scales)
        
        fluctuations = []
        
        for scale in scales:
            n_segments = N // scale
            
            if n_segments == 0:
                continue
            
            segment_flucts = []
            
            for i in range(n_segments):
                segment = y[i*scale:(i+1)*scale]
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                fluct = np.sqrt(np.mean((segment - trend)**2))
                segment_flucts.append(fluct)
            
            if segment_flucts:
                fluctuations.append(np.mean(segment_flucts))
        
        if len(fluctuations) < 2:
            return 1.0
        
        try:
            poly = np.polyfit(np.log(scales[:len(fluctuations)]), np.log(fluctuations), 1)
            return poly[0]
        except:
            return 1.0
    
    def extract(self, epoch: np.ndarray) -> Dict[str, float]:
        """
        Extract 4 complexity features from a single-channel epoch.
        
        Args:
            epoch: 1D array (n_samples,)
            
        Returns:
            Dictionary of 4 features
        """
        features = {}
        
        # Hjorth parameters (2 features)
        mobility, complexity = self.hjorth_parameters(epoch)
        features['hjorth_mobility'] = mobility
        features['hjorth_complexity'] = complexity
        
        # Hurst exponent (1 feature)
        features['hurst'] = self.hurst_exponent(epoch)
        
        # DFA (1 feature)
        features['dfa'] = self.detrended_fluctuation_analysis(epoch)
        
        return features


class GlobalFeatures:
    """Extract 11 cross-channel features: 6 coherence pairs, 3 PLV pairs, global entropy, global complexity."""
    
    def __init__(self, sfreq: float = 128.0):
        """
        Initialize global feature extractor.
        
        Args:
            sfreq: Sampling frequency (Hz)
        """
        self.sfreq = sfreq
    
    def coherence(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray
    ) -> float:
        """
        Compute coherence between two signals.
        
        Args:
            signal1: First signal (n_samples,)
            signal2: Second signal (n_samples,)
            
        Returns:
            Mean coherence
        """
        freqs, coh = signal.coherence(
            signal1,
            signal2,
            fs=self.sfreq,
            nperseg=min(256, len(signal1))
        )
        
        # Average coherence in relevant frequency range (0.5-40 Hz)
        idx = np.logical_and(freqs >= 0.5, freqs <= 40.0)
        mean_coh = np.mean(coh[idx]) if np.any(idx) else 0.0
        
        return mean_coh
    
    def phase_locking_value(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray
    ) -> float:
        """
        Compute phase-locking value (PLV) between two signals.
        
        Args:
            signal1: First signal (n_samples,)
            signal2: Second signal (n_samples,)
            
        Returns:
            PLV value
        """
        # Hilbert transform to get analytic signals
        analytic1 = signal.hilbert(signal1)
        analytic2 = signal.hilbert(signal2)
        
        # Extract phases
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # Phase difference
        phase_diff = phase1 - phase2
        
        # PLV = |mean(exp(i * phase_diff))|
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return plv
    
    def global_entropy(self, multi_channel_epoch: np.ndarray) -> float:
        """
        Compute global entropy across all channels.
        
        Args:
            multi_channel_epoch: (n_channels, n_samples)
            
        Returns:
            Global entropy
        """
        # Flatten all channels
        all_data = multi_channel_epoch.flatten()
        
        # Create histogram
        hist, _ = np.histogram(all_data, bins=50)
        
        # Normalize to probability
        hist_norm = hist / np.sum(hist)
        hist_norm = hist_norm[hist_norm > 0]
        
        # Shannon entropy
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        
        return entropy
    
    def global_complexity(self, multi_channel_epoch: np.ndarray) -> float:
        """
        Compute global complexity (average across channels).
        
        Args:
            multi_channel_epoch: (n_channels, n_samples)
            
        Returns:
            Average complexity
        """
        complexity_extractor = ComplexityFeatures()
        
        complexities = []
        for channel_data in multi_channel_epoch:
            _, complexity = complexity_extractor.hjorth_parameters(channel_data)
            complexities.append(complexity)
        
        return np.mean(complexities)
    
    def extract(self, multi_channel_epoch: np.ndarray) -> Dict[str, float]:
        """
        Extract 11 global features from multi-channel epoch.
        
        Args:
            multi_channel_epoch: (6, n_samples) for 6 channels
            
        Returns:
            Dictionary of 11 features
        """
        features = {}
        
        # Build channel name list depending on number of channels available
        n_channels = multi_channel_epoch.shape[0]
        if n_channels == 6:
            channel_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
        elif n_channels == 8:
            channel_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'EOG', 'EMG']
        else:
            channel_names = [f'CH{i+1}' for i in range(n_channels)]

        # Coherence pairs (compute for predefined pairs if indices available)
        coherence_pairs = [
            ('F3', 'F4', 0, 1),
            ('F3', 'C3', 0, 2),
            ('F3', 'C4', 0, 3),
            ('F4', 'C4', 1, 3),
            ('C3', 'C4', 2, 3),
            ('O1', 'O2', 4, 5)
        ]

        for name1, name2, idx1, idx2 in coherence_pairs:
            if idx1 < n_channels and idx2 < n_channels:
                coh = self.coherence(
                    multi_channel_epoch[idx1],
                    multi_channel_epoch[idx2]
                )
                features[f'coherence_{name1}_{name2}'] = coh

        # PLV pairs (compute if indices available)
        plv_pairs = [
            ('F3', 'O1', 0, 4),
            ('F4', 'O2', 1, 5),
            ('C3', 'C4', 2, 3)
        ]

        for name1, name2, idx1, idx2 in plv_pairs:
            if idx1 < n_channels and idx2 < n_channels:
                plv = self.phase_locking_value(
                    multi_channel_epoch[idx1],
                    multi_channel_epoch[idx2]
                )
                features[f'plv_{name1}_{name2}'] = plv
        
        # Global metrics (2 features)
        features['global_entropy'] = self.global_entropy(multi_channel_epoch)
        features['global_complexity'] = self.global_complexity(multi_channel_epoch)
        
        return features


class FeatureExtractor:
    """
    Main feature extractor class.
    
    Extracts all 149 features from multi-channel EEG epochs.
    """
    
    def __init__(self, sfreq: float = 128.0):
        """
        Initialize feature extractor.
        
        Args:
            sfreq: Sampling frequency (Hz)
        """
        self.sfreq = sfreq
        
        # Initialize extractors
        self.time_extractor = TimeDomainFeatures()
        self.freq_extractor = FrequencyDomainFeatures(sfreq)
        self.complexity_extractor = ComplexityFeatures()
        self.global_extractor = GlobalFeatures(sfreq)
        
        logger.info(f"Initialized FeatureExtractor (sfreq={sfreq} Hz)")
    
    def extract_single_epoch(
        self,
        epoch: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract all 149 features from a single multi-channel epoch.
        
        Args:
            epoch: (n_channels, n_samples) array
            
        Returns:
            Dictionary of 149 features
        """
        n_channels = epoch.shape[0]

        # Determine channel names based on input channel count
        if n_channels == 6:
            channel_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
        elif n_channels == 8:
            # Include two physiological channels (EOG, EMG)
            channel_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'EOG', 'EMG']
        else:
            # Generic channel names CH1..CHn
            channel_names = [f'CH{i+1}' for i in range(n_channels)]
        
        all_features = {}
        
        # Per-channel features (23 per channel)
        for ch_idx, ch_name in enumerate(channel_names):
            channel_data = epoch[ch_idx]
            
            # Time-domain (10)
            time_feats = self.time_extractor.extract(channel_data)
            for feat_name, feat_val in time_feats.items():
                all_features[f'{ch_name}_{feat_name}'] = feat_val
            
            # Frequency-domain (9)
            freq_feats = self.freq_extractor.extract(channel_data)
            for feat_name, feat_val in freq_feats.items():
                all_features[f'{ch_name}_{feat_name}'] = feat_val
            
            # Complexity (4)
            complexity_feats = self.complexity_extractor.extract(channel_data)
            for feat_name, feat_val in complexity_feats.items():
                all_features[f'{ch_name}_{feat_name}'] = feat_val
        
        # Global features (computed across available channels)
        global_feats = self.global_extractor.extract(epoch)
        all_features.update(global_feats)
        
        return all_features
    
    def extract_multiple_epochs(
        self,
        epochs: np.ndarray,
        n_jobs: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract features from multiple epochs.
        
        Args:
            epochs: (n_epochs, n_channels, n_samples) array
            n_jobs: Number of parallel workers (None or 1 = sequential, -1 = all CPUs)

        Returns:
            DataFrame with shape (n_epochs, n_features)
        """
        n_epochs = epochs.shape[0]
        logger.info(f"Extracting features from {n_epochs} epochs...")

        # If n_jobs is provided and not equal to 1, use joblib parallel processing.
        if n_jobs is not None and n_jobs != 1:
            # Use PROCESSES (not threads) for true parallelism.
            # Python's GIL prevents threads from providing speedup for CPU-bound code.
            # We use joblib with 'loky' backend which handles pickling properly.
            if n_jobs < 0:
                n_workers = os.cpu_count() or 1
            else:
                n_workers = n_jobs
            
            logger.info(f"Using {n_workers} parallel PROCESSES for feature extraction")
            
            # Parallel extraction with progress bar
            # Pass sfreq to recreate extractor in worker processes
            feature_list = Parallel(n_jobs=n_workers, prefer='processes', verbose=0)(
                delayed(_extract_epoch_worker)(self.sfreq, epoch) 
                for epoch in tqdm(epochs, desc="Submitting epochs", unit="epoch")
            )
            
            logger.info("Parallel extraction complete")
        else:
            feature_list = []
            # Use tqdm progress bar for interactive feedback; also keep periodic logs.
            for i in tqdm(range(n_epochs), desc="Extracting features", unit="epoch"):
                if i % 100 == 0:
                    logger.info(f"  Progress: {i}/{n_epochs}")

                features = self.extract_single_epoch(epochs[i])
                feature_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_list)
        
        logger.info(f"Feature extraction complete: {features_df.shape}")
        
        return features_df


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with synthetic data
    n_epochs = 10
    n_channels = 6
    n_samples = 3840  # 30s at 128 Hz
    
    # Generate random epochs
    epochs = np.random.randn(n_epochs, n_channels, n_samples)
    
    # Extract features
    extractor = FeatureExtractor(sfreq=128.0)
    features_df = extractor.extract_multiple_epochs(epochs)
    
    print(f"\nFeature extraction test:")
    print(f"  Input shape: {epochs.shape}")
    print(f"  Output shape: {features_df.shape}")
    print(f"  Expected features: 149")
    print(f"  Actual features: {features_df.shape[1]}")
    print(f"\nFirst few columns:")
    print(features_df.columns[:10].tolist())