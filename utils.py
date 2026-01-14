"""
Utility Functions for ML Caching Pipeline
=========================================

Common helper functions used across modules.

Author: Lennart Gorzel
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import sys

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


def create_output_directories(base_dir: Path):
    """
    Create standard output directory structure.
    
    Args:
        base_dir: Base output directory
    """
    directories = [
        base_dir,
        base_dir / "preprocessed",
        base_dir / "features",
        base_dir / "models",
        base_dir / "results",
        base_dir / "logs",
        base_dir / "figures"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


def save_dataframe(df: pd.DataFrame, filepath: Path, format: str = 'csv'):
    """
    Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        format: File format ('csv', 'parquet', 'pickle')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved DataFrame to {filepath} ({format} format)")


def load_dataframe(filepath: Path, format: str = None) -> pd.DataFrame:
    """
    Load DataFrame from file.
    
    Args:
        filepath: Input file path
        format: File format (auto-detect if None)
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Auto-detect format
    if format is None:
        suffix = filepath.suffix.lower()
        if suffix == '.csv':
            format = 'csv'
        elif suffix == '.parquet':
            format = 'parquet'
        elif suffix in ['.pkl', '.pickle']:
            format = 'pickle'
        else:
            raise ValueError(f"Cannot auto-detect format for: {suffix}")
    
    # Load
    if format == 'csv':
        df = pd.read_csv(filepath)
    elif format == 'parquet':
        df = pd.read_parquet(filepath)
    elif format == 'pickle':
        df = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded DataFrame from {filepath} ({format} format)")
    return df


def save_numpy_array(arr: np.ndarray, filepath: Path):
    """
    Save NumPy array to file.
    
    Args:
        arr: Array to save
        filepath: Output file path (.npy)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(filepath, arr)
    logger.info(f"Saved array {arr.shape} to {filepath}")


def load_numpy_array(filepath: Path) -> np.ndarray:
    """
    Load NumPy array from file.
    
    Args:
        filepath: Input file path (.npy)
        
    Returns:
        Loaded array
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    arr = np.load(filepath)
    logger.info(f"Loaded array {arr.shape} from {filepath}")
    return arr


def save_json(data: Dict, filepath: Path):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path (.json)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Input file path (.json)
        
    Returns:
        Loaded dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON from {filepath}")
    return data


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def format_bytes(bytes_size: int) -> str:
    """
    Format bytes into human-readable size string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, desc: str = "Progress"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            desc: Description of operation
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        self._print_progress()
    
    def _print_progress(self):
        """Print progress bar."""
        percent = (self.current / self.total) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Estimate time remaining
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = format_time(eta)
        else:
            eta_str = "?"
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * self.current / self.total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\r{self.desc}: [{bar}] {percent:.1f}% ({self.current}/{self.total}) ETA: {eta_str}", end='')
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def close(self):
        """Close progress tracker."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.desc} complete in {format_time(elapsed)}")


def validate_epochs_labels(
    epochs: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate that epochs and labels match and are valid.
    
    Args:
        epochs: Epoch data (n_epochs, n_channels, n_samples)
        labels: Sleep stage labels (n_epochs,)
        
    Returns:
        Tuple of (validated_epochs, validated_labels)
        
    Raises:
        ValueError: If validation fails
    """
    # Check shapes match
    if len(epochs) != len(labels):
        raise ValueError(f"Mismatch: {len(epochs)} epochs but {len(labels)} labels")
    
    # Check for NaN/Inf in epochs
    if np.any(np.isnan(epochs)) or np.any(np.isinf(epochs)):
        logger.warning("Found NaN/Inf in epochs, filtering...")
        valid_mask = ~(np.isnan(epochs).any(axis=(1,2)) | np.isinf(epochs).any(axis=(1,2)))
        epochs = epochs[valid_mask]
        labels = labels[valid_mask]
        logger.info(f"Filtered to {len(epochs)} valid epochs")
    
    # Check labels are in valid range
    valid_stages = {0, 1, 2, 3, 4}
    unique_labels = set(np.unique(labels))
    invalid = unique_labels - valid_stages
    
    if invalid:
        raise ValueError(f"Invalid sleep stage labels found: {invalid}")
    
    logger.debug(f"Validated {len(epochs)} epochs with labels: {unique_labels}")
    
    return epochs, labels


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: Class labels
        
    Returns:
        Dictionary mapping class → weight
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    # Inverse frequency weighting
    weights = {}
    for cls, count in zip(unique, counts):
        weights[int(cls)] = total / (len(unique) * count)
    
    logger.debug(f"Class weights: {weights}")
    return weights


def split_train_test(
    features: np.ndarray,
    labels: np.ndarray,
    test_subject_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test sets based on subject mask.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Labels (n_samples,)
        test_subject_mask: Boolean mask for test samples
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    train_mask = ~test_subject_mask
    
    X_train = features[train_mask]
    X_test = features[test_subject_mask]
    y_train = labels[train_mask]
    y_test = labels[test_subject_mask]
    
    logger.info(f"Split: {len(X_train)} train, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging("INFO")
    
    # Test progress tracker
    print("\nTesting progress tracker:")
    tracker = ProgressTracker(100, "Processing")
    for i in range(100):
        import time
        time.sleep(0.01)  # Simulate work
        tracker.update()
    tracker.close()
    
    # Test time formatting
    print(f"\nTime formatting:")
    print(f"  {format_time(65)} = 1m 5s")
    print(f"  {format_time(3665)} = 1h 1m 5s")
    
    # Test byte formatting
    print(f"\nByte formatting:")
    print(f"  {format_bytes(1024)} = 1.00 KB")
    print(f"  {format_bytes(1024**2)} = 1.00 MB")
    print(f"  {format_bytes(1024**3)} = 1.00 GB")
