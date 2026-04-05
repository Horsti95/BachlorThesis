"""
Cross-Validation Module for Sleep Stage Classification
=======================================================

Implements cross-validation strategies for thesis experiments:
- Leave-One-Subject-Out (LOSO) - Primary method for generalization testing
- K-Fold - Alternative for faster iteration

LOSO Rationale:
    Sleep patterns are highly individual. Training on 127 subjects and testing
    on 1 held-out subject provides realistic generalization estimates. This is
    the gold standard for subject-independent sleep staging research.

Implementation Details:
    - LOSO: 128 folds (one per subject)
    - Each fold: ~950 epochs test, ~120,000 epochs train
    - Total training runs: 128 × 27 configs = 3,456 (managed by training.py)

Author: Lennart Gorzel
Date: December 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Iterator, Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.model_selection import LeaveOneGroupOut, KFold
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CVFold:
    """
    Represents a single cross-validation fold.
    
    Attributes:
        fold_id: Fold identifier (0-indexed)
        train_indices: Indices into the full dataset for training
        test_indices: Indices into the full dataset for testing
        test_subject: Subject ID being held out (LOSO only)
        n_train: Number of training samples
        n_test: Number of test samples
    """
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_subject: Optional[str] = None
    
    @property
    def n_train(self) -> int:
        return len(self.train_indices)
    
    @property
    def n_test(self) -> int:
        return len(self.test_indices)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'fold_id': self.fold_id,
            'test_subject': self.test_subject,
            'n_train': self.n_train,
            'n_test': self.n_test,
        }


class LOSOCrossValidator:
    """
    Leave-One-Subject-Out Cross-Validator.
    
    For 128 subjects, this creates 128 folds where each fold:
    - Trains on 127 subjects (~120,000 epochs)
    - Tests on 1 held-out subject (~950 epochs)
    
    This is the gold standard for evaluating subject-independent
    sleep staging models, as it tests true generalization to unseen individuals.
    
    Usage:
        cv = LOSOCrossValidator()
        for fold in cv.split(X, y, subject_ids):
            X_train = X.iloc[fold.train_indices]
            X_test = X.iloc[fold.test_indices]
            # ... train and evaluate
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize LOSO cross-validator.
        
        Args:
            verbose: Whether to log detailed information
        """
        self.verbose = verbose
        self.n_folds_: int = 0
        self.subjects_: Optional[List[str]] = None
        self._logo = LeaveOneGroupOut()
    
    def get_n_splits(self, subject_ids: np.ndarray) -> int:
        """
        Get number of folds.
        
        Args:
            subject_ids: Array of subject IDs for each sample
            
        Returns:
            Number of folds (= number of unique subjects)
        """
        return len(np.unique(subject_ids))
    
    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        subject_ids: np.ndarray
    ) -> Iterator[CVFold]:
        """
        Generate train/test splits for LOSO cross-validation.
        
        Args:
            X: Feature DataFrame (n_samples, n_features)
            y: Labels (n_samples,)
            subject_ids: Subject ID for each sample (n_samples,)
            
        Yields:
            CVFold objects with train/test indices and metadata
        """
        self.subjects_ = list(np.unique(subject_ids))
        self.n_folds_ = len(self.subjects_)
        
        if self.verbose:
            logger.info(f"LOSO Cross-Validation: {self.n_folds_} folds for {len(X)} samples")
        
        # Use sklearn's LeaveOneGroupOut
        for fold_id, (train_idx, test_idx) in enumerate(
            self._logo.split(X, y, groups=subject_ids)
        ):
            # Identify the held-out subject
            test_subject = str(subject_ids[test_idx[0]])
            
            fold = CVFold(
                fold_id=fold_id,
                train_indices=train_idx,
                test_indices=test_idx,
                test_subject=test_subject
            )
            
            if self.verbose:
                logger.debug(
                    f"Fold {fold_id}: test subject={test_subject}, "
                    f"train={fold.n_train}, test={fold.n_test}"
                )
            
            yield fold
    
    def split_with_progress(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        subject_ids: np.ndarray,
        desc: str = "LOSO CV"
    ) -> Iterator[CVFold]:
        """
        Generate splits with a progress bar.
        
        Args:
            X: Feature DataFrame
            y: Labels
            subject_ids: Subject IDs
            desc: Progress bar description
            
        Yields:
            CVFold objects
        """
        n_folds = self.get_n_splits(subject_ids)
        
        for fold in tqdm(
            self.split(X, y, subject_ids),
            total=n_folds,
            desc=desc,
            unit="fold"
        ):
            yield fold


class KFoldCrossValidator:
    """
    K-Fold Cross-Validator.
    
    Standard K-fold that splits by samples, not subjects.
    Faster than LOSO but doesn't test subject-independent generalization.
    
    Use cases:
    - Quick iteration during development
    - When subject-independence is not required
    - Hyperparameter tuning (faster than full LOSO)
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        """
        Initialize K-Fold cross-validator.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._kfold = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
    
    def get_n_splits(self) -> int:
        """Get number of folds."""
        return self.n_splits
    
    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Iterator[CVFold]:
        """
        Generate train/test splits for K-Fold cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Labels
            
        Yields:
            CVFold objects with train/test indices
        """
        logger.info(f"K-Fold Cross-Validation: {self.n_splits} folds for {len(X)} samples")
        
        for fold_id, (train_idx, test_idx) in enumerate(self._kfold.split(X)):
            fold = CVFold(
                fold_id=fold_id,
                train_indices=train_idx,
                test_indices=test_idx,
                test_subject=None  # Not applicable for K-Fold
            )
            
            logger.debug(f"Fold {fold_id}: train={fold.n_train}, test={fold.n_test}")
            yield fold


class StratifiedGroupKFold:
    """
    Stratified K-Fold that respects subject groups.
    
    Ensures:
    1. Each subject appears in only one fold's test set
    2. Class distribution is approximately preserved across folds
    
    This is a middle ground between LOSO (too slow) and regular K-Fold
    (doesn't respect subject boundaries).
    """
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize Stratified Group K-Fold.
        
        Args:
            n_splits: Number of folds
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.random_state = random_state
    
    def split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        subject_ids: np.ndarray
    ) -> Iterator[CVFold]:
        """
        Generate stratified group splits.
        
        Args:
            X: Feature DataFrame
            y: Labels
            subject_ids: Subject IDs
            
        Yields:
            CVFold objects
        """
        logger.info(f"Stratified Group K-Fold: {self.n_splits} folds")
        
        # Get unique subjects and their class distributions
        subjects = np.unique(subject_ids)
        n_subjects = len(subjects)
        
        # Calculate mean label per subject as stratification target
        subject_labels = {}
        for subj in subjects:
            mask = subject_ids == subj
            subject_labels[subj] = y[mask].mean()  # Proxy for class distribution
        
        # Sort subjects by mean label for stratified splitting
        sorted_subjects = sorted(subjects, key=lambda s: subject_labels[s])

        # Assign subjects to folds in round-robin fashion
        # Note: NOT shuffling after sort, as that would defeat stratification.
        # The sorted round-robin ensures each fold gets a mix of class distributions.
        
        fold_assignments = {}
        for i, subj in enumerate(sorted_subjects):
            fold_assignments[subj] = i % self.n_splits
        
        # Generate folds
        for fold_id in range(self.n_splits):
            test_subjects = {s for s, f in fold_assignments.items() if f == fold_id}
            
            test_mask = np.isin(subject_ids, list(test_subjects))
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            # Get test subject list for reference
            test_subj_list = sorted(test_subjects)
            
            fold = CVFold(
                fold_id=fold_id,
                train_indices=train_idx,
                test_indices=test_idx,
                test_subject=','.join(map(str, test_subj_list[:3])) + ('...' if len(test_subj_list) > 3 else '')  # Summary
            )
            
            logger.debug(
                f"Fold {fold_id}: {len(test_subjects)} test subjects, "
                f"train={fold.n_train}, test={fold.n_test}"
            )
            yield fold


def create_cv_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    subject_ids: np.ndarray,
    method: str = 'loso',
    n_folds: Optional[int] = None,
    random_state: int = 42
) -> List[CVFold]:
    """
    Create cross-validation splits.
    
    Factory function that creates the appropriate CV strategy.
    
    Args:
        X: Feature DataFrame
        y: Labels
        subject_ids: Subject IDs
        method: 'loso', 'kfold', or 'stratified_group'
        n_folds: Number of folds (only for kfold/stratified_group)
        random_state: Random seed
        
    Returns:
        List of CVFold objects
    """
    if method == 'loso':
        cv = LOSOCrossValidator()
        splits = list(cv.split(X, y, subject_ids))
    
    elif method == 'kfold':
        n = n_folds or 5
        cv = KFoldCrossValidator(n_splits=n, random_state=random_state)
        splits = list(cv.split(X, y))
    
    elif method == 'stratified_group':
        n = n_folds or 5
        cv = StratifiedGroupKFold(n_splits=n, random_state=random_state)
        splits = list(cv.split(X, y, subject_ids))
    
    else:
        raise ValueError(f"Unknown CV method: {method}. Choose from: loso, kfold, stratified_group")
    
    logger.info(f"Created {len(splits)} CV splits using method='{method}'")
    return splits


def get_train_test_data(
    X: pd.DataFrame,
    y: np.ndarray,
    fold: CVFold
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Extract train/test data for a given fold.
    
    Args:
        X: Full feature DataFrame
        y: Full labels array
        fold: CVFold object
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    X_train = X.iloc[fold.train_indices].reset_index(drop=True)
    y_train = y[fold.train_indices]
    X_test = X.iloc[fold.test_indices].reset_index(drop=True)
    y_test = y[fold.test_indices]
    
    return X_train, y_train, X_test, y_test


def summarize_cv_splits(folds: List[CVFold]) -> Dict[str, Any]:
    """
    Generate summary statistics for CV splits.
    
    Args:
        folds: List of CVFold objects
        
    Returns:
        Dictionary with CV summary statistics
    """
    train_sizes = [f.n_train for f in folds]
    test_sizes = [f.n_test for f in folds]
    
    summary = {
        'n_folds': len(folds),
        'total_samples': folds[0].n_train + folds[0].n_test if folds else 0,
        'train_size_mean': np.mean(train_sizes),
        'train_size_std': np.std(train_sizes),
        'test_size_mean': np.mean(test_sizes),
        'test_size_std': np.std(test_sizes),
        'test_subjects': [f.test_subject for f in folds if f.test_subject],
    }
    
    return summary


# =============================================================================
# Module test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Cross-Validation Module")
    print("=" * 60)
    
    # Create synthetic data mimicking BOAS structure
    np.random.seed(42)
    n_subjects = 10
    epochs_per_subject = 100
    n_features = 149
    
    # Generate data
    X_list = []
    y_list = []
    subject_ids = []
    
    for subj in range(1, n_subjects + 1):
        X_list.append(np.random.randn(epochs_per_subject, n_features))
        y_list.append(np.random.randint(0, 5, epochs_per_subject))
        subject_ids.extend([str(subj)] * epochs_per_subject)
    
    X = pd.DataFrame(
        np.vstack(X_list),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = np.concatenate(y_list)
    subject_ids = np.array(subject_ids)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Subjects: {np.unique(subject_ids)}")
    
    # Test LOSO
    print("\n" + "-" * 40)
    print("Testing LOSO Cross-Validation")
    loso = LOSOCrossValidator(verbose=True)
    loso_folds = list(loso.split(X, y, subject_ids))
    print(f"  Number of folds: {len(loso_folds)}")
    print(f"  First fold: test_subject={loso_folds[0].test_subject}, "
          f"train={loso_folds[0].n_train}, test={loso_folds[0].n_test}")
    
    # Test K-Fold
    print("\n" + "-" * 40)
    print("Testing K-Fold Cross-Validation")
    kfold_folds = create_cv_splits(X, y, subject_ids, method='kfold', n_folds=5)
    print(f"  Number of folds: {len(kfold_folds)}")
    
    # Test Stratified Group K-Fold
    print("\n" + "-" * 40)
    print("Testing Stratified Group K-Fold")
    sgk_folds = create_cv_splits(X, y, subject_ids, method='stratified_group', n_folds=5)
    print(f"  Number of folds: {len(sgk_folds)}")
    
    # Test data extraction
    print("\n" + "-" * 40)
    print("Testing data extraction")
    X_train, y_train, X_test, y_test = get_train_test_data(X, y, loso_folds[0])
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Summary
    print("\n" + "-" * 40)
    summary = summarize_cv_splits(loso_folds)
    print(f"LOSO Summary:")
    print(f"  Folds: {summary['n_folds']}")
    print(f"  Train size: {summary['train_size_mean']:.0f} ± {summary['train_size_std']:.0f}")
    print(f"  Test size: {summary['test_size_mean']:.0f} ± {summary['test_size_std']:.0f}")
    
    print("\n" + "=" * 60)
    print("Cross-Validation Module: ✓ All tests passed")
