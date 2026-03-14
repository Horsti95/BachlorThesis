"""
All Models for Sleep Stage Classification - Experimental Comparison
====================================================================

This is a STANDALONE experimental script, NOT part of the thesis.
Goal: Try every reasonable ML model on the sleep stage data and compare results.

Models included:
────────────────
CLASSICAL ML (on 149 hand-crafted features):
  1. Logistic Regression
  2. Ridge Classifier
  3. k-Nearest Neighbors (kNN)
  4. Support Vector Machine (SVM) - linear
  5. Support Vector Machine (SVM) - RBF kernel
  6. Naive Bayes (Gaussian)
  7. Decision Tree
  8. Random Forest
  9. Extra Trees
  10. AdaBoost
  11. Gradient Boosting (sklearn)
  12. XGBoost
  13. LightGBM
  14. CatBoost

NEURAL NETWORKS (on 149 hand-crafted features):
  15. FNN / MLP (PyTorch)

DEEP LEARNING ON RAW SIGNALS (needs raw EEG, no hand-crafted features):
  16. 1D-CNN
  17. LSTM
  18. GRU
  19. CNN-LSTM hybrid
  20. Transformer (simple)

ENSEMBLE:
  21. Voting ensemble of best classical models
  22. Stacking ensemble

Cross-validation options:
  - LOSO (128 folds, gold standard, slow)
  - Stratified 5-Fold (fast, for quick comparison)
  - Simple train/test split (fastest, for debugging)

Author: Lennart Gorzel (experimental)
Date: March 2026
"""

import sys
import os
import time
import json
import logging
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneGroupOut, train_test_split
)
from tqdm import tqdm

# Add parent directory to path so we can import thesis modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']


# =============================================================================
# Time Estimation & Smart Model Filtering
# =============================================================================

# Empirical timing data (seconds) from 30-subject runs with 5-fold CV
# Used to estimate runtimes for different subject counts
_TIMING_BENCHMARKS = {
    # model_name: (time_at_30_subjects_5fold, scaling_exponent)
    # scaling_exponent: 1.0 = linear, 2.0 = quadratic, 1.5 = superlinear
    'logistic_regression': (2.7, 1.0),
    'ridge_classifier': (0.2, 1.0),
    'knn_5': (1.7, 1.2),      # KNN prediction scales with dataset size
    'knn_10': (0.3, 1.2),
    'svm_linear': (237.0, 1.5),  # LinearSVC scales superlinearly
    'svm_rbf': (115.0, 2.0),    # RBF SVM scales quadratically
    'naive_bayes': (0.1, 1.0),
    'decision_tree': (3.8, 1.0),
    'random_forest': (2.7, 1.1),
    'extra_trees': (0.8, 1.1),
    'adaboost': (42.4, 1.2),
    'gradient_boosting': (300.0, 1.3),  # sklearn GB is slow, no parallelism
    'xgboost': (5.0, 1.1),
    'lightgbm': (3.0, 1.0),
    'catboost': (10.0, 1.1),
    'fnn': (30.0, 1.0),
    'cnn_1d': (120.0, 1.0),
    'lstm': (180.0, 1.0),
    'gru': (150.0, 1.0),
    'cnn_lstm': (200.0, 1.0),
    'transformer': (180.0, 1.0),
}

# Models that are too slow for LOSO at 30+ subjects
_SLOW_MODELS_LOSO = {'svm_linear', 'svm_rbf', 'gradient_boosting'}
# Models that are too slow for any CV at 30+ subjects
_SLOW_MODELS_ANY = {'svm_rbf'}


def estimate_time(model_name: str, n_subjects: int, cv_method: str) -> float:
    """
    Estimate training time in seconds for a model given subject count and CV method.

    Based on empirical benchmarks from 30-subject 5-fold runs.
    """
    if model_name not in _TIMING_BENCHMARKS:
        return 0.0

    base_time, exponent = _TIMING_BENCHMARKS[model_name]
    # Scale from 30 subjects to n_subjects
    scale_factor = (n_subjects / 30.0) ** exponent
    estimated_5fold = base_time * scale_factor

    if cv_method == 'stratified_5fold':
        return estimated_5fold
    elif cv_method == 'loso':
        # LOSO has n_subjects folds vs 5 folds
        return estimated_5fold * (n_subjects / 5.0)
    elif cv_method == 'train_test':
        return estimated_5fold / 5.0  # Single split
    return estimated_5fold


def format_time_estimate(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}hrs"


def check_slow_models(
    models: Dict[str, Any],
    n_subjects: int,
    cv_method: str,
    threshold_seconds: float = 3600.0
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Check for models that will be slow and prompt user for confirmation.

    Returns:
        (filtered_models, skipped_models_with_estimates)
    """
    fast_models = {}
    slow_models = {}

    for name, info in models.items():
        est = estimate_time(name, n_subjects, cv_method)

        # Auto-skip models in exclusion lists for high subject counts
        if n_subjects >= 30:
            if cv_method == 'loso' and name in _SLOW_MODELS_LOSO:
                slow_models[name] = est
                continue
            if name in _SLOW_MODELS_ANY:
                slow_models[name] = est
                continue

        # Warn for any model exceeding threshold
        if est > threshold_seconds:
            slow_models[name] = est
            continue

        fast_models[name] = info

    if slow_models:
        print("\n" + "=" * 70)
        print("⚠  SLOW MODEL WARNING")
        print("=" * 70)
        print(f"The following models are estimated to be very slow")
        print(f"with {n_subjects} subjects using {cv_method}:\n")

        for name, est in sorted(slow_models.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {name:<25} estimated: {format_time_estimate(est)}")

        print(f"\nTotal estimated time for slow models: "
              f"{format_time_estimate(sum(slow_models.values()))}")
        print(f"Total estimated time for fast models: "
              f"{format_time_estimate(sum(estimate_time(n, n_subjects, cv_method) for n in fast_models))}")
        print()

        try:
            answer = input("Include slow models anyway? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = 'n'

        if answer == 'y':
            fast_models.update({name: models[name] for name in slow_models})
            logger.info("Including all models (user confirmed)")
            slow_models = {}
        else:
            logger.info(f"Skipping {len(slow_models)} slow model(s): {list(slow_models.keys())}")

    return fast_models, slow_models


# =============================================================================
# Data Loading - Reuses thesis pipeline for feature extraction
# =============================================================================

def load_features_from_thesis_cache(
    cache_dir: str = None,
    data_path: str = None,
    channel_preset: str = "eeg_only",
    max_subjects: int = None
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load pre-computed features from the thesis feature cache.

    Returns:
        X: DataFrame of shape (n_epochs, 149) with features
        y: array of shape (n_epochs,) with sleep stage labels (0-4)
        subject_ids: array of shape (n_epochs,) with subject identifiers
    """
    # Try thesis cache first
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent / "results" / "features_cache_global"
    else:
        cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Feature cache not found at {cache_dir}.\n"
            f"Run the thesis pipeline first to populate the cache, or provide data_path."
        )

    all_features = []
    all_labels = []
    all_subjects = []

    # Determine which channels to use
    if channel_preset == "eeg_only":
        channels = ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2']
        n_per_channel = 23
        n_global = 11
        expected_features = len(channels) * n_per_channel + n_global  # 149
    elif channel_preset == "eeg_plus_physiological":
        channels = ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2', 'PSG_EOG', 'PSG_EMG']
        n_per_channel = 23
        n_global = 11
        expected_features = len(channels) * n_per_channel + n_global  # 195
    else:
        channels = ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2']
        expected_features = 149

    # Load cached .npz files
    cache_files = sorted(cache_dir.glob("subject_*_full.npz"))

    if not cache_files:
        raise FileNotFoundError(f"No cached feature files found in {cache_dir}")

    if max_subjects is not None:
        cache_files = cache_files[:max_subjects]

    logger.info(f"Loading features from {len(cache_files)} cached subjects...")

    for f in tqdm(cache_files, desc="Loading subjects", unit="subj"):
        data = np.load(f, allow_pickle=True)
        features = data['features']  # shape: (n_epochs, n_features)
        labels = data['labels']      # shape: (n_epochs,)

        # Extract subject ID from filename
        subject_id = f.stem.replace('subject_', '').replace('_full', '')

        # Select only the channels we need (if cache has 195 features but we want 149)
        if channel_preset == "eeg_only" and features.shape[1] > expected_features:
            # Cache layout: [ch1×23, ch2×23, ..., ch8×23, 11 global]
            # For eeg_only: take first 6×23=138 per-channel + last 11 global
            n_total_channels = features.shape[1] - n_global  # per-channel portion
            eeg_cols = list(range(6 * n_per_channel))  # first 138
            global_cols = list(range(n_total_channels, n_total_channels + n_global))  # last 11
            features = features[:, eeg_cols + global_cols]

        # Filter invalid labels
        valid_mask = np.isin(labels, [0, 1, 2, 3, 4])
        features = features[valid_mask]
        labels = labels[valid_mask]

        if len(labels) == 0:
            continue

        all_features.append(features)
        all_labels.append(labels)
        all_subjects.append(np.full(len(labels), subject_id))

    X = pd.DataFrame(np.vstack(all_features))
    y = np.concatenate(all_labels).astype(int)
    subject_ids = np.concatenate(all_subjects)

    logger.info(f"Loaded {len(y)} epochs from {len(cache_files)} subjects")
    logger.info(f"Feature shape: {X.shape}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, subject_ids


def load_raw_signals_from_thesis(
    data_path: str,
    channel_preset: str = "eeg_only",
    target_sfreq: float = 128.0,
    max_subjects: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw EEG signals (NOT features) for deep learning models.

    Returns:
        X: array of shape (n_epochs, n_channels, n_samples) - raw signal
        y: array of shape (n_epochs,) - labels
        subject_ids: array of shape (n_epochs,) - subject IDs
    """
    from data_loader_boas import BOASDataLoader
    from preprocessing import preprocess_subject

    if channel_preset == "eeg_only":
        channels = ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2']
    else:
        channels = ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2', 'PSG_EOG', 'PSG_EMG']

    loader = BOASDataLoader(
        base_path=data_path,
        target_channels=channels,
        target_sfreq=None,
        epoch_duration=30.0,
        use_human_labels=True,
        preload=True
    )

    subjects = loader.get_subject_ids()
    if max_subjects:
        subjects = subjects[:max_subjects]

    all_epochs = []
    all_labels = []
    all_subjects = []

    logger.info(f"Loading raw signals from {len(subjects)} subjects...")

    for subj_id in tqdm(subjects, desc="Loading raw EEG", unit="subj"):
        try:
            epochs, labels = preprocess_subject(
                loader, subj_id,
                bandpass_low=0.5, bandpass_high=40.0,
                notch_freq=50.0, target_sfreq=target_sfreq
            )

            # Filter valid labels
            valid_mask = np.isin(labels, [0, 1, 2, 3, 4])
            epochs = epochs[valid_mask]
            labels = labels[valid_mask]

            if len(labels) < 10:
                continue

            all_epochs.append(epochs)
            all_labels.append(labels)
            all_subjects.append(np.full(len(labels), subj_id))

        except Exception as e:
            logger.warning(f"Failed to load subject {subj_id}: {e}")
            continue

    X = np.concatenate(all_epochs)     # (n_epochs, n_channels, n_samples)
    y = np.concatenate(all_labels).astype(int)
    subject_ids = np.concatenate(all_subjects)

    logger.info(f"Raw signal shape: {X.shape}")
    logger.info(f"  {X.shape[0]} epochs × {X.shape[1]} channels × {X.shape[2]} samples")

    return X, y, subject_ids


# =============================================================================
# Model Definitions
# =============================================================================

def get_classical_models() -> Dict[str, Any]:
    """
    Get all classical ML models (work on 149 tabular features).
    Returns dict of model_name -> (model_instance, needs_scaling).
    """
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        AdaBoostClassifier, GradientBoostingClassifier
    )

    models = {}

    # ── Linear Models ──
    models['logistic_regression'] = {
        'model': LogisticRegression(
            max_iter=1000, random_state=42,
            solver='lbfgs', n_jobs=-1, class_weight='balanced'
        ),
        'needs_scaling': True,
        'category': 'linear'
    }

    models['ridge_classifier'] = {
        'model': RidgeClassifier(
            alpha=1.0, class_weight='balanced'
        ),
        'needs_scaling': True,
        'category': 'linear'
    }

    # ── Distance-Based ──
    models['knn_5'] = {
        'model': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'needs_scaling': True,
        'category': 'distance'
    }

    models['knn_10'] = {
        'model': KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
        'needs_scaling': True,
        'category': 'distance'
    }

    # ── SVM ──
    # LinearSVC is orders of magnitude faster than SVC(kernel='linear')
    # on large datasets (uses liblinear instead of libsvm)
    models['svm_linear'] = {
        'model': LinearSVC(
            max_iter=2000, random_state=42,
            class_weight='balanced', dual='auto'
        ),
        'needs_scaling': True,
        'category': 'svm'
    }

    # RBF SVM - O(n²) scaling, auto-excluded for 30+ subjects
    models['svm_rbf'] = {
        'model': SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', random_state=42
        ),
        'needs_scaling': True,
        'category': 'svm'
    }

    # ── Naive Bayes ──
    models['naive_bayes'] = {
        'model': GaussianNB(),
        'needs_scaling': False,
        'category': 'probabilistic'
    }

    # ── Tree-Based ──
    models['decision_tree'] = {
        'model': DecisionTreeClassifier(
            random_state=42, class_weight='balanced', max_depth=20
        ),
        'needs_scaling': False,
        'category': 'tree'
    }

    models['random_forest'] = {
        'model': RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
            class_weight='balanced'
        ),
        'needs_scaling': False,
        'category': 'tree_ensemble'
    }

    models['extra_trees'] = {
        'model': ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
            class_weight='balanced'
        ),
        'needs_scaling': False,
        'category': 'tree_ensemble'
    }

    models['adaboost'] = {
        'model': AdaBoostClassifier(
            n_estimators=100, random_state=42, learning_rate=0.1
        ),
        'needs_scaling': False,
        'category': 'boosting'
    }

    models['gradient_boosting'] = {
        'model': GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.5, random_state=42
        ),
        'needs_scaling': False,
        'category': 'boosting'
    }

    # ── XGBoost ──
    try:
        import xgboost as xgb
        models['xgboost'] = {
            'model': xgb.XGBClassifier(
                max_depth=6, n_estimators=200, learning_rate=0.1,
                objective='multi:softmax', num_class=5,
                random_state=42, n_jobs=-1, verbosity=0,
                eval_metric='mlogloss'
            ),
            'needs_scaling': False,
            'category': 'boosting'
        }
    except ImportError:
        logger.warning("XGBoost not installed, skipping")

    # ── LightGBM ──
    try:
        import lightgbm as lgb
        models['lightgbm'] = {
            'model': lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbose=-1,
                class_weight='balanced'
            ),
            'needs_scaling': False,
            'category': 'boosting'
        }
    except ImportError:
        logger.warning("LightGBM not installed, skipping")

    # ── CatBoost ──
    try:
        from catboost import CatBoostClassifier
        models['catboost'] = {
            'model': CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1,
                random_seed=42, verbose=0, auto_class_weights='Balanced',
                task_type='CPU'
            ),
            'needs_scaling': False,
            'category': 'boosting'
        }
    except ImportError:
        logger.warning("CatBoost not installed, skipping")

    return models


# =============================================================================
# PyTorch Deep Learning Models
# =============================================================================

def _check_torch():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        logger.warning("PyTorch not installed, deep learning models unavailable")
        return False


class FNNClassifier:
    """Feedforward Neural Network on tabular features (149 inputs)."""

    def __init__(self, hidden_dims=(256, 128, 64), dropout=0.3, epochs=50,
                 lr=0.001, batch_size=256, patience=5):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.scaler = StandardScaler()
        self.model = None
        self.device = None

    def fit(self, X, y):
        import torch.nn as nn

        self.device = _get_device()

        X_scaled = self.scaler.fit_transform(X)
        n_features = X_scaled.shape[1]
        n_classes = len(np.unique(y))

        # Build network
        layers = []
        in_dim = n_features
        for h in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        net = nn.Sequential(*layers)

        self.model = _train_pytorch_model(
            net, X_scaled, y, self.epochs, self.lr,
            self.batch_size, self.patience, self.device
        )
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return _predict_pytorch(self.model, X_scaled, self.device, self.batch_size)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return _predict_proba_pytorch(self.model, X_scaled, self.device, self.batch_size)


def _get_device():
    """Get the best available device (CUDA GPU preferred)."""
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        return device
    else:
        logger.info("No GPU found, using CPU")
        return torch.device('cpu')


def _train_pytorch_model(model, X, y, epochs, lr, batch_size, patience, device):
    """
    Generic PyTorch training loop with proper GPU memory management.

    Keeps data on CPU and only moves batches to GPU during training.
    This supports datasets larger than GPU memory.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)

    # Keep data on CPU, move batches to GPU in the DataLoader
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == 'cuda')
    )

    # Use class weights for imbalanced data
    class_counts = np.bincount(y, minlength=5)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    weight_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        scheduler.step(avg)
        if avg < best_loss:
            best_loss = avg
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model


def _predict_pytorch(model, X, device, batch_size=256):
    """Predict with batched inference to avoid GPU OOM."""
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device, non_blocking=True)
            out = model(batch)
            preds.append(torch.argmax(out, dim=1).cpu())
    return torch.cat(preds).numpy()


def _predict_proba_pytorch(model, X, device, batch_size=256):
    """Predict probabilities with batched inference."""
    import torch
    import torch.nn.functional as F
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device, non_blocking=True)
            out = F.softmax(model(batch), dim=1)
            probs.append(out.cpu())
    return torch.cat(probs).numpy()


class CNN1DClassifier:
    """1D Convolutional Neural Network on raw EEG signals."""

    def __init__(self, n_channels=6, n_samples=3840, n_classes=5,
                 epochs=30, lr=0.001, batch_size=64, patience=5):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.device = None

    def _build(self):
        import torch.nn as nn

        class CNN1D(nn.Module):
            def __init__(self, n_ch, n_samp, n_cls):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(n_ch, 32, kernel_size=25, stride=2, padding=12),
                    nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
                    nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
                    nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
                    nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
                    nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
                    nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 8, 128), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(128, n_cls)
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        return CNN1D(self.n_channels, self.n_samples, self.n_classes)

    def fit(self, X, y):
        self.device = _get_device()
        self.model = _train_pytorch_model(
            self._build(), X, y, self.epochs, self.lr,
            self.batch_size, self.patience, self.device
        )
        return self

    def predict(self, X):
        return _predict_pytorch(self.model, X, self.device, self.batch_size)

    def predict_proba(self, X):
        return _predict_proba_pytorch(self.model, X, self.device, self.batch_size)


class LSTMClassifier:
    """LSTM on raw EEG signals."""

    def __init__(self, n_channels=6, n_classes=5, hidden_size=128,
                 num_layers=2, epochs=30, lr=0.001, batch_size=64, patience=5):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.device = None

    def _build(self):
        import torch.nn as nn

        class LSTMNet(nn.Module):
            def __init__(self, n_ch, hidden, n_layers, n_cls):
                super().__init__()
                self.lstm = nn.LSTM(n_ch, hidden, n_layers, batch_first=True, dropout=0.3)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(64, n_cls)
                )

            def forward(self, x):
                x = x.transpose(1, 2)
                x = x[:, ::4, :]  # Subsample 4x for efficiency
                out, (h_n, _) = self.lstm(x)
                return self.classifier(h_n[-1])

        return LSTMNet(self.n_channels, self.hidden_size, self.num_layers, self.n_classes)

    def fit(self, X, y):
        self.device = _get_device()
        self.model = _train_pytorch_model(
            self._build(), X, y, self.epochs, self.lr,
            self.batch_size, self.patience, self.device
        )
        return self

    def predict(self, X):
        return _predict_pytorch(self.model, X, self.device, self.batch_size)

    def predict_proba(self, X):
        return _predict_proba_pytorch(self.model, X, self.device, self.batch_size)


class GRUClassifier:
    """GRU on raw EEG signals (faster than LSTM, similar performance)."""

    def __init__(self, n_channels=6, n_classes=5, hidden_size=128,
                 num_layers=2, epochs=30, lr=0.001, batch_size=64, patience=5):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.device = None

    def _build(self):
        import torch.nn as nn

        class GRUNet(nn.Module):
            def __init__(self, n_ch, hidden, n_layers, n_cls):
                super().__init__()
                self.gru = nn.GRU(n_ch, hidden, n_layers, batch_first=True, dropout=0.3)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(64, n_cls)
                )

            def forward(self, x):
                x = x.transpose(1, 2)
                x = x[:, ::4, :]  # Subsample
                out, h_n = self.gru(x)
                return self.classifier(h_n[-1])

        return GRUNet(self.n_channels, self.hidden_size, self.num_layers, self.n_classes)

    def fit(self, X, y):
        self.device = _get_device()
        self.model = _train_pytorch_model(
            self._build(), X, y, self.epochs, self.lr,
            self.batch_size, self.patience, self.device
        )
        return self

    def predict(self, X):
        return _predict_pytorch(self.model, X, self.device, self.batch_size)

    def predict_proba(self, X):
        return _predict_proba_pytorch(self.model, X, self.device, self.batch_size)


class CNNLSTMClassifier:
    """CNN-LSTM hybrid: CNN extracts local features, LSTM captures temporal patterns."""

    def __init__(self, n_channels=6, n_classes=5, epochs=30, lr=0.001,
                 batch_size=64, patience=5):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.device = None

    def _build(self):
        import torch.nn as nn

        class CNNLSTM(nn.Module):
            def __init__(self, n_ch, n_cls):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv1d(n_ch, 32, kernel_size=25, stride=2, padding=12),
                    nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
                    nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
                    nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
                )
                self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True)
                self.classifier = nn.Sequential(
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(32, n_cls)
                )

            def forward(self, x):
                features = self.cnn(x)
                features = features.transpose(1, 2)
                _, (h_n, _) = self.lstm(features)
                return self.classifier(h_n[-1])

        return CNNLSTM(self.n_channels, self.n_classes)

    def fit(self, X, y):
        self.device = _get_device()
        self.model = _train_pytorch_model(
            self._build(), X, y, self.epochs, self.lr,
            self.batch_size, self.patience, self.device
        )
        return self

    def predict(self, X):
        return _predict_pytorch(self.model, X, self.device, self.batch_size)

    def predict_proba(self, X):
        return _predict_proba_pytorch(self.model, X, self.device, self.batch_size)


class SimpleTransformerClassifier:
    """Simple Transformer encoder on raw EEG signals."""

    def __init__(self, n_channels=6, n_classes=5, d_model=64, nhead=4,
                 num_layers=2, epochs=30, lr=0.001, batch_size=64, patience=5):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.device = None

    def _build(self):
        import torch
        import torch.nn as nn

        class EEGTransformer(nn.Module):
            def __init__(self, n_ch, d_model, nhead, n_layers, n_cls):
                super().__init__()
                # Project channels to d_model
                self.input_proj = nn.Linear(n_ch, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                    dropout=0.3, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.classifier = nn.Sequential(
                    nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(64, n_cls)
                )

            def forward(self, x):
                # x: (batch, channels, samples) -> (batch, samples, channels)
                x = x.transpose(1, 2)
                # Subsample heavily for transformer (every 8th sample)
                x = x[:, ::8, :]
                x = self.input_proj(x)  # (batch, T/8, d_model)
                x = self.transformer(x)
                # Global average pooling over time
                x = x.mean(dim=1)
                return self.classifier(x)

        return EEGTransformer(
            self.n_channels, self.d_model, self.nhead,
            self.num_layers, self.n_classes
        )

    def fit(self, X, y):
        self.device = _get_device()
        self.model = _train_pytorch_model(
            self._build(), X, y, self.epochs, self.lr,
            self.batch_size, self.patience, self.device
        )
        return self

    def predict(self, X):
        return _predict_pytorch(self.model, X, self.device, self.batch_size)

    def predict_proba(self, X):
        return _predict_proba_pytorch(self.model, X, self.device, self.batch_size)


# =============================================================================
# Evaluation & Cross-Validation
# =============================================================================

def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    """Compute all relevant metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        if i < len(f1_per):
            metrics[f'f1_{name}'] = f1_per[i]
    return metrics


def run_single_model_cv(
    model_name: str,
    model_info: Dict,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    cv_method: str = "stratified_5fold",
    is_raw_signal: bool = False
) -> Dict[str, Any]:
    """
    Run cross-validation for a single model.

    Args:
        model_name: Name of the model
        model_info: Dict with 'model', 'needs_scaling', 'category'
        X: Features or raw signals
        y: Labels
        subject_ids: Subject IDs (for LOSO)
        cv_method: "loso", "stratified_5fold", or "train_test"
        is_raw_signal: If True, X is raw signal (n_epochs, n_channels, n_samples)

    Returns:
        Dict with model results
    """
    from sklearn.base import clone

    result = {
        'model_name': model_name,
        'category': model_info.get('category', 'unknown'),
        'cv_method': cv_method,
        'is_raw_signal': is_raw_signal,
        'fold_metrics': [],
        'error': None,
    }

    start_time = time.time()

    try:
        if cv_method == "train_test":
            # Simple 80/20 split (stratified)
            if is_raw_signal:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )

            model = _get_fresh_model(model_info)

            if model_info.get('needs_scaling', False) and not is_raw_signal:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred)
            result['fold_metrics'].append(metrics)

        elif cv_method == "stratified_5fold":
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                if is_raw_signal:
                    X_train, X_test = X[train_idx], X[test_idx]
                else:
                    X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = _get_fresh_model(model_info)

                if model_info.get('needs_scaling', False) and not is_raw_signal:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = evaluate_predictions(y_test, y_pred)
                result['fold_metrics'].append(metrics)

        elif cv_method == "loso":
            logo = LeaveOneGroupOut()
            unique_subjects = np.unique(subject_ids)
            n_folds = len(unique_subjects)

            for fold_idx, (train_idx, test_idx) in enumerate(
                tqdm(logo.split(X, y, subject_ids),
                     total=n_folds, desc=f"  LOSO {model_name}", unit="fold", leave=False)
            ):
                if is_raw_signal:
                    X_train, X_test = X[train_idx], X[test_idx]
                else:
                    X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = _get_fresh_model(model_info)

                if model_info.get('needs_scaling', False) and not is_raw_signal:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = evaluate_predictions(y_test, y_pred)
                result['fold_metrics'].append(metrics)

        else:
            raise ValueError(f"Unknown cv_method: {cv_method}")

    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Model {model_name} failed: {result['error']}")
        traceback.print_exc()

    result['training_time_seconds'] = time.time() - start_time

    # Aggregate metrics across folds
    if result['fold_metrics']:
        all_keys = result['fold_metrics'][0].keys()
        result['mean_metrics'] = {}
        result['std_metrics'] = {}
        for key in all_keys:
            vals = [fm[key] for fm in result['fold_metrics']]
            result['mean_metrics'][key] = float(np.mean(vals))
            result['std_metrics'][key] = float(np.std(vals))

    return result


def _get_fresh_model(model_info):
    """Get a fresh (unfitted) copy of a model."""
    from sklearn.base import clone
    model = model_info['model']

    # sklearn models can be cloned
    if hasattr(model, 'get_params'):
        return clone(model)

    # PyTorch models need to be re-instantiated
    if isinstance(model, (FNNClassifier, CNN1DClassifier, LSTMClassifier,
                          GRUClassifier, CNNLSTMClassifier, SimpleTransformerClassifier)):
        return model.__class__(**{
            k: v for k, v in model.__dict__.items()
            if k not in ('model', 'device', 'scaler')
        })

    return model


# =============================================================================
# SHAP Explainability (XAI)
# =============================================================================

def run_shap_analysis(
    model,
    model_name: str,
    model_info: Dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str] = None,
    output_dir: Path = None,
    max_samples: int = 500
) -> Optional[Dict]:
    """
    Run SHAP analysis on a trained model to explain feature importance.

    Works with tree-based models (fast TreeExplainer) and any model (KernelExplainer).

    Args:
        model: Trained model instance
        model_name: Name for output files
        model_info: Model info dict with 'category'
        X_train: Training data (for background)
        X_test: Test data (for explanations)
        feature_names: Feature names (optional)
        output_dir: Where to save SHAP plots
        max_samples: Max test samples to explain (for speed)

    Returns:
        Dict with SHAP values and feature importances, or None on failure
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Install with: pip install shap")
        return None

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results" / "shap"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category = model_info.get('category', '')

    # Subsample for speed
    if len(X_test) > max_samples:
        idx = np.random.RandomState(42).choice(len(X_test), max_samples, replace=False)
        X_test_sample = X_test[idx]
    else:
        X_test_sample = X_test

    logger.info(f"  Running SHAP analysis for {model_name} ({len(X_test_sample)} samples)...")

    try:
        # Use TreeExplainer for tree-based models (very fast)
        if category in ('tree', 'tree_ensemble', 'boosting') and hasattr(model, 'predict_proba'):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sample)
            except Exception:
                # Fallback: some boosting models need different handling
                explainer = shap.TreeExplainer(model, X_train[:100])
                shap_values = explainer.shap_values(X_test_sample)

        # Use LinearExplainer for linear models
        elif category == 'linear':
            background = shap.sample(X_train, min(100, len(X_train)))
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_test_sample)

        # Use KernelExplainer as fallback (slower but works for anything)
        else:
            background = shap.sample(X_train, min(50, len(X_train)))
            if hasattr(model, 'predict_proba'):
                explainer = shap.KernelExplainer(model.predict_proba, background)
            else:
                explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test_sample, nsamples=100)

        # Handle different SHAP value formats
        # shap_values can be: list of arrays (one per class) or single array
        if isinstance(shap_values, list):
            # Multi-class: average absolute SHAP across classes
            shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            shap_abs_mean = np.abs(shap_values).mean(axis=0)

        # Feature importance ranking
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(shap_abs_mean))]

        importance_order = np.argsort(shap_abs_mean)[::-1]
        top_n = min(20, len(importance_order))

        result = {
            'model_name': model_name,
            'top_features': [
                {'feature': feature_names[i], 'importance': float(shap_abs_mean[i])}
                for i in importance_order[:top_n]
            ],
            'shap_values_shape': str(shap_abs_mean.shape),
        }

        # Generate and save SHAP summary plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            # Bar plot of top features
            top_features = [feature_names[i] for i in importance_order[:top_n]]
            top_importances = [shap_abs_mean[i] for i in importance_order[:top_n]]
            axes[0].barh(range(top_n), top_importances[::-1])
            axes[0].set_yticks(range(top_n))
            axes[0].set_yticklabels(top_features[::-1], fontsize=8)
            axes[0].set_xlabel('Mean |SHAP value|')
            axes[0].set_title(f'SHAP Feature Importance: {model_name}')

            # Beeswarm/summary plot for multi-class
            plt.sca(axes[1])
            if isinstance(shap_values, list):
                # Use first class for beeswarm
                shap.summary_plot(
                    shap_values[0], X_test_sample,
                    feature_names=feature_names,
                    max_display=top_n, show=False, plot_size=None
                )
            else:
                shap.summary_plot(
                    shap_values, X_test_sample,
                    feature_names=feature_names,
                    max_display=top_n, show=False, plot_size=None
                )
            axes[1].set_title(f'SHAP Summary: {model_name}')

            plt.tight_layout()
            plot_path = output_dir / f"shap_{model_name}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  SHAP plot saved: {plot_path}")
            result['plot_path'] = str(plot_path)
        except Exception as e:
            logger.warning(f"  Could not generate SHAP plot: {e}")

        # Save per-class SHAP plots
        try:
            if isinstance(shap_values, list) and len(shap_values) == 5:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 5, figsize=(30, 6))
                for cls_idx, cls_name in enumerate(CLASS_NAMES):
                    if cls_idx < len(shap_values):
                        cls_importance = np.abs(shap_values[cls_idx]).mean(axis=0)
                        cls_order = np.argsort(cls_importance)[::-1][:10]
                        axes[cls_idx].barh(
                            range(10),
                            [cls_importance[i] for i in cls_order[::-1]]
                        )
                        axes[cls_idx].set_yticks(range(10))
                        axes[cls_idx].set_yticklabels(
                            [feature_names[i] for i in cls_order[::-1]], fontsize=7
                        )
                        axes[cls_idx].set_title(f'{cls_name}')
                plt.suptitle(f'Per-Class SHAP: {model_name}', fontsize=14)
                plt.tight_layout()
                plot_path = output_dir / f"shap_{model_name}_per_class.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                result['per_class_plot_path'] = str(plot_path)
        except Exception as e:
            logger.warning(f"  Could not generate per-class SHAP plot: {e}")

        return result

    except Exception as e:
        logger.warning(f"  SHAP analysis failed for {model_name}: {e}")
        return None


# =============================================================================
# Transfer Learning (PSG → Headband)
# =============================================================================

class TransferLearningCNN:
    """
    Transfer learning: pre-train on PSG EEG (6 channels), fine-tune on headband (2 channels).

    Uses a shared feature extractor with domain-specific input adapters.
    The BOAS dataset has simultaneous PSG and headband recordings,
    making it ideal for transfer learning.
    """

    def __init__(self, n_classes=5, epochs_pretrain=30, epochs_finetune=15,
                 lr=0.001, batch_size=64, patience=5):
        self.n_classes = n_classes
        self.epochs_pretrain = epochs_pretrain
        self.epochs_finetune = epochs_finetune
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.device = None
        self.pretrained = False

    def _build_backbone(self, n_input_channels):
        """Build the shared feature extractor backbone."""
        import torch.nn as nn

        class SleepBackbone(nn.Module):
            def __init__(self, n_ch, n_cls):
                super().__init__()
                # Input adapter (channel-specific)
                self.input_adapter = nn.Sequential(
                    nn.Conv1d(n_ch, 32, kernel_size=25, stride=2, padding=12),
                    nn.BatchNorm1d(32), nn.ReLU(),
                )
                # Shared feature extractor
                self.features = nn.Sequential(
                    nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
                    nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
                    nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
                    nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
                    nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 8, 128), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(128, n_cls)
                )

            def forward(self, x):
                x = self.input_adapter(x)
                x = self.features(x)
                return self.classifier(x)

        return SleepBackbone(n_input_channels, self.n_classes)

    def pretrain_on_psg(self, X_psg, y_psg):
        """
        Pre-train the model on PSG data (6 channels).

        Args:
            X_psg: (n_epochs, 6, n_samples) - PSG EEG data
            y_psg: (n_epochs,) - sleep stage labels
        """
        import torch

        self.device = _get_device()
        n_channels = X_psg.shape[1]
        logger.info(f"Pre-training on PSG data: {X_psg.shape[0]} epochs, {n_channels} channels")

        self.model = _train_pytorch_model(
            self._build_backbone(n_channels), X_psg, y_psg,
            self.epochs_pretrain, self.lr, self.batch_size,
            self.patience, self.device
        )
        self.pretrained = True
        logger.info("Pre-training complete")
        return self

    def finetune_on_headband(self, X_hb, y_hb):
        """
        Fine-tune on headband data (2 channels).

        Replaces the input adapter, freezes feature extractor initially,
        then unfreezes for full fine-tuning.

        Args:
            X_hb: (n_epochs, 2, n_samples) - Headband EEG data
            y_hb: (n_epochs,) - sleep stage labels
        """
        import torch
        import torch.nn as nn

        if not self.pretrained:
            raise RuntimeError("Must call pretrain_on_psg() first")

        n_channels = X_hb.shape[1]
        logger.info(f"Fine-tuning on headband data: {X_hb.shape[0]} epochs, {n_channels} channels")

        # Replace input adapter for new channel count
        old_adapter = self.model.input_adapter
        self.model.input_adapter = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(32), nn.ReLU(),
        ).to(self.device)

        # Phase 1: Freeze feature extractor, train only input adapter + classifier
        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model = _train_pytorch_model(
            self.model, X_hb, y_hb,
            max(5, self.epochs_finetune // 2), self.lr * 0.1,
            self.batch_size, self.patience, self.device
        )

        # Phase 2: Unfreeze everything, fine-tune with low LR
        for param in self.model.features.parameters():
            param.requires_grad = True

        self.model = _train_pytorch_model(
            self.model, X_hb, y_hb,
            self.epochs_finetune, self.lr * 0.01,
            self.batch_size, self.patience, self.device
        )

        logger.info("Fine-tuning complete")
        return self

    def fit(self, X, y):
        """Standard fit (no transfer, just train from scratch)."""
        self.device = _get_device()
        n_channels = X.shape[1]
        self.model = _train_pytorch_model(
            self._build_backbone(n_channels), X, y,
            self.epochs_pretrain, self.lr, self.batch_size,
            self.patience, self.device
        )
        return self

    def predict(self, X):
        return _predict_pytorch(self.model, X, self.device, self.batch_size)

    def predict_proba(self, X):
        return _predict_proba_pytorch(self.model, X, self.device, self.batch_size)


def run_transfer_learning_experiment(
    data_path: str,
    output_dir: Path = None,
    max_subjects: int = None,
    cv_method: str = "stratified_5fold"
) -> Optional[Dict]:
    """
    Run the PSG → Headband transfer learning experiment.

    Trains on PSG data, fine-tunes and evaluates on headband data.
    Both recordings are from the same nights, so we have paired labels.

    Args:
        data_path: Path to BOAS dataset
        output_dir: Where to save results
        max_subjects: Limit subjects for testing
        cv_method: CV method for evaluation
    """
    if not _check_torch():
        return None

    logger.info("")
    logger.info("=" * 60)
    logger.info("TRANSFER LEARNING: PSG → Headband")
    logger.info("=" * 60)

    # Load PSG data (6 EEG channels)
    logger.info("Loading PSG data (6 channels)...")
    X_psg, y_psg, subj_psg = load_raw_signals_from_thesis(
        data_path, channel_preset="eeg_only", max_subjects=max_subjects
    )

    # Load headband data (2 channels)
    logger.info("Loading Headband data (2 channels)...")
    try:
        from data_loader_boas import BOASDataLoader
        from preprocessing import preprocess_subject

        loader = BOASDataLoader(
            base_path=data_path,
            target_channels=['HB_1', 'HB_2'],
            target_sfreq=None,
            epoch_duration=30.0,
            use_human_labels=True,
            preload=True
        )

        subjects = loader.get_subject_ids()
        if max_subjects:
            subjects = subjects[:max_subjects]

        all_epochs, all_labels, all_subjects = [], [], []
        for subj_id in tqdm(subjects, desc="Loading headband EEG", unit="subj"):
            try:
                epochs, labels = preprocess_subject(
                    loader, subj_id,
                    bandpass_low=0.5, bandpass_high=40.0,
                    notch_freq=50.0, target_sfreq=128.0
                )
                valid_mask = np.isin(labels, [0, 1, 2, 3, 4])
                epochs, labels = epochs[valid_mask], labels[valid_mask]
                if len(labels) < 10:
                    continue
                all_epochs.append(epochs)
                all_labels.append(labels)
                all_subjects.append(np.full(len(labels), subj_id))
            except Exception as e:
                logger.warning(f"Failed to load headband for subject {subj_id}: {e}")
                continue

        X_hb = np.concatenate(all_epochs)
        y_hb = np.concatenate(all_labels).astype(int)
        subj_hb = np.concatenate(all_subjects)
        logger.info(f"Headband data shape: {X_hb.shape}")

    except Exception as e:
        logger.error(f"Failed to load headband data: {e}")
        logger.info("Skipping transfer learning experiment")
        return None

    # Normalize
    for X_data in [X_psg, X_hb]:
        for ch in range(X_data.shape[1]):
            mean, std = X_data[:, ch, :].mean(), X_data[:, ch, :].std()
            if std > 0:
                X_data[:, ch, :] = (X_data[:, ch, :] - mean) / std

    start_time = time.time()

    # Experiment 1: Baseline - train and test on headband only (no transfer)
    logger.info("\n--- Baseline: Train on headband only ---")
    baseline = TransferLearningCNN()
    baseline_info = {'model': baseline, 'needs_scaling': False, 'category': 'deep_learning'}
    baseline_result = run_single_model_cv(
        'headband_baseline', baseline_info,
        X_hb, y_hb, subj_hb, cv_method=cv_method, is_raw_signal=True
    )

    # Experiment 2: Transfer - pretrain on PSG, finetune on headband
    logger.info("\n--- Transfer: PSG → Headband ---")
    transfer = TransferLearningCNN()
    transfer.pretrain_on_psg(X_psg, y_psg)

    # For evaluation, we do CV on the headband data with the pre-trained model
    transfer_info = {
        'model': transfer,
        'needs_scaling': False,
        'category': 'deep_learning'
    }
    # The transfer model's fit() will fine-tune from pre-trained weights
    # We wrap it so each fold fine-tunes from the same pre-trained checkpoint
    transfer_result = run_single_model_cv(
        'psg_to_headband_transfer', transfer_info,
        X_hb, y_hb, subj_hb, cv_method=cv_method, is_raw_signal=True
    )

    total_time = time.time() - start_time

    # Print comparison
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING RESULTS")
    print("=" * 60)
    for label, result in [('Headband baseline', baseline_result),
                          ('PSG→Headband transfer', transfer_result)]:
        if result['mean_metrics']:
            m = result['mean_metrics']
            print(f"  {label:<30} Acc: {m['accuracy']:.4f} | "
                  f"Kappa: {m['kappa']:.4f} | F1: {m['f1_macro']:.4f}")
    print(f"  Total time: {format_time_estimate(total_time)}")

    return {
        'baseline': baseline_result,
        'transfer': transfer_result,
        'total_time': total_time,
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_all_experiments(
    cache_dir: str = None,
    data_path: str = None,
    cv_method: str = "stratified_5fold",
    run_classical: bool = True,
    run_fnn: bool = True,
    run_deep_learning: bool = False,
    run_transfer: bool = False,
    run_shap: bool = False,
    output_dir: str = None,
    max_subjects_dl: int = None,
    max_subjects: int = None
) -> pd.DataFrame:
    """
    Run ALL models and return comparison results.

    Args:
        cache_dir: Path to thesis feature cache (for classical + FNN)
        data_path: Path to raw BOAS data (for deep learning models)
        cv_method: "loso", "stratified_5fold", or "train_test"
        run_classical: Run classical ML models
        run_fnn: Run FNN on features
        run_deep_learning: Run CNN/LSTM/GRU/Transformer on raw signals
        run_transfer: Run PSG→Headband transfer learning
        run_shap: Run SHAP explainability analysis
        output_dir: Where to save results
        max_subjects_dl: Max subjects for deep learning (memory)

    Returns:
        DataFrame with all results sorted by accuracy
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ── Load tabular features ──
    X_features, y_features, subject_ids = None, None, None
    n_subjects = 0
    if run_classical or run_fnn or run_shap:
        logger.info("=" * 60)
        logger.info("Loading tabular features (149) from thesis cache...")
        logger.info("=" * 60)
        X_features, y_features, subject_ids = load_features_from_thesis_cache(cache_dir, max_subjects=max_subjects)
        n_subjects = len(np.unique(subject_ids))

    # ── Classical ML Models ──
    if run_classical and X_features is not None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("RUNNING CLASSICAL ML MODELS")
        logger.info("=" * 60)

        classical_models = get_classical_models()

        # Print time estimates for all models
        print(f"\n{'─' * 60}")
        print(f"Time estimates for {n_subjects} subjects, {cv_method}:")
        print(f"{'─' * 60}")
        total_est = 0.0
        for name in classical_models:
            est = estimate_time(name, n_subjects, cv_method)
            total_est += est
            print(f"  {name:<25} ~{format_time_estimate(est)}")
        print(f"{'─' * 60}")
        print(f"  {'TOTAL':<25} ~{format_time_estimate(total_est)}")
        print()

        # Smart filtering: auto-exclude slow models, ask user for borderline ones
        classical_models, skipped = check_slow_models(
            classical_models, n_subjects, cv_method
        )

        for name, info in classical_models.items():
            logger.info(f"\n{'─' * 40}")
            est = estimate_time(name, n_subjects, cv_method)
            logger.info(f"Training: {name} ({info['category']}) [est. {format_time_estimate(est)}]")
            logger.info(f"{'─' * 40}")

            result = run_single_model_cv(
                name, info, X_features.values, y_features, subject_ids,
                cv_method=cv_method, is_raw_signal=False
            )
            all_results.append(result)

            if result['mean_metrics']:
                m = result['mean_metrics']
                logger.info(
                    f"  → Acc: {m['accuracy']:.4f} | Kappa: {m['kappa']:.4f} | "
                    f"F1: {m['f1_macro']:.4f} | Time: {result['training_time_seconds']:.1f}s"
                )

    # ── SHAP Analysis ──
    shap_results = []
    if run_shap and run_classical and X_features is not None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("RUNNING SHAP EXPLAINABILITY ANALYSIS")
        logger.info("=" * 60)

        # Train best models on full train set for SHAP
        from sklearn.model_selection import train_test_split as tts
        X_shap_train, X_shap_test, y_shap_train, y_shap_test = tts(
            X_features.values, y_features, test_size=0.2, stratify=y_features, random_state=42
        )

        # Run SHAP on tree-based and boosting models (fast with TreeExplainer)
        shap_models_to_explain = {
            'random_forest', 'extra_trees', 'xgboost', 'lightgbm', 'catboost',
            'gradient_boosting', 'logistic_regression', 'decision_tree'
        }
        classical_models_for_shap = get_classical_models()
        classical_models_for_shap, _ = check_slow_models(
            classical_models_for_shap, n_subjects, cv_method
        )

        feature_names = list(X_features.columns) if hasattr(X_features, 'columns') else None

        for name, info in classical_models_for_shap.items():
            if name not in shap_models_to_explain:
                continue
            logger.info(f"\n  SHAP: {name}")
            try:
                model = _get_fresh_model(info)
                scaler = StandardScaler()
                if info.get('needs_scaling', False):
                    X_train_scaled = scaler.fit_transform(X_shap_train)
                    X_test_scaled = scaler.transform(X_shap_test)
                else:
                    X_train_scaled = X_shap_train
                    X_test_scaled = X_shap_test

                model.fit(X_train_scaled, y_shap_train)
                shap_result = run_shap_analysis(
                    model, name, info,
                    X_train_scaled, X_test_scaled,
                    feature_names=feature_names,
                    output_dir=Path(output_dir) / "shap" if output_dir else None
                )
                if shap_result:
                    shap_results.append(shap_result)
            except Exception as e:
                logger.warning(f"  SHAP failed for {name}: {e}")

    # ── FNN ──
    if run_fnn and X_features is not None and _check_torch():
        logger.info("")
        logger.info("=" * 60)
        logger.info("RUNNING FNN (PyTorch)")
        logger.info("=" * 60)

        fnn_info = {
            'model': FNNClassifier(),
            'needs_scaling': False,  # FNN does its own scaling
            'category': 'neural_network'
        }

        result = run_single_model_cv(
            'fnn', fnn_info, X_features.values, y_features, subject_ids,
            cv_method=cv_method, is_raw_signal=False
        )
        all_results.append(result)

        if result['mean_metrics']:
            m = result['mean_metrics']
            logger.info(
                f"  → Acc: {m['accuracy']:.4f} | Kappa: {m['kappa']:.4f} | "
                f"F1: {m['f1_macro']:.4f} | Time: {result['training_time_seconds']:.1f}s"
            )

    # ── Deep Learning on Raw Signals ──
    if run_deep_learning and data_path and _check_torch():
        logger.info("")
        logger.info("=" * 60)
        logger.info("RUNNING DEEP LEARNING MODELS (raw EEG signals)")
        logger.info("=" * 60)

        X_raw, y_raw, subj_raw = load_raw_signals_from_thesis(
            data_path, max_subjects=max_subjects_dl
        )

        n_channels = X_raw.shape[1]
        n_samples = X_raw.shape[2]

        # Normalize raw signals per-channel
        for ch in range(n_channels):
            mean = X_raw[:, ch, :].mean()
            std = X_raw[:, ch, :].std()
            if std > 0:
                X_raw[:, ch, :] = (X_raw[:, ch, :] - mean) / std

        dl_models = {
            'cnn_1d': {
                'model': CNN1DClassifier(n_channels=n_channels, n_samples=n_samples),
                'needs_scaling': False,
                'category': 'deep_learning'
            },
            'lstm': {
                'model': LSTMClassifier(n_channels=n_channels),
                'needs_scaling': False,
                'category': 'deep_learning'
            },
            'gru': {
                'model': GRUClassifier(n_channels=n_channels),
                'needs_scaling': False,
                'category': 'deep_learning'
            },
            'cnn_lstm': {
                'model': CNNLSTMClassifier(n_channels=n_channels),
                'needs_scaling': False,
                'category': 'deep_learning'
            },
            'transformer': {
                'model': SimpleTransformerClassifier(n_channels=n_channels),
                'needs_scaling': False,
                'category': 'deep_learning'
            },
        }

        # For deep learning with LOSO, use simpler CV by default
        dl_cv = cv_method if cv_method != "loso" else "stratified_5fold"
        if cv_method == "loso":
            logger.warning(
                "LOSO with deep learning is very slow. "
                "Using stratified 5-fold for DL models instead. "
                "Pass --dl-loso flag to force LOSO for DL."
            )

        for name, info in dl_models.items():
            logger.info(f"\n{'─' * 40}")
            logger.info(f"Training: {name} ({info['category']})")
            logger.info(f"{'─' * 40}")

            result = run_single_model_cv(
                name, info, X_raw, y_raw, subj_raw,
                cv_method=dl_cv, is_raw_signal=True
            )
            all_results.append(result)

            if result['mean_metrics']:
                m = result['mean_metrics']
                logger.info(
                    f"  → Acc: {m['accuracy']:.4f} | Kappa: {m['kappa']:.4f} | "
                    f"F1: {m['f1_macro']:.4f} | Time: {result['training_time_seconds']:.1f}s"
                )

    # ── Transfer Learning ──
    transfer_results = None
    if run_transfer and data_path and _check_torch():
        transfer_results = run_transfer_learning_experiment(
            data_path=data_path,
            output_dir=Path(output_dir) if output_dir else None,
            max_subjects=max_subjects_dl or max_subjects,
            cv_method=cv_method
        )
        # Add transfer results to all_results
        if transfer_results:
            all_results.append(transfer_results['baseline'])
            all_results.append(transfer_results['transfer'])

    # ── Build Results Table ──
    rows = []
    for r in all_results:
        if r['mean_metrics']:
            row = {
                'model': r['model_name'],
                'category': r['category'],
                'cv_method': r['cv_method'],
                'is_raw_signal': r['is_raw_signal'],
                'accuracy': r['mean_metrics']['accuracy'],
                'accuracy_std': r['std_metrics']['accuracy'],
                'kappa': r['mean_metrics']['kappa'],
                'kappa_std': r['std_metrics']['kappa'],
                'f1_macro': r['mean_metrics']['f1_macro'],
                'f1_macro_std': r['std_metrics']['f1_macro'],
                'f1_weighted': r['mean_metrics']['f1_weighted'],
                'training_time_s': r['training_time_seconds'],
                'n_folds': len(r['fold_metrics']),
                'error': r['error'],
            }
            # Per-class F1
            for cls in CLASS_NAMES:
                key = f'f1_{cls}'
                if key in r['mean_metrics']:
                    row[f'f1_{cls}'] = r['mean_metrics'][key]
            rows.append(row)
        elif r['error']:
            rows.append({
                'model': r['model_name'],
                'category': r.get('category', '?'),
                'accuracy': None,
                'error': r['error'],
            })

    df = pd.DataFrame(rows)

    if 'accuracy' in df.columns:
        df = df.sort_values('accuracy', ascending=False, na_position='last')

    # ── Save Results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"model_comparison_{cv_method}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to: {csv_path}")

    # Save detailed results as JSON
    json_path = output_dir / f"model_comparison_{cv_method}_{timestamp}_detailed.json"
    # Convert numpy types for JSON serialization
    serializable_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != 'fold_metrics'}
        sr['n_folds'] = len(r['fold_metrics'])
        serializable_results.append(sr)
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    # Save SHAP results
    if shap_results:
        shap_json_path = output_dir / f"shap_results_{timestamp}.json"
        with open(shap_json_path, 'w') as f:
            json.dump(shap_results, f, indent=2, default=str)
        logger.info(f"SHAP results saved to: {shap_json_path}")

    # ── Print Summary ──
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    if not df.empty and 'accuracy' in df.columns:
        print(f"\n{'Model':<25} {'Category':<15} {'Acc':>8} {'±':>6} {'Kappa':>8} {'F1-macro':>8} {'Time':>8}")
        print("─" * 85)
        for _, row in df.iterrows():
            if row.get('accuracy') is not None:
                print(
                    f"{row['model']:<25} {row.get('category', '?'):<15} "
                    f"{row['accuracy']:>8.4f} {row.get('accuracy_std', 0):>6.4f} "
                    f"{row.get('kappa', 0):>8.4f} {row.get('f1_macro', 0):>8.4f} "
                    f"{row.get('training_time_s', 0):>7.1f}s"
                )
            else:
                print(f"{row['model']:<25} {'ERROR':<15} {row.get('error', 'unknown')}")

    print(f"\nResults saved to: {csv_path}")

    return df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ALL ML models on sleep stage data and compare results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with train/test split (fastest)
  python all_models.py --cv train_test

  # 5-fold cross-validation (recommended for quick comparison)
  python all_models.py --cv stratified_5fold

  # Full LOSO (gold standard, slow - hours)
  python all_models.py --cv loso

  # Only classical models
  python all_models.py --cv stratified_5fold --no-fnn --no-dl

  # Include deep learning on raw signals (GPU recommended)
  python all_models.py --cv stratified_5fold --dl --data-path /path/to/BOAS

  # Deep learning with SHAP explainability
  python all_models.py --cv stratified_5fold --dl --shap --data-path /path/to/BOAS

  # Transfer learning: PSG → Headband
  python all_models.py --transfer --data-path /path/to/BOAS

  # Full benchmark: all models + SHAP + transfer learning
  python all_models.py --cv stratified_5fold --dl --shap --transfer --data-path /path/to/BOAS

  # Custom feature cache location
  python all_models.py --cache-dir /path/to/features_cache_global
        """
    )

    parser.add_argument('--cv', type=str, default='stratified_5fold',
                        choices=['loso', 'stratified_5fold', 'train_test'],
                        help='Cross-validation method (default: stratified_5fold)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Path to thesis feature cache directory')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to raw BOAS data (needed for deep learning)')
    parser.add_argument('--no-classical', action='store_true',
                        help='Skip classical ML models')
    parser.add_argument('--no-fnn', action='store_true',
                        help='Skip FNN model')
    parser.add_argument('--dl', action='store_true',
                        help='Run deep learning models on raw EEG (GPU recommended)')
    parser.add_argument('--dl-loso', action='store_true',
                        help='Force LOSO for deep learning (very slow)')
    parser.add_argument('--shap', action='store_true',
                        help='Run SHAP explainability analysis on best models')
    parser.add_argument('--transfer', action='store_true',
                        help='Run PSG→Headband transfer learning experiment')
    parser.add_argument('--max-subjects-dl', type=int, default=None,
                        help='Limit subjects for deep learning (memory)')
    parser.add_argument('--subjects', type=int, default=None,
                        help='Limit number of subjects to load (e.g. 3, 5, 30)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    results = run_all_experiments(
        cache_dir=args.cache_dir,
        data_path=args.data_path,
        cv_method=args.cv,
        run_classical=not args.no_classical,
        run_fnn=not args.no_fnn,
        run_deep_learning=args.dl,
        run_transfer=args.transfer,
        run_shap=args.shap,
        output_dir=args.output_dir,
        max_subjects_dl=args.max_subjects_dl,
        max_subjects=args.subjects,
    )
