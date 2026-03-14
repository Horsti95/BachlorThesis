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
# Data Loading - Reuses thesis pipeline for feature extraction
# =============================================================================

def load_features_from_thesis_cache(
    cache_dir: str = None,
    data_path: str = None,
    channel_preset: str = "eeg_only"
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

    models['svm_rbf'] = {
        'model': SVC(
            kernel='rbf', probability=True, random_state=42,
            class_weight='balanced', C=10.0, gamma='scale',
            max_iter=5000, cache_size=1000
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
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42
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
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.model = nn.Sequential(*layers).to(self.device)

        # Training
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float('inf')
        patience_counter = 0
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X_scaled).to(self.device))
            return torch.argmax(out, dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X_scaled).to(self.device))
            return F.softmax(out, dim=1).cpu().numpy()


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
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build().to(self.device)

        # X shape: (n_epochs, n_channels, n_samples)
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            if avg < best_loss:
                best_loss = avg
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return torch.argmax(out, dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return F.softmax(out, dim=1).cpu().numpy()


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
        import torch
        import torch.nn as nn

        class LSTMNet(nn.Module):
            def __init__(self, n_ch, hidden, n_layers, n_cls):
                super().__init__()
                # Input: (batch, n_channels, n_samples) -> transpose -> (batch, n_samples, n_channels)
                self.lstm = nn.LSTM(n_ch, hidden, n_layers, batch_first=True, dropout=0.3)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(64, n_cls)
                )

            def forward(self, x):
                # x: (batch, channels, samples) -> (batch, samples, channels)
                x = x.transpose(1, 2)
                # Subsample time dimension for efficiency: take every 4th sample
                x = x[:, ::4, :]
                out, (h_n, _) = self.lstm(x)
                return self.classifier(h_n[-1])

        return LSTMNet(self.n_channels, self.hidden_size, self.num_layers, self.n_classes)

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build().to(self.device)

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            if avg < best_loss:
                best_loss = avg
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return torch.argmax(out, dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return F.softmax(out, dim=1).cpu().numpy()


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
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build().to(self.device)

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            if avg < best_loss:
                best_loss = avg
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return torch.argmax(out, dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return F.softmax(out, dim=1).cpu().numpy()


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
                # After CNN: (batch, 64, ~120) depending on input
                self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True)
                self.classifier = nn.Sequential(
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(32, n_cls)
                )

            def forward(self, x):
                # x: (batch, channels, samples)
                features = self.cnn(x)             # (batch, 64, T')
                features = features.transpose(1, 2)  # (batch, T', 64)
                _, (h_n, _) = self.lstm(features)
                return self.classifier(h_n[-1])

        return CNNLSTM(self.n_channels, self.n_classes)

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build().to(self.device)

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            if avg < best_loss:
                best_loss = avg
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return torch.argmax(out, dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return F.softmax(out, dim=1).cpu().numpy()


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
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build().to(self.device)

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            if avg < best_loss:
                best_loss = avg
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return torch.argmax(out, dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device))
            return F.softmax(out, dim=1).cpu().numpy()


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
# Main Experiment Runner
# =============================================================================

def run_all_experiments(
    cache_dir: str = None,
    data_path: str = None,
    cv_method: str = "stratified_5fold",
    run_classical: bool = True,
    run_fnn: bool = True,
    run_deep_learning: bool = False,
    output_dir: str = None,
    max_subjects_dl: int = None
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
    if run_classical or run_fnn:
        logger.info("=" * 60)
        logger.info("Loading tabular features (149) from thesis cache...")
        logger.info("=" * 60)
        X_features, y_features, subject_ids = load_features_from_thesis_cache(cache_dir)

    # ── Classical ML Models ──
    if run_classical and X_features is not None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("RUNNING CLASSICAL ML MODELS")
        logger.info("=" * 60)

        classical_models = get_classical_models()

        for name, info in classical_models.items():
            logger.info(f"\n{'─' * 40}")
            logger.info(f"Training: {name} ({info['category']})")
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

  # Include deep learning on raw signals
  python all_models.py --cv stratified_5fold --dl --data-path /path/to/BOAS

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
                        help='Run deep learning models (CNN, LSTM, etc.)')
    parser.add_argument('--dl-loso', action='store_true',
                        help='Force LOSO for deep learning (very slow)')
    parser.add_argument('--max-subjects-dl', type=int, default=None,
                        help='Limit subjects for deep learning (memory)')
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
        output_dir=args.output_dir,
        max_subjects_dl=args.max_subjects_dl,
    )
