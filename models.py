"""
Model Implementations for Sleep Stage Classification
=====================================================

Implements the three models from the thesis configuration grid:
- XGBoost (gradient boosting) - IMPLEMENTED ✓ (used in experiments)
- Random Forest (ensemble trees) - IMPLEMENTED ✓ (used in experiments)
- FNN (feedforward neural network) - IMPLEMENTED ✓ (available but excluded)

Thesis Experimental Grid: 2 × 3 × 3 = 18 configurations
- 2 models: XGBoost, Random Forest
- 3 correlation thresholds: 0.75, 0.90, None
- 3 top-k features: 30, 50, None (all 149)

NOTE: FNN is fully implemented and functional but excluded from the main
experimental evaluation due to computational constraints (~10 additional
hours per configuration). The model remains in the codebase to demonstrate
framework extensibility. To include FNN, modify the grid in training.py.

Author: Lennart Gorzel
Date: December 2025
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, params: Dict[str, Any], random_seed: int = 42):
        self.params = params
        self.random_seed = random_seed
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for samples."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()


class XGBoostModel(BaseModel):
    """
    XGBoost classifier for sleep stage classification.
    
    Status: ✓ IMPLEMENTED
    
    Characteristics:
    - Fast training
    - Good accuracy
    - Deterministic with fixed seed
    - Handles imbalanced classes well
    """
    
    def __init__(self, params: Dict[str, Any] = None, random_seed: int = 42):
        default_params = {
            'max_depth': 6,
            'n_estimators': 200,
            'learning_rate': 0.1,
            'objective': 'multi:softmax',
            'num_class': 5,
            'random_state': random_seed,
            'eval_metric': 'mlogloss',
            'n_jobs': -1,
            'verbosity': 0  # Suppress warnings
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(default_params, random_seed)
        
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(**self.params)
            logger.info("XGBoost model initialized")
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostModel':
        """Fit XGBoost model."""
        logger.info(f"Training XGBoost on {X.shape[0]} samples, {X.shape[1]} features")
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def feature_importance(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        return self.model.feature_importances_


class RandomForestModel(BaseModel):
    """
    Random Forest classifier for sleep stage classification.
    
    Status: ✓ IMPLEMENTED
    
    Characteristics:
    - Robust baseline
    - Less prone to overfitting
    - Deterministic with fixed seed
    - Good feature importance
    """
    
    def __init__(self, params: Dict[str, Any] = None, random_seed: int = 42):
        default_params = {
            'n_estimators': 200,
            'max_depth': None,  # No limit
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': random_seed,
            'n_jobs': -1,
            'class_weight': 'balanced'  # Handle imbalanced classes
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(default_params, random_seed)
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**self.params)
            logger.info("Random Forest model initialized")
        except ImportError:
            raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """Fit Random Forest model."""
        logger.info(f"Training Random Forest on {X.shape[0]} samples, {X.shape[1]} features")
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def feature_importance(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        return self.model.feature_importances_


class FNNModel(BaseModel):
    """
    Feedforward Neural Network for sleep stage classification.
    
    Status: ✓ IMPLEMENTED (PyTorch)
    
    Architecture: input → 256 → 128 → 64 → 5 (softmax)
    
    Characteristics:
    - Deep learning approach
    - May have ±0.5% nondeterminism due to GPU/parallel ops
    - Requires more careful hyperparameter tuning
    - Potentially higher accuracy ceiling
    
    Note from thesis:
    "FNN nondeterminism ±0.5% expected due to GPU operations"
    This will be documented in Chapter 5 experimental results.
    """
    
    def __init__(self, params: Dict[str, Any] = None, random_seed: int = 42):
        default_params = {
            'hidden_dims': [256, 128, 64],
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 50,
            'random_state': random_seed,
            'early_stopping_patience': 5
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(default_params, random_seed)
        
        # Import PyTorch
        try:
            import torch
            import torch.nn as nn
            # Don't store module references - they can't be pickled
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            logger.warning("PyTorch not installed. FNN model will not work.")
            return
        
        # Set seeds for reproducibility (torch only, avoid global numpy seed)
        torch.manual_seed(random_seed)
        
        # Detect device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"FNN model initialized (device: {self.device})")
        
        self.model = None
        self.scaler = None
        self._n_features = None
    
    def _build_network(self, n_features: int, n_classes: int = 5):
        """Build the neural network architecture."""
        import torch.nn as nn
        
        hidden_dims = self.params['hidden_dims']
        dropout = self.params['dropout']
        
        layers = []
        in_dim = n_features
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, n_classes))
        
        return nn.Sequential(*layers)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FNNModel':
        """Fit FNN model using PyTorch."""
        if not self._torch_available:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        
        logger.info(f"Training FNN on {X.shape[0]} samples, {X.shape[1]} features")
        
        # Store feature count for later
        self._n_features = X.shape[1]
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = min(self.params['batch_size'], len(X))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Build network
        n_classes = len(np.unique(y))
        self.model = self._build_network(X.shape[1], n_classes).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.params['learning_rate']
        )
        
        # Training loop
        import copy
        epochs = self.params['epochs']
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        patience = self.params.get('early_stopping_patience', 5)
        
        self.model.train()
        
        # Progress bar for epochs
        pbar = tqdm(
            range(epochs),
            desc="    FNN Training",
            unit="epoch",
            leave=False,
            ncols=80
        )
        
        for epoch in pbar:
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            
            # Update progress bar with loss
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'best': f'{best_loss:.4f}'})
            
            # Early stopping with best model checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    pbar.close()
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.is_fitted = True
        logger.info(f"FNN training complete (best loss: {best_loss:.4f})")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        import torch
        
        # Standardize
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        import torch
        import torch.nn.functional as F
        
        # Standardize
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Predict probabilities
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = F.softmax(outputs, dim=1)
        
        return proba.cpu().numpy()


def create_model(model_type: str, params: Dict[str, Any] = None, random_seed: int = 42) -> BaseModel:
    """
    Factory function to create models.
    
    Args:
        model_type: 'xgboost', 'random_forest', or 'fnn'
        params: Model-specific parameters
        random_seed: Random seed for reproducibility
    
    Returns:
        Initialized model instance
    
    Raises:
        ValueError: If model_type is not recognized
    """
    models = {
        'xgboost': XGBoostModel,
        'random_forest': RandomForestModel,
        'fnn': FNNModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(models.keys())}")
    
    return models[model_type](params, random_seed)


def get_model_info() -> Dict[str, Dict]:
    """
    Get information about available models.
    
    Returns:
        Dictionary with model info including implementation status
    """
    return {
        'xgboost': {
            'name': 'XGBoost',
            'class': XGBoostModel,
            'implemented': True,
            'description': 'Gradient boosting classifier',
            'deterministic': True,
            'notes': 'Fast, accurate, handles imbalanced classes'
        },
        'random_forest': {
            'name': 'Random Forest',
            'class': RandomForestModel,
            'implemented': True,
            'description': 'Ensemble of decision trees',
            'deterministic': True,
            'notes': 'Robust baseline, good feature importance'
        },
        'fnn': {
            'name': 'Feedforward Neural Network',
            'class': FNNModel,
            'implemented': False,
            'description': 'Deep learning classifier',
            'deterministic': False,  # ±0.5% variance expected
            'notes': 'TODO: Expected ±0.5% nondeterminism with GPU'
        }
    }


# Model evaluation metrics
def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate model predictions against clinical targets.
    
    Clinical Targets (from thesis):
    - Accuracy ≥ 0.85
    - Cohen's Kappa ≥ 0.75
    - F1 Macro ≥ 0.80
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score, f1_score,
        classification_report, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }
    
    # Per-class F1 — force all 5 classes to ensure correct index-to-name mapping
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    for i, name in enumerate(class_names):
        metrics[f'f1_{name}'] = f1_per_class[i]
    
    # Clinical target checks
    metrics['meets_accuracy_target'] = metrics['accuracy'] >= 0.85
    metrics['meets_kappa_target'] = metrics['kappa'] >= 0.75
    metrics['meets_f1_target'] = metrics['f1_macro'] >= 0.80
    metrics['meets_all_targets'] = all([
        metrics['meets_accuracy_target'],
        metrics['meets_kappa_target'],
        metrics['meets_f1_target']
    ])
    
    return metrics


if __name__ == "__main__":
    # Test model creation
    print("Testing model implementations...")
    
    for model_type in ['xgboost', 'random_forest', 'fnn']:
        print(f"\n{model_type}:")
        try:
            model = create_model(model_type)
            print(f"  ✓ Created successfully")
            
            # Test with dummy data
            X = np.random.randn(100, 149)
            y = np.random.randint(0, 5, 100)
            
            model.fit(X, y)
            preds = model.predict(X)
            print(f"  ✓ Fit and predict work")
            
        except NotImplementedError as e:
            print(f"  ⚠ Not implemented: {str(e)[:50]}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")
