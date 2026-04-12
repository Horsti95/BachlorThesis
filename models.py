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
import inspect
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import catboost  # noqa: F401
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

try:
    import lightgbm  # noqa: F401
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False


class BaseModel(ABC):
    """Abstract base class for all sleep stage classification models.

    Subclasses must implement fit, predict, and predict_proba.
    """

    def __init__(self, params: Dict[str, Any], random_seed: int = 42):
        """Initialize base model.

        Args:
            params: Model-specific hyperparameters.
            random_seed: Random seed for reproducibility.
        """
        self.params = params
        self.random_seed = random_seed
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Fit the model to training data.

        Args:
            X: Training feature matrix (n_samples, n_features).
            y: Target labels (n_samples,).

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict sleep stage labels for samples.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Predicted labels (n_samples,).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for each sleep stage.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Probability matrix (n_samples, n_classes).
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get a copy of the model's hyperparameters."""
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
        """Initialize XGBoost classifier with default or custom parameters.

        Args:
            params: Optional dict of XGBoost hyperparameters to override defaults.
            random_seed: Random seed for reproducibility.
        """
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
            logger.debug("XGBoost model initialized")
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostModel':
        """Train the XGBoost classifier on the provided data."""
        logger.debug(f"Training XGBoost on {X.shape[0]} samples, {X.shape[1]} features")
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
        """Get feature importances from the trained XGBoost model.

        Returns:
            Array of importance scores, one per feature.
        """
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
        """Initialize Random Forest classifier with default or custom parameters.

        Args:
            params: Optional dict of RF hyperparameters to override defaults.
            random_seed: Random seed for reproducibility.
        """
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
            logger.debug("Random Forest model initialized")
        except ImportError:
            raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """Train the Random Forest classifier on the provided data."""
        logger.debug(f"Training Random Forest on {X.shape[0]} samples, {X.shape[1]} features")
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
        """Get feature importances from the trained Random Forest model.

        Returns:
            Array of importance scores, one per feature.
        """
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
        """Initialize FNN model. Requires PyTorch.

        Args:
            params: Optional dict of hyperparameters (hidden_dims, dropout, lr, etc.).
            random_seed: Random seed for reproducibility.
        """
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
        logger.debug(f"FNN model initialized (device: {self.device})")
        
        self.model = None
        self.scaler = None
        self._n_features = None
    
    def _build_network(self, n_features: int, n_classes: int = 5):
        """Build the feedforward neural network as nn.Sequential.

        Args:
            n_features: Number of input features.
            n_classes: Number of output classes (default 5 sleep stages).

        Returns:
            nn.Sequential model with BatchNorm, ReLU, and Dropout layers.
        """
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
        """Train the FNN using PyTorch with early stopping.

        Scales features with StandardScaler, then trains via Adam optimizer
        with CrossEntropyLoss. Stops early if loss plateaus.
        """
        if not self._torch_available:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        
        logger.debug(f"Training FNN on {X.shape[0]} samples, {X.shape[1]} features")
        
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


class SklearnPipelineModel(BaseModel):
    """Generic wrapper for sklearn-style estimators with optional scaling."""

    ESTIMATOR_CLASS = None
    DEFAULT_PARAMS: Dict[str, Any] = {}
    USE_SCALER = False

    def __init__(self, params: Dict[str, Any] = None, random_seed: int = 42):
        default_params = self.DEFAULT_PARAMS.copy()
        if params:
            default_params.update(params)

        if self.ESTIMATOR_CLASS is None:
            raise NotImplementedError("ESTIMATOR_CLASS must be set in subclasses")

        try:
            estimator_signature = inspect.signature(self.ESTIMATOR_CLASS.__init__)
            if "random_state" in estimator_signature.parameters and "random_state" not in default_params:
                default_params["random_state"] = random_seed
        except (TypeError, ValueError):
            pass

        super().__init__(default_params, random_seed)

        try:
            if self.USE_SCALER:
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                self.model = make_pipeline(StandardScaler(), self.ESTIMATOR_CLASS(**self.params))
            else:
                self.model = self.ESTIMATOR_CLASS(**self.params)
        except ImportError as e:
            raise ImportError(f"Required dependency for {self.__class__.__name__} is not installed: {e}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SklearnPipelineModel':
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        raise RuntimeError(f"{self.__class__.__name__} does not support probability predictions")


class LinearSVMModel(SklearnPipelineModel):
    from sklearn.svm import SVC as ESTIMATOR_CLASS
    DEFAULT_PARAMS = {
        'kernel': 'linear',
        'C': 1.0,
        'probability': True,
        'class_weight': 'balanced',
    }
    USE_SCALER = True


class RBFSVMModel(SklearnPipelineModel):
    from sklearn.svm import SVC as ESTIMATOR_CLASS
    DEFAULT_PARAMS = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'probability': True,
        'class_weight': 'balanced',
    }
    USE_SCALER = True


class AdaBoostModel(SklearnPipelineModel):
    from sklearn.ensemble import AdaBoostClassifier as ESTIMATOR_CLASS
    DEFAULT_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 1.0,
    }


class GradientBoostingModel(SklearnPipelineModel):
    from sklearn.ensemble import GradientBoostingClassifier as ESTIMATOR_CLASS
    DEFAULT_PARAMS = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 3,
    }


class LogisticRegressionModel(SklearnPipelineModel):
    from sklearn.linear_model import LogisticRegression as ESTIMATOR_CLASS
    DEFAULT_PARAMS = {
        'max_iter': 1000,
        'solver': 'lbfgs',
        'multi_class': 'auto',
        'class_weight': 'balanced',
    }
    USE_SCALER = True


class DecisionTreeModel(SklearnPipelineModel):
    from sklearn.tree import DecisionTreeClassifier as ESTIMATOR_CLASS
    DEFAULT_PARAMS = {
        'class_weight': 'balanced',
    }


class RidgeClassifierModel(SklearnPipelineModel):
    from sklearn.linear_model import RidgeClassifier as ESTIMATOR_CLASS
    DEFAULT_PARAMS = {
        'alpha': 1.0,
        'class_weight': 'balanced',
    }
    USE_SCALER = True


class NaiveBayesModel(SklearnPipelineModel):
    from sklearn.naive_bayes import GaussianNB as ESTIMATOR_CLASS
    DEFAULT_PARAMS = {}


class ExtraTreesModel(SklearnPipelineModel):
    from sklearn.ensemble import ExtraTreesClassifier as ESTIMATOR_CLASS
    DEFAULT_PARAMS = {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'n_jobs': -1,
    }


class KNN5Model(SklearnPipelineModel):
    from sklearn.neighbors import KNeighborsClassifier as ESTIMATOR_CLASS
    DEFAULT_PARAMS = {
        'n_neighbors': 5,
        'weights': 'uniform',
        'n_jobs': -1,
    }
    USE_SCALER = True


class KNN10Model(SklearnPipelineModel):
    from sklearn.neighbors import KNeighborsClassifier as ESTIMATOR_CLASS
    DEFAULT_PARAMS = {
        'n_neighbors': 10,
        'weights': 'uniform',
        'n_jobs': -1,
    }
    USE_SCALER = True


class CatBoostModel(BaseModel):
    """CatBoost classifier wrapper with a graceful import error if unavailable."""

    def __init__(self, params: Dict[str, Any] = None, random_seed: int = 42):
        default_params = {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.1,
            'loss_function': 'MultiClass',
            'eval_metric': 'Accuracy',
            'random_seed': random_seed,
            'verbose': False,
            'allow_writing_files': False,
        }
        if params:
            default_params.update(params)

        super().__init__(default_params, random_seed)

        try:
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier(**self.params)
        except ImportError as e:
            raise ImportError("catboost is not installed. Run: pip install catboost") from e

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CatBoostModel':
        self.model.fit(X, y, verbose=False)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        preds = self.model.predict(X)
        return np.asarray(preds).reshape(-1).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class LightGBMModel(BaseModel):
    """LightGBM classifier wrapper with a graceful import error if unavailable."""

    def __init__(self, params: Dict[str, Any] = None, random_seed: int = 42):
        default_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'objective': 'multiclass',
            'num_class': 5,
            'random_state': random_seed,
            'n_jobs': -1,
            'verbosity': -1,
        }
        if params:
            default_params.update(params)

        super().__init__(default_params, random_seed)

        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(**self.params)
        except ImportError as e:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm") from e

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LightGBMModel':
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


def create_model(model_type: str, params: Dict[str, Any] = None, random_seed: int = 42) -> BaseModel:
    """
    Factory function to create models.
    
    Args:
        model_type: Model name string such as 'xgboost', 'random_forest',
            'svm_linear', 'svm_rbf', 'adaboost', 'gradient_boosting',
            'logistic_regression', 'decision_tree', 'catboost',
            'ridge_classifier', 'naive_bayes', 'lightgbm', 'knn_5',
            'knn_10', 'extra_trees', or 'fnn'
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
        'fnn': FNNModel,
        'svm_linear': LinearSVMModel,
        'svm_rbf': RBFSVMModel,
        'adaboost': AdaBoostModel,
        'gradient_boosting': GradientBoostingModel,
        'logistic_regression': LogisticRegressionModel,
        'decision_tree': DecisionTreeModel,
        'catboost': CatBoostModel,
        'ridge_classifier': RidgeClassifierModel,
        'naive_bayes': NaiveBayesModel,
        'lightgbm': LightGBMModel,
        'knn_5': KNN5Model,
        'knn_10': KNN10Model,
        'extra_trees': ExtraTreesModel,
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
        'svm_linear': {
            'name': 'Linear SVM',
            'class': LinearSVMModel,
            'implemented': True,
            'description': 'Support Vector Machine with linear kernel',
            'deterministic': True,
            'notes': 'Uses StandardScaler and probability=True'
        },
        'adaboost': {
            'name': 'AdaBoost',
            'class': AdaBoostModel,
            'implemented': True,
            'description': 'Boosted decision stumps / trees',
            'deterministic': True,
            'notes': 'Sklearn AdaBoostClassifier'
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting',
            'class': GradientBoostingModel,
            'implemented': True,
            'description': 'Gradient boosting classifier',
            'deterministic': True,
            'notes': 'Sklearn GradientBoostingClassifier'
        },
        'logistic_regression': {
            'name': 'Logistic Regression',
            'class': LogisticRegressionModel,
            'implemented': True,
            'description': 'Multinomial logistic regression',
            'deterministic': True,
            'notes': 'Uses StandardScaler'
        },
        'decision_tree': {
            'name': 'Decision Tree',
            'class': DecisionTreeModel,
            'implemented': True,
            'description': 'Single decision tree classifier',
            'deterministic': True,
            'notes': 'Sklearn DecisionTreeClassifier'
        },
        'catboost': {
            'name': 'CatBoost',
            'class': CatBoostModel,
            'implemented': _CATBOOST_AVAILABLE,
            'description': 'CatBoost gradient boosting',
            'deterministic': True,
            'notes': 'Requires catboost package'
        },
        'ridge_classifier': {
            'name': 'Ridge Classifier',
            'class': RidgeClassifierModel,
            'implemented': True,
            'description': 'Linear classifier with L2 regularization',
            'deterministic': True,
            'notes': 'Uses StandardScaler'
        },
        'naive_bayes': {
            'name': 'Naive Bayes',
            'class': NaiveBayesModel,
            'implemented': True,
            'description': 'Gaussian Naive Bayes classifier',
            'deterministic': True,
            'notes': 'Fast probabilistic baseline'
        },
        'lightgbm': {
            'name': 'LightGBM',
            'class': LightGBMModel,
            'implemented': _LIGHTGBM_AVAILABLE,
            'description': 'LightGBM gradient boosting',
            'deterministic': True,
            'notes': 'Requires lightgbm package'
        },
        'knn_5': {
            'name': 'KNN (k=5)',
            'class': KNN5Model,
            'implemented': True,
            'description': 'K-nearest neighbors with k=5',
            'deterministic': True,
            'notes': 'Uses StandardScaler'
        },
        'knn_10': {
            'name': 'KNN (k=10)',
            'class': KNN10Model,
            'implemented': True,
            'description': 'K-nearest neighbors with k=10',
            'deterministic': True,
            'notes': 'Uses StandardScaler'
        },
        'extra_trees': {
            'name': 'Extra Trees',
            'class': ExtraTreesModel,
            'implemented': True,
            'description': 'Extremely randomized trees ensemble',
            'deterministic': True,
            'notes': 'Sklearn ExtraTreesClassifier'
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
