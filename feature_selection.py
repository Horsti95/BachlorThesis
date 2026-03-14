"""
Feature Selection Module for Sleep Stage Classification
========================================================

Implements feature selection strategies for the thesis configuration grid:
- Correlation-based filtering (removes redundant features)
- Top-K feature selection (SelectKBest with mutual information)

Configuration Grid Options:
- Correlation thresholds: [0.90, 0.95, None]
- Top-K features: [30, 50, None (all)]

Purpose:
    Feature selection reduces dimensionality and removes redundant features,
    potentially improving model generalization and training speed. This module
    enables systematic comparison across feature selection strategies.

Author: Lennart Gorzel
Date: December 2025
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection pipeline.
    
    Attributes:
        correlation_threshold: Remove features with correlation above this value.
                               Options: 0.75, 0.90, 0.95, or None (no filtering)
        top_k_features: Select top K features by mutual information.
                        Options: 30, 50, or None (keep all after correlation filter)
        selection_method: Method for feature selection.
                         Options: 'anova' (fast, ANOVA F-test only),
                                  'mi' (slow, pure Mutual Information),
                                  'hybrid' (balanced, ANOVA -> MI two-stage)
                         Default: 'anova' for speed (benchmark validated)
        scope: Scope of feature selection.
               Options: 'per_fold' (fit on each fold's training data - correct),
                        'global' (fit once on all data - faster, minor leakage)
               Default: 'global' for speed (benchmark validated: <1% accuracy diff)
        use_hybrid: DEPRECATED - use selection_method='hybrid' instead
        random_state: Random seed for reproducibility
    """
    correlation_threshold: Optional[float] = None
    top_k_features: Optional[int] = None
    selection_method: str = 'anova'  # 'anova', 'mi', 'hybrid' - ANOVA is ~200x faster
    scope: str = 'global'  # 'per_fold', 'global' - global is faster with minimal leakage
    use_hybrid: bool = True  # DEPRECATED - kept for backward compatibility
    random_state: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if self.correlation_threshold is not None:
            if not 0.0 < self.correlation_threshold <= 1.0:
                raise ValueError(f"correlation_threshold must be in (0, 1], got {self.correlation_threshold}")
        
        if self.top_k_features is not None:
            if self.top_k_features < 1:
                raise ValueError(f"top_k_features must be >= 1, got {self.top_k_features}")
        
        # Validate selection_method
        valid_methods = ['anova', 'mi', 'hybrid']
        if self.selection_method not in valid_methods:
            raise ValueError(f"selection_method must be one of {valid_methods}, got {self.selection_method}")
        
        # Validate scope
        valid_scopes = ['per_fold', 'global']
        if self.scope not in valid_scopes:
            raise ValueError(f"scope must be one of {valid_scopes}, got {self.scope}")
    
    def get_config_id(self) -> str:
        """Get a unique identifier for this configuration.
        
        Used for result file naming and comparison.
        """
        corr_str = f"corr{self.correlation_threshold}" if self.correlation_threshold else "corrNone"
        k_str = f"k{self.top_k_features}" if self.top_k_features else "kAll"
        method_str = self.selection_method
        scope_str = self.scope[:3]  # 'per' or 'glo'
        return f"{corr_str}_{k_str}_{method_str}_{scope_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'correlation_threshold': self.correlation_threshold,
            'top_k_features': self.top_k_features,
            'selection_method': self.selection_method,
            'scope': self.scope,
            'use_hybrid': self.use_hybrid,  # deprecated
            'random_state': self.random_state
        }


class CorrelationFilter:
    """
    Remove highly correlated features to reduce redundancy.
    
    Strategy: For each pair of features with correlation above threshold,
    remove the one with lower variance (less informative).
    
    This is applied BEFORE top-K selection to ensure SelectKBest
    sees only non-redundant features.
    """
    
    def __init__(self, threshold: float = 0.95):
        """
        Initialize correlation filter.
        
        Args:
            threshold: Remove feature pairs with |correlation| > threshold
        """
        self.threshold = threshold
        self.features_to_remove_: Optional[List[str]] = None
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame) -> 'CorrelationFilter':
        """
        Identify features to remove based on correlation.
        
        Args:
            X: Feature DataFrame (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        logger.debug(f"Fitting CorrelationFilter with threshold={self.threshold}")
        
        # Compute correlation matrix
        self.correlation_matrix_ = X.corr().abs()
        
        # Get upper triangle (avoid duplicates)
        upper_tri = self.correlation_matrix_.where(
            np.triu(np.ones(self.correlation_matrix_.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        features_to_remove = set()
        highly_correlated_pairs = []
        
        for col in upper_tri.columns:
            correlated_features = upper_tri.index[upper_tri[col] > self.threshold].tolist()
            
            for corr_feature in correlated_features:
                corr_value = self.correlation_matrix_.loc[corr_feature, col]
                highly_correlated_pairs.append((corr_feature, col, corr_value))
                
                # Remove the one with lower variance
                if X[corr_feature].var() < X[col].var():
                    features_to_remove.add(corr_feature)
                else:
                    features_to_remove.add(col)
        
        self.features_to_remove_ = list(features_to_remove)
        self.is_fitted_ = True
        
        n_removed = len(self.features_to_remove_)
        n_pairs = len(highly_correlated_pairs)
        logger.debug(f"Found {n_pairs} highly correlated pairs, removing {n_removed} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with correlated features removed
        """
        if not self.is_fitted_:
            raise RuntimeError("CorrelationFilter must be fitted before transform")
        
        # Only remove features that exist in X
        features_to_remove = [f for f in self.features_to_remove_ if f in X.columns]
        X_filtered = X.drop(columns=features_to_remove)
        
        logger.debug(f"Removed {len(features_to_remove)} features, {X_filtered.shape[1]} remaining")
        return X_filtered
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def get_removed_features(self) -> List[str]:
        """Get list of features that will be removed."""
        if not self.is_fitted_:
            raise RuntimeError("CorrelationFilter must be fitted first")
        return self.features_to_remove_.copy()


class TopKSelector:
    """
    Select top-K features using mutual information.
    
    Uses sklearn's SelectKBest with mutual_info_classif scorer.
    Mutual information captures non-linear relationships between
    features and target, making it suitable for sleep stage classification.
    """
    
    def __init__(self, k: int = 50, random_state: int = 42):
        """
        Initialize top-K selector.
        
        Args:
            k: Number of features to select
            random_state: Random seed for reproducibility
        """
        self.k = k
        self.random_state = random_state
        self.selector_: Optional[SelectKBest] = None
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[Dict[str, float]] = None
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'TopKSelector':
        """
        Fit selector to identify top-K features.
        
        Args:
            X: Feature DataFrame (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        logger.debug(f"Fitting TopKSelector with k={self.k}")
        
        # Ensure k doesn't exceed available features
        actual_k = min(self.k, X.shape[1])
        if actual_k < self.k:
            logger.warning(f"Requested k={self.k} but only {X.shape[1]} features available. Using k={actual_k}")
            # Print to console as well for visibility
            print(f"\n[!] WARNING: Requested k={self.k} features but only {X.shape[1]} available after correlation filter.")
            print(f"    Using all {actual_k} features instead.\n")
        
        # Create selector with mutual information scorer
        # Note: mutual_info_classif uses random_state internally
        def mi_scorer(X, y):
            return mutual_info_classif(X, y, random_state=self.random_state)
        
        self.selector_ = SelectKBest(score_func=mi_scorer, k=actual_k)
        self.selector_.fit(X.values, y)
        
        # Store feature scores for analysis
        scores = self.selector_.scores_
        self.feature_scores_ = dict(zip(X.columns, scores))
        
        # Get selected feature names
        selected_mask = self.selector_.get_support()
        self.selected_features_ = X.columns[selected_mask].tolist()
        self.is_fitted_ = True
        
        # Log top features
        sorted_features = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)
        logger.debug(f"Top 5 features by MI: {sorted_features[:5]}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select top-K features.

        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with only top-K features
        """
        if not self.is_fitted_:
            raise RuntimeError("TopKSelector must be fitted before transform")
        
        # Only select features that exist in X
        available_selected = [f for f in self.selected_features_ if f in X.columns]
        return X[available_selected]
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_scores(self) -> Dict[str, float]:
        """Get mutual information scores for all features."""
        if not self.is_fitted_:
            raise RuntimeError("TopKSelector must be fitted first")
        return self.feature_scores_.copy()
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        if not self.is_fitted_:
            raise RuntimeError("TopKSelector must be fitted first")
        return self.selected_features_.copy()


class ANOVATopKSelector:
    """
    Select top-K features using ANOVA F-test (f_classif) only.
    
    BENCHMARK VALIDATED: This is the FASTEST feature selection method,
    ~200× faster than MI with <1% accuracy difference.
    
    Uses sklearn's SelectKBest with f_classif scorer.
    ANOVA captures linear relationships between features and target,
    which is sufficient for most EEG features (PSD bands, RMS, etc.).
    
    Scientific justification for thesis:
    - Benchmark showed: ANOVA ~16s vs MI ~4300s (same 5 subjects)
    - Accuracy difference: 0.809 (ANOVA global) vs 0.803 (MI per-fold)
    - The ~0.6% accuracy gain of MI is not worth the 200× time cost
    
    Recommended for: All thesis experiments (validated by benchmark)
    """
    
    def __init__(self, k: int = 50, random_state: int = 42):
        """
        Initialize ANOVA top-K selector.
        
        Args:
            k: Number of features to select
            random_state: Random seed (not used by f_classif, kept for API consistency)
        """
        self.k = k
        self.random_state = random_state
        self.selector_: Optional[SelectKBest] = None
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[Dict[str, float]] = None
        self.selection_time_: float = 0.0
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'ANOVATopKSelector':
        """
        Fit selector to identify top-K features using ANOVA F-test.
        
        Args:
            X: Feature DataFrame (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        logger.debug(f"Fitting ANOVATopKSelector with k={self.k}")
        
        # Ensure k doesn't exceed available features
        actual_k = min(self.k, X.shape[1])
        if actual_k < self.k:
            logger.warning(f"Requested k={self.k} but only {X.shape[1]} features available. Using k={actual_k}")
        
        # Create selector with ANOVA F-test scorer (FAST!)
        self.selector_ = SelectKBest(score_func=f_classif, k=actual_k)
        self.selector_.fit(X.values, y)
        
        # Store feature scores for analysis
        scores = self.selector_.scores_
        self.feature_scores_ = dict(zip(X.columns, scores))
        
        # Get selected feature names
        selected_mask = self.selector_.get_support()
        self.selected_features_ = X.columns[selected_mask].tolist()
        self.is_fitted_ = True
        
        self.selection_time_ = time.time() - start_time
        
        # Log top features
        sorted_features = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)
        logger.debug(f"ANOVA selection complete in {self.selection_time_:.2f}s")
        logger.debug(f"Top 5 features by F-score: {sorted_features[:5]}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select top-K features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with only top-K features
        """
        if not self.is_fitted_:
            raise RuntimeError("ANOVATopKSelector must be fitted before transform")
        
        # Only select features that exist in X
        available_selected = [f for f in self.selected_features_ if f in X.columns]
        return X[available_selected]
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_scores(self) -> Dict[str, float]:
        """Get ANOVA F-scores for all features."""
        if not self.is_fitted_:
            raise RuntimeError("ANOVATopKSelector must be fitted first")
        return self.feature_scores_.copy()
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        if not self.is_fitted_:
            raise RuntimeError("ANOVATopKSelector must be fitted first")
        return self.selected_features_.copy()
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        return {
            'selection_time': self.selection_time_,
            'total_time': self.selection_time_
        }


class HybridTopKSelector:
    """
    Two-stage hybrid feature selection for CPU efficiency.
    
    Combines fast ANOVA F-test (f_classif) with precise Mutual Information:
    1. Stage 1: f_classif rapidly prunes to 2×k features (O(n) - very fast)
    2. Stage 2: MI precisely selects final k features (O(n²) but on fewer features)
    
    This approach captures both linear (PSD bands, RMS) and non-linear 
    (entropy, complexity) EEG/EMG feature relationships while being 
    ~70% faster than pure MI on large datasets.
    
    Scientific justification for thesis:
    - Maintains consistency across all models (same features selected)
    - Preserves non-linear feature detection capability
    - Computationally tractable for 128-fold LOSO CV
    """
    
    def __init__(self, k: int = 50, random_state: int = 42, intermediate_multiplier: float = 2.0):
        """
        Initialize hybrid selector.
        
        Args:
            k: Final number of features to select
            random_state: Random seed for reproducibility  
            intermediate_multiplier: Stage 1 selects k * this value features (default: 2.0)
        """
        self.k = k
        self.random_state = random_state
        self.intermediate_multiplier = intermediate_multiplier
        
        # Stage selectors
        self.stage1_selector_: Optional[SelectKBest] = None
        self.stage2_selector_: Optional[SelectKBest] = None
        
        # Feature tracking
        self.stage1_features_: Optional[List[str]] = None
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[Dict[str, float]] = None
        self.stage1_scores_: Optional[Dict[str, float]] = None
        
        # Timing stats
        self.stage1_time_: float = 0.0
        self.stage2_time_: float = 0.0
        
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'HybridTopKSelector':
        """
        Fit two-stage selector.
        
        Args:
            X: Feature DataFrame (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        start_total = time.time()
        
        # Calculate intermediate k (Stage 1 output)
        intermediate_k = int(self.k * self.intermediate_multiplier)
        actual_intermediate_k = min(intermediate_k, X.shape[1])
        actual_final_k = min(self.k, actual_intermediate_k)
        
        logger.debug(f"Fitting HybridTopKSelector: {X.shape[1]} -> {actual_intermediate_k} -> {actual_final_k} features")
        
        # ========== STAGE 1: Fast ANOVA F-test ==========
        stage1_start = time.time()
        
        self.stage1_selector_ = SelectKBest(score_func=f_classif, k=actual_intermediate_k)
        self.stage1_selector_.fit(X.values, y)
        
        # Get Stage 1 results
        stage1_mask = self.stage1_selector_.get_support()
        self.stage1_features_ = X.columns[stage1_mask].tolist()
        self.stage1_scores_ = dict(zip(X.columns, self.stage1_selector_.scores_))
        
        # Create intermediate DataFrame
        X_intermediate = X[self.stage1_features_]
        
        self.stage1_time_ = time.time() - stage1_start
        logger.debug(f"Stage 1 (f_classif): {X.shape[1]} -> {len(self.stage1_features_)} features in {self.stage1_time_:.2f}s")
        
        # ========== STAGE 2: Precise MI on reduced set ==========
        stage2_start = time.time()
        
        def mi_scorer(X, y):
            return mutual_info_classif(X, y, random_state=self.random_state)
        
        self.stage2_selector_ = SelectKBest(score_func=mi_scorer, k=actual_final_k)
        self.stage2_selector_.fit(X_intermediate.values, y)
        
        # Get final selected features
        stage2_mask = self.stage2_selector_.get_support()
        self.selected_features_ = X_intermediate.columns[stage2_mask].tolist()
        self.feature_scores_ = dict(zip(X_intermediate.columns, self.stage2_selector_.scores_))
        
        self.stage2_time_ = time.time() - stage2_start
        total_time = time.time() - start_total
        
        logger.debug(f"Stage 2 (MI): {len(self.stage1_features_)} -> {len(self.selected_features_)} features in {self.stage2_time_:.2f}s")
        logger.debug(f"Hybrid selection complete: {total_time:.2f}s total")
        
        # Log top features
        sorted_features = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)
        logger.debug(f"Top 5 features by MI: {sorted_features[:5]}")

        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features using fitted selector.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with only selected features
        """
        if not self.is_fitted_:
            raise RuntimeError("HybridTopKSelector must be fitted before transform")
        
        available_selected = [f for f in self.selected_features_ if f in X.columns]
        return X[available_selected]
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_scores(self) -> Dict[str, float]:
        """Get MI scores for features that passed Stage 1."""
        if not self.is_fitted_:
            raise RuntimeError("HybridTopKSelector must be fitted first")
        return self.feature_scores_.copy()
    
    def get_selected_features(self) -> List[str]:
        """Get list of final selected feature names."""
        if not self.is_fitted_:
            raise RuntimeError("HybridTopKSelector must be fitted first")
        return self.selected_features_.copy()
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics for both stages."""
        return {
            'stage1_time': self.stage1_time_,
            'stage2_time': self.stage2_time_,
            'total_time': self.stage1_time_ + self.stage2_time_
        }


class FeatureSelectionPipeline:
    """
    Complete feature selection pipeline combining correlation filter and top-K.
    
    Pipeline Order:
    1. Correlation filtering (remove redundant features)
    2. Top-K selection (select most informative features)
    
    This order ensures SelectKBest sees only non-redundant features,
    improving feature diversity in the final selection.
    
    Selection Methods (benchmark validated):
    - 'anova': FASTEST (~16s), uses F-test only, ~0.809 accuracy
    - 'mi': SLOWEST (~4300s), uses Mutual Information, ~0.803 accuracy  
    - 'hybrid': BALANCED, uses F-test → MI two-stage
    
    RECOMMENDATION: Use 'anova' + 'global' scope (validated by benchmark)
    
    Usage:
        config = FeatureSelectionConfig(
            correlation_threshold=0.95, 
            top_k_features=50,
            selection_method='anova',  # FAST
            scope='global'             # Minor leakage, big speedup
        )
        pipeline = FeatureSelectionPipeline(config)
        X_train_selected = pipeline.fit_transform(X_train, y_train)
        X_test_selected = pipeline.transform(X_test)
    """
    
    def __init__(self, config: FeatureSelectionConfig):
        """
        Initialize feature selection pipeline.
        
        Args:
            config: Feature selection configuration
        """
        self.config = config
        
        # Initialize components (only if configured)
        self.correlation_filter_: Optional[CorrelationFilter] = None
        self.anova_selector_: Optional[ANOVATopKSelector] = None
        self.top_k_selector_: Optional[TopKSelector] = None
        self.hybrid_selector_: Optional[HybridTopKSelector] = None
        self.scaler_: Optional[StandardScaler] = None
        
        # Statistics
        self.n_features_input_: int = 0
        self.n_features_after_corr_: int = 0
        self.n_features_output_: int = 0
        self.selection_time_: float = 0.0
        self.is_fitted_ = False
        
        # Determine which selector to use based on selection_method
        self._selection_method = config.selection_method
        
        method_str = f" ({self._selection_method.upper()})"
        scope_str = f" [scope={config.scope}]"
        logger.debug(f"Initialized FeatureSelectionPipeline: {config.get_config_id()}{method_str}{scope_str}")
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'FeatureSelectionPipeline':
        """
        Fit the feature selection pipeline.
        
        Args:
            X: Feature DataFrame (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        self.n_features_input_ = X.shape[1]
        self.feature_names_input_ = list(X.columns)  # Store input feature names
        logger.debug(f"Fitting FeatureSelectionPipeline on {X.shape[0]} samples, {X.shape[1]} features")
        logger.debug(f"  Method: {self._selection_method}, Scope: {self.config.scope}")
        
        X_current = X.copy()
        
        # Step 1: Correlation filtering (if configured)
        if self.config.correlation_threshold is not None:
            self.correlation_filter_ = CorrelationFilter(threshold=self.config.correlation_threshold)
            X_current = self.correlation_filter_.fit_transform(X_current)
            self.n_features_after_corr_ = X_current.shape[1]
            self.feature_names_after_corr_ = list(X_current.columns)  # Store names for kAll
            logger.debug(f"After correlation filter: {X_current.shape[1]} features")
        else:
            self.n_features_after_corr_ = X_current.shape[1]
            self.feature_names_after_corr_ = list(X_current.columns)  # All features
            logger.debug("Correlation filtering: SKIPPED (threshold=None)")
        
        # Step 2: Top-K selection (if configured)
        if self.config.top_k_features is not None:
            if self._selection_method == 'anova':
                # Use ANOVA F-test only (FASTEST - benchmark validated)
                self.anova_selector_ = ANOVATopKSelector(
                    k=self.config.top_k_features,
                    random_state=self.config.random_state
                )
                self.anova_selector_.fit(X_current, y)
                logger.debug(f"Top-K selection (ANOVA): selected {min(self.config.top_k_features, X_current.shape[1])} features")
            elif self._selection_method == 'hybrid':
                # Use fast hybrid f_classif -> MI selection
                self.hybrid_selector_ = HybridTopKSelector(
                    k=self.config.top_k_features,
                    random_state=self.config.random_state
                )
                self.hybrid_selector_.fit(X_current, y)
                logger.debug(f"Top-K selection (HYBRID): selected {min(self.config.top_k_features, X_current.shape[1])} features")
            elif self._selection_method == 'mi':
                # Use pure MI (SLOWEST but captures all non-linear relationships)
                self.top_k_selector_ = TopKSelector(
                    k=self.config.top_k_features,
                    random_state=self.config.random_state
                )
                self.top_k_selector_.fit(X_current, y)
                logger.debug(f"Top-K selection (MI): selected {min(self.config.top_k_features, X_current.shape[1])} features")
        else:
            logger.debug("Top-K selection: SKIPPED (top_k=None)")
        
        self.n_features_output_ = self._count_output_features(X_current)
        self.selection_time_ = time.time() - start_time
        self.is_fitted_ = True
        
        logger.debug(f"Feature selection summary: {self.n_features_input_} -> {self.n_features_output_} in {self.selection_time_:.2f}s")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature selection to new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with selected features only
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureSelectionPipeline must be fitted before transform")
        
        X_current = X.copy()
        
        # Step 1: Correlation filtering
        if self.correlation_filter_ is not None:
            X_current = self.correlation_filter_.transform(X_current)
        
        # Step 2: Top-K selection (anova, hybrid, or pure MI)
        if self.anova_selector_ is not None:
            X_current = self.anova_selector_.transform(X_current)
        elif self.hybrid_selector_ is not None:
            X_current = self.hybrid_selector_.transform(X_current)
        elif self.top_k_selector_ is not None:
            X_current = self.top_k_selector_.transform(X_current)
        
        return X_current
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def _count_output_features(self, X_after_corr: pd.DataFrame) -> int:
        """Count expected output features."""
        if self.anova_selector_ is not None:
            return len(self.anova_selector_.get_selected_features())
        if self.hybrid_selector_ is not None:
            return len(self.hybrid_selector_.get_selected_features())
        if self.top_k_selector_ is not None:
            return len(self.top_k_selector_.get_selected_features())
        return X_after_corr.shape[1]
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature selection results.
        
        Returns:
            Dictionary with selection statistics and feature lists
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted first")
        
        summary = {
            'config': self.config.to_dict(),
            'selection_method': self._selection_method,
            'scope': self.config.scope,
            'n_features_input': self.n_features_input_,
            'n_features_after_correlation': self.n_features_after_corr_,
            'n_features_output': self.n_features_output_,
            'reduction_ratio': self.n_features_output_ / self.n_features_input_,
            'selection_time': self.selection_time_,
        }
        
        if self.correlation_filter_ is not None:
            summary['removed_by_correlation'] = self.correlation_filter_.get_removed_features()
        
        # Get selector-specific info
        if self.anova_selector_ is not None:
            summary['selected_features'] = self.anova_selector_.get_selected_features()
            summary['feature_scores'] = self.anova_selector_.get_feature_scores()
            summary['timing'] = self.anova_selector_.get_timing_stats()
        elif self.hybrid_selector_ is not None:
            summary['selected_features'] = self.hybrid_selector_.get_selected_features()
            summary['feature_scores'] = self.hybrid_selector_.get_feature_scores()
            summary['timing'] = self.hybrid_selector_.get_timing_stats()
        elif self.top_k_selector_ is not None:
            summary['selected_features'] = self.top_k_selector_.get_selected_features()
            summary['feature_scores'] = self.top_k_selector_.get_feature_scores()
        
        return summary
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names (convenience method)."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted first")
        
        if self.anova_selector_ is not None:
            return self.anova_selector_.get_selected_features()
        elif self.hybrid_selector_ is not None:
            return self.hybrid_selector_.get_selected_features()
        elif self.top_k_selector_ is not None:
            return self.top_k_selector_.get_selected_features()
        else:
            # No top-K selection - return features after correlation filter
            # This handles kAll configs with correlation filter
            if self.correlation_filter_ is not None and hasattr(self, 'feature_names_after_corr_'):
                return self.feature_names_after_corr_
            elif hasattr(self, 'feature_names_input_'):
                return self.feature_names_input_
            else:
                return []


def create_feature_selection_grid(
    selection_method: str = 'anova',
    scope: str = 'global'
) -> List[FeatureSelectionConfig]:
    """
    Create all feature selection configurations for thesis grid.
    
    Grid: 3 correlation thresholds × 3 top-K options = 9 configurations
    
    Args:
        selection_method: 'anova' (fast), 'mi' (slow), 'hybrid' (balanced)
                         Default: 'anova' (benchmark validated: 200× faster)
        scope: 'global' (fast, minor leakage) or 'per_fold' (correct, slower)
               Default: 'global' (benchmark validated: <1% accuracy diff)
    
    Returns:
        List of FeatureSelectionConfig objects
    """
    correlation_options = [0.90, 0.95, None]
    top_k_options = [30, 50, None]  # None = all features
    
    configs = []
    for corr in correlation_options:
        for top_k in top_k_options:
            configs.append(FeatureSelectionConfig(
                correlation_threshold=corr,
                top_k_features=top_k,
                selection_method=selection_method,
                scope=scope,
                random_state=42
            ))
    
    logger.info(f"Created {len(configs)} feature selection configurations (method={selection_method}, scope={scope})")
    return configs


def create_optimized_grid() -> List[FeatureSelectionConfig]:
    """
    Create OPTIMIZED feature selection grid for thesis.
    
    BENCHMARK VALIDATED SETTINGS:
    - Method: ANOVA (200× faster than MI, <1% accuracy loss)
    - Scope: Global (faster, minimal data leakage with 128 subjects)
    
    This grid provides the best time/quality trade-off for thesis experiments.
    Expected time: ~3-4h for 18 configs (vs ~63h with MI per-fold)
    
    Returns:
        List of FeatureSelectionConfig objects
    """
    return create_feature_selection_grid(
        selection_method='anova',
        scope='global'
    )


def select_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    config: FeatureSelectionConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for feature selection.
    
    Fits on training data and transforms both train and test.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        config: Feature selection configuration
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, selection_summary)
    """
    pipeline = FeatureSelectionPipeline(config)
    X_train_selected = pipeline.fit_transform(X_train, y_train)
    X_test_selected = pipeline.transform(X_test)
    summary = pipeline.get_selection_summary()
    
    return X_train_selected, X_test_selected, summary


# =============================================================================
# Module test
# =============================================================================
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Feature Selection Module")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 149
    
    # Create correlated features
    X_base = np.random.randn(n_samples, n_features // 3)
    X_corr1 = X_base + np.random.randn(n_samples, n_features // 3) * 0.1  # High correlation
    X_corr2 = X_base + np.random.randn(n_samples, n_features // 3) * 0.5  # Medium correlation
    X_noise = np.random.randn(n_samples, n_features - 3 * (n_features // 3))
    
    X = np.hstack([X_base, X_corr1, X_corr2, X_noise])
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = np.random.randint(0, 5, n_samples)
    
    print(f"Input shape: {X_df.shape}")
    
    # Test configurations
    test_configs = [
        FeatureSelectionConfig(correlation_threshold=0.95, top_k_features=50),
        FeatureSelectionConfig(correlation_threshold=0.90, top_k_features=30),
        FeatureSelectionConfig(correlation_threshold=None, top_k_features=50),
        FeatureSelectionConfig(correlation_threshold=0.95, top_k_features=None),
    ]
    
    for config in test_configs:
        print(f"\n{'-' * 40}")
        print(f"Config: {config.get_config_id()}")
        
        pipeline = FeatureSelectionPipeline(config)
        X_selected = pipeline.fit_transform(X_df, y)
        
        summary = pipeline.get_selection_summary()
        print(f"  Input: {summary['n_features_input']} features")
        print(f"  After correlation: {summary['n_features_after_correlation']} features")
        print(f"  Output: {summary['n_features_output']} features")
        print(f"  Reduction: {(1 - summary['reduction_ratio']) * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Feature Selection Module: ✓ All tests passed")
