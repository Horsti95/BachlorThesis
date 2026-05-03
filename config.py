"""
Configuration Management for ML Caching Pipeline
================================================

Handles all configuration parameters for the experiment pipeline.
Uses YAML for human-readable configuration files.

Author: Lennart Gorzel
Date: December 2025
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration parameters."""
    
    # Filtering
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    notch_frequency: float = 50.0
    
    # Resampling
    original_sfreq: float = 256.0
    target_sfreq: float = 128.0
    
    # Epoching
    epoch_duration: float = 30.0
    
    # Quality control
    min_epochs_per_subject: int = 100
    filter_disconnections: bool = True
    filter_artifacts: bool = True


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    
    compute_time_domain: bool = True
    compute_frequency_domain: bool = True
    compute_complexity: bool = True
    compute_global_features: bool = True
    
    # Frequency bands
    freq_bands: Dict[str, List[float]] = field(default_factory=lambda: {
        'delta': [0.5, 4.0],
        'theta': [4.0, 8.0],
        'alpha': [8.0, 13.0],
        'sigma': [12.0, 16.0],
        'beta': [16.0, 30.0],
        'gamma': [30.0, 40.0],
    })
    
    # Feature selection
    correlation_threshold: Optional[float] = None
    
    def expected_feature_count(self, n_channels: int = 6) -> int:
        """Calculate expected number of features for the given channel count.

        Returns total = n_channels * 23 (per-channel) + 11 (global).
        """
        # Per channel: 10 time + 9 frequency + 4 complexity = 23
        per_channel = 23
        # Global: 11 features
        global_features = 11
        return (n_channels * per_channel) + global_features


@dataclass
class DataConfig:
    """Data loading configuration."""
    
    base_path: str = r"C:\Users\DerHo\Desktop\Data"
    use_human_labels: bool = True
    subjects: Optional[List[str]] = None
    channel_preset: str = "eeg_only"  # eeg_only, eeg_plus_physiological, custom
    channels: Optional[List[str]] = None
    
    def get_channels(self) -> List[str]:
        """Return channel names based on channel_preset or custom list."""
        if self.channel_preset == "eeg_only":
            return ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2']
        elif self.channel_preset == "eeg_plus_physiological":
            return ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2', 'PSG_EOG', 'PSG_EMG']
        elif self.channels:
            return self.channels
        else:
            return ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2']
    
    def get_expected_features(self) -> int:
        """Calculate expected number of features based on channel preset.
        
        Formula: n_channels * 23 (per-channel) + 11 (global)
        - 6 channels: 6 * 23 + 11 = 149 features
        - 8 channels: 8 * 23 + 11 = 195 features
        """
        n_channels = len(self.get_channels())
        per_channel_features = 23  # 10 time + 9 frequency + 4 complexity
        global_features = 11
        return (n_channels * per_channel_features) + global_features


@dataclass
class ModelConfig:
    """Model configuration (type, hyperparameters, random seed)."""
    
    model_type: str = "xgboost"
    random_seed: int = 42
    params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 6,
        'n_estimators': 200,
        'learning_rate': 0.1,
        'objective': 'multi:softmax',
        'num_class': 5,
    })


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration (LOSO or k-fold)."""
    
    method: str = "loso"  # loso, kfold
    n_folds: Optional[int] = None
    fixed_hyperparams: bool = True


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration aggregating all sub-configs."""
    
    experiment_name: str = "experiment"
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    output_dir: str = "./results"
    log_level: str = "INFO"

    # Training grid (set by interactive menu for multi-config experiments)
    training_models: Optional[List[str]] = None
    training_feature_configs: Optional[List[tuple]] = None
    
    def get_output_dir(self) -> Path:
        """Build a timestamped output directory path (does not create it on disk)."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.output_dir) / f"{self.experiment_name}_{timestamp}"


class ConfigManager:
    """Manages loading and validation of experiment configurations."""
    
    @staticmethod
    def from_yaml(yaml_path: str) -> ExperimentConfig:
        """Load and parse a YAML file into an ExperimentConfig.

        Raises FileNotFoundError if the file does not exist.
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return ConfigManager._dict_to_config(data)
    
    @staticmethod
    def _dict_to_config(data: Dict) -> ExperimentConfig:
        """Convert a nested dictionary (from YAML) into an ExperimentConfig.

        Missing keys fall back to dataclass defaults.
        """
        config = ExperimentConfig()
        
        if 'experiment_name' in data:
            config.experiment_name = data['experiment_name']
        
        if 'data' in data:
            d = data['data']
            config.data = DataConfig(
                base_path=d.get('base_path', config.data.base_path),
                use_human_labels=d.get('use_human_labels', True),
                subjects=d.get('subjects'),
                channel_preset=d.get('channel_preset', 'eeg_only'),
                channels=d.get('channels'),
            )
        
        if 'preprocessing' in data:
            p = data['preprocessing']
            config.preprocessing = PreprocessingConfig(
                bandpass_low=p.get('bandpass_low', 0.5),
                bandpass_high=p.get('bandpass_high', 40.0),
                notch_frequency=p.get('notch_frequency', 50.0),
                original_sfreq=p.get('original_sfreq', 256.0),
                target_sfreq=p.get('target_sfreq', 128.0),
                epoch_duration=p.get('epoch_duration', 30.0),
                min_epochs_per_subject=p.get('min_epochs_per_subject', 100),
                filter_disconnections=p.get('filter_disconnections', True),
                filter_artifacts=p.get('filter_artifacts', True),
            )
        
        if 'features' in data:
            f = data['features']
            config.features = FeatureConfig(
                compute_time_domain=f.get('compute_time_domain', True),
                compute_frequency_domain=f.get('compute_frequency_domain', True),
                compute_complexity=f.get('compute_complexity', True),
                compute_global_features=f.get('compute_global_features', True),
                freq_bands=f.get('freq_bands', config.features.freq_bands),
                correlation_threshold=f.get('correlation_threshold'),
            )
        
        if 'model' in data:
            m = data['model']
            config.model = ModelConfig(
                model_type=m.get('model_type', 'xgboost'),
                random_seed=m.get('random_seed', 42),
                params=m.get('params', {}),
            )
        
        if 'cross_validation' in data:
            cv = data['cross_validation']
            config.cross_validation = CrossValidationConfig(
                method=cv.get('method', 'loso'),
                n_folds=cv.get('n_folds'),
                fixed_hyperparams=cv.get('fixed_hyperparams', True),
            )
        
        if 'output_dir' in data:
            config.output_dir = data['output_dir']
        
        if 'log_level' in data:
            config.log_level = data['log_level']
        
        return config
    
    @staticmethod
    def to_yaml(config: ExperimentConfig, yaml_path: str):
        """Serialize an ExperimentConfig to a YAML file, creating parent dirs if needed."""
        data = ConfigManager._config_to_dict(config)
        
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def _config_to_dict(config: ExperimentConfig) -> Dict:
        """Flatten an ExperimentConfig into a nested dict suitable for YAML serialization."""
        return {
            'experiment_name': config.experiment_name,
            'data': {
                'base_path': config.data.base_path,
                'use_human_labels': config.data.use_human_labels,
                'subjects': config.data.subjects,
                'channel_preset': config.data.channel_preset,
                'channels': config.data.channels,
            },
            'preprocessing': {
                'bandpass_low': config.preprocessing.bandpass_low,
                'bandpass_high': config.preprocessing.bandpass_high,
                'notch_frequency': config.preprocessing.notch_frequency,
                'original_sfreq': config.preprocessing.original_sfreq,
                'target_sfreq': config.preprocessing.target_sfreq,
                'epoch_duration': config.preprocessing.epoch_duration,
                'min_epochs_per_subject': config.preprocessing.min_epochs_per_subject,
                'filter_disconnections': config.preprocessing.filter_disconnections,
                'filter_artifacts': config.preprocessing.filter_artifacts,
            },
            'features': {
                'compute_time_domain': config.features.compute_time_domain,
                'compute_frequency_domain': config.features.compute_frequency_domain,
                'compute_complexity': config.features.compute_complexity,
                'compute_global_features': config.features.compute_global_features,
                'freq_bands': config.features.freq_bands,
                'correlation_threshold': config.features.correlation_threshold,
            },
            'model': {
                'model_type': config.model.model_type,
                'random_seed': config.model.random_seed,
                'params': config.model.params,
            },
            'cross_validation': {
                'method': config.cross_validation.method,
                'n_folds': config.cross_validation.n_folds,
                'fixed_hyperparams': config.cross_validation.fixed_hyperparams,
            },
            'output_dir': config.output_dir,
            'log_level': config.log_level,
        }
    
    @staticmethod
    def create_default_config(
        experiment_name: str = "experiment",
        data_path: str = r"C:\Users\DerHo\Desktop\Data",
        model_type: str = "xgboost"
    ) -> ExperimentConfig:
        """Create an ExperimentConfig with default values, overriding name/path/model."""
        config = ExperimentConfig()
        config.experiment_name = experiment_name
        config.data.base_path = data_path
        config.model.model_type = model_type
        return config
    
    @staticmethod
    def validate_config(config: ExperimentConfig) -> List[str]:
        """Validate configuration and return a list of issue descriptions.

        Returns an empty list if everything is valid.
        """
        issues = []
        
        # Check data path
        if not Path(config.data.base_path).exists():
            issues.append(f"Data path does not exist: {config.data.base_path}")
        
        # Check preprocessing params
        if config.preprocessing.bandpass_low >= config.preprocessing.bandpass_high:
            issues.append("bandpass_low must be less than bandpass_high")
        
        if config.preprocessing.target_sfreq > config.preprocessing.original_sfreq:
            issues.append("target_sfreq cannot be higher than original_sfreq")
        
        # Check model type
        valid_models = ['xgboost', 'random_forest', 'fnn']
        if config.model.model_type not in valid_models:
            issues.append(f"Invalid model type: {config.model.model_type}. Must be one of {valid_models}")
        
        return issues


# Example usage
if __name__ == "__main__":
    # Create default config
    config = ConfigManager.create_default_config(
        experiment_name="pilot_xgboost",
        data_path=r"C:\Users\DerHo\Desktop\Data",
        model_type="xgboost"
    )
    
    print("Default Configuration:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Data path: {config.data.base_path}")
    print(f"  Preprocessing: {config.preprocessing.bandpass_low}-{config.preprocessing.bandpass_high} Hz")
    print(f"  Target sampling rate: {config.preprocessing.target_sfreq} Hz")
    print(f"  Expected features: {config.features.expected_feature_count()}")
    print(f"  Model: {config.model.model_type}")
    print(f"  Cross-validation: {config.cross_validation.method}")
    
    # Save to YAML
    ConfigManager.to_yaml(config, "example_config.yaml")
    print("\nConfiguration saved to example_config.yaml")
