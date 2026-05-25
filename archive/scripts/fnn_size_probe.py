from pathlib import Path
from run_training import load_cached_features, GLOBAL_CACHE_DIR
from training import TrainingPipeline, TrainingConfig
from feature_selection import FeatureSelectionConfig

try:
import torch
except Exception as e:
raise SystemExit("PyTorch not available: " + str(e))

subjects = [str(i) for i in range(1, 129)]
features_df, labels, subject_ids = load_cached_features(subjects, GLOBAL_CACHE_DIR, n_channels=6)

cfg = TrainingConfig(
model_type="fnn",
model_params={
"hidden_dims": [256, 128, 64],
"dropout": 0.3,
"learning_rate": 0.001,
"batch_size": 256,
"epochs": 3,
"early_stopping_patience": 2
},
feature_selection=FeatureSelectionConfig(
correlation_threshold=None,
top_k_features=None,
selection_method="anova",
use_hybrid=True,
scope="global"
),
random_state=42
)

probe_dir = Path("results/fnn_size_probe")
cache_dir = probe_dir / "loso_model_cache"

pipeline = TrainingPipeline(
features_df=features_df,
labels=labels,
subject_ids=subject_ids,
output_dir=probe_dir,
experiment_name="fnn_one_fold_probe",
enable_model_cache=True,
model_cache_dir=str(cache_dir),
max_folds=1
)

_ = pipeline.run_single_config(cfg, show_progress=False, config_idx=1, total_configs=1)

pt_files = sorted(cache_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
if not pt_files:
raise SystemExit("No FNN .pt cache file found after run.")

pt = pt_files[0]
scaler = Path(str(pt).replace(".pt", "_scaler.joblib"))

pt_mb = pt.stat().st_size / (1024 * 1024)
scaler_mb = scaler.stat().st_size / (1024 * 1024) if scaler.exists() else 0.0
total_mb = pt_mb + scaler_mb

print("")
print("FNN one-fold cache artifact size")
print("PT file: {} -> {:.2f} MB".format(pt.name, pt_mb))
print("Scaler file: {} -> {:.2f} MB".format(scaler.name if scaler.exists() else "missing", scaler_mb))
print("TOTAL per fold: {:.2f} MB".format(total_mb))
