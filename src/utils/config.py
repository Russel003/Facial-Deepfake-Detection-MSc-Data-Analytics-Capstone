"""
src/utils/config.py

Single source of truth for all paths, constants, and hyperparameters.
No absolute paths. Every path resolves from PROJECT_ROOT or an env var.

On the VM:
    export DEEPFAKE_DATA_ROOT=/home/hduser/deepfake-data
    export DEEPFAKE_SHAPE_PREDICTOR=/home/hduser/deepfake-data/shape_predictor_68_face_landmarks.dat

On any other machine (CSV-only mode, no raw data needed):
    leave both env vars unset — all modelling phases run from the committed CSV.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root: two levels up from this file (src/utils/config.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# External data root (raw videos, parquet, shape predictor).
# Falls back to PROJECT_ROOT/data so CSV-only machines work without any setup.
_env_data_root = os.environ.get("DEEPFAKE_DATA_ROOT")
DATA_ROOT: Path = Path(_env_data_root) if _env_data_root else PROJECT_ROOT / "data"

# ── Feature tables ────────────────────────────────────────────────────────────
# CSV is committed to git and works on any machine.
FEATURE_TABLE_CSV: Path = (
    PROJECT_ROOT / "data" / "features" / "feature_table_augmented.csv"
)
# Parquet requires DATA_ROOT to point at the VM external data directory.
FEATURE_TABLE_PARQUET: Path = (
    DATA_ROOT / "processed" / "parquet" / "feature_table_augmented.parquet"
)
DATASET_MANIFEST: Path = (
    PROJECT_ROOT / "data" / "features" / "dataset_manifest.json"
)

# ── dlib shape predictor ──────────────────────────────────────────────────────
# 96MB binary asset — not committed to git.
# VM default: sits one level above DATA_ROOT at /home/hduser/deepfake-data/
# Override via env var on any machine.
_env_sp = os.environ.get("DEEPFAKE_SHAPE_PREDICTOR")
SHAPE_PREDICTOR_PATH: Path = (
    Path(_env_sp)
    if _env_sp
    else DATA_ROOT / "shape_predictor_68_face_landmarks.dat"
)

# ── Metadata ──────────────────────────────────────────────────────────────────
METADATA_CSV: Path = DATA_ROOT / "metadata" / "metadata_ffpp.csv"

# ── Output directories (all relative to project root) ────────────────────────
REPORTS_DIR: Path   = PROJECT_ROOT / "reports"
FIGURES_DIR: Path   = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR: Path    = PROJECT_ROOT / "reports" / "tables"
LOGS_DIR: Path      = PROJECT_ROOT / "logs"
EXP_LOGS_DIR: Path  = PROJECT_ROOT / "logs" / "experiments"
SPLITS_DIR: Path    = PROJECT_ROOT / "data" / "splits"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ── Column roles ──────────────────────────────────────────────────────────────
LABEL_COL: str    = "class_label"
ID_COL: str       = "video_id"
GROUP_COL: str    = "manipulation_type"
POSITIVE_CLASS: str = "fake"

# ── Feature sets ──────────────────────────────────────────────────────────────
# All 27 numeric predictor columns extracted by Phase 2 ETL.
ALL_FEATURES: list[str] = [
    "total_frames", "frames_sampled", "frames_with_face", "face_detection_rate",
    "brightness_mean", "brightness_std", "brightness_min", "brightness_max",
    "optical_flow_mean", "optical_flow_std", "optical_flow_min", "optical_flow_max",
    "ear_mean", "ear_std", "ear_min", "ear_max",
    "landmark_disp_mean", "landmark_disp_std", "landmark_disp_min", "landmark_disp_max",
    "dct_low_mean", "dct_low_std", "dct_high_mean", "dct_high_std",
    "lm_ratio_var", "lm_face_asym", "lbp_chi_dist",
]

# 3 features that survived Phase 3 statistical validation.
# Canonical input to all downstream modelling (Model A in ablation).
VALIDATED_FEATURES: list[str] = ["ear_std", "ear_min", "lm_ratio_var"]

# SHAP top-8 features (Model B in ablation — from phase5_shap_log.json).
SHAP_TOP8_FEATURES: list[str] = [
    "lm_ratio_var", "ear_std", "ear_min", "ear_max",
    "landmark_disp_mean", "optical_flow_std", "dct_high_std", "lm_face_asym",
]

# ── ETL pipeline constants ────────────────────────────────────────────────────
FRAME_STEP: int      = 5
MAX_FRAMES: int      = 100
BATCH_SIZE: int      = 50

# ── Phase 4 best-model hyperparameters ───────────────────────────────────────
# Kept here so all phases (ablation, holdout, robustness) use identical settings.
RF_PARAMS: dict = {
    "n_estimators":      500,
    "max_depth":         3,
    "max_features":      "sqrt",
    "min_samples_leaf":  10,
    "class_weight":      "balanced",
    "random_state":      RANDOM_SEED,
}

XGB_PARAMS: dict = {
    "n_estimators":       100,
    "max_depth":          3,
    "learning_rate":      0.01,
    "subsample":          1.0,
    "colsample_bytree":   1.0,
    "use_label_encoder":  False,
    "eval_metric":        "logloss",
    "scale_pos_weight":   4,     # 800 fake / 200 real
    "random_state":       RANDOM_SEED,
}

# ── Evaluation constants ──────────────────────────────────────────────────────
BOOTSTRAP_N_ITERS: int = 1000
BOOTSTRAP_CI: int      = 95   # percent
CV_FOLDS: int          = 5
TEST_SIZE: float       = 0.20

# ── Phase 6b robustness test constants ───────────────────────────────────────
NOISE_FRACTIONS: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
NOISE_TRIALS: int             = 20


def validate_paths(raise_on_missing: bool = False) -> dict[str, bool]:
    """
    Check which key paths exist. Call at the top of any notebook or script.

    Returns a dict of {label: exists}.
    If raise_on_missing=True, raises FileNotFoundError for critical missing paths.
    """
    checks: dict[str, bool] = {
        "FEATURE_TABLE_CSV":     FEATURE_TABLE_CSV.exists(),
        "DATASET_MANIFEST":      DATASET_MANIFEST.exists(),
        "LOGS_DIR":              LOGS_DIR.exists(),
        "EXP_LOGS_DIR":          EXP_LOGS_DIR.exists(),
        "PHASE4_LOG":            (LOGS_DIR / "phase4_models_log.json").exists(),
        "SHAPE_PREDICTOR":       SHAPE_PREDICTOR_PATH.exists(),
        "FEATURE_TABLE_PARQUET": FEATURE_TABLE_PARQUET.exists(),
        "METADATA_CSV":          METADATA_CSV.exists(),
    }

    for label, ok in checks.items():
        status = "OK     " if ok else "MISSING"
        print(f"  {status}  {label}")

    if raise_on_missing:
        critical = ["FEATURE_TABLE_CSV", "LOGS_DIR", "PHASE4_LOG"]
        for label in critical:
            if not checks[label]:
                raise FileNotFoundError(
                    f"Critical path missing: {label}. "
                    "Clone the repo fully and ensure logs are present."
                )

    return checks
