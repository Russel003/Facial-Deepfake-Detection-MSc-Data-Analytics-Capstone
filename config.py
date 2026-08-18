# config.py — Portable path configuration
# Convenience alias. Canonical configuration is at src/utils/config.py

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR          = ROOT / "data" / "features"
FEATURES_CSV      = DATA_DIR / "feature_table_augmented.csv"
FEATURES_PARQUET  = DATA_DIR / "feature_table_augmented.parquet"
LOGS_DIR          = ROOT / "logs"
REPORTS_DIR       = ROOT / "reports" / "phase3"
FIGURES_DIR       = ROOT / "reports" / "figures"
SRC_DIR           = ROOT / "src"
NOTEBOOKS_DIR     = ROOT / "notebooks"

PHASE3_FEATURES_JSON = LOGS_DIR / "phase3_surviving_features.json"
PHASE3_STATS_CSV     = LOGS_DIR / "phase3_feature_stats_table.csv"
PHASE4_MODELS_LOG    = LOGS_DIR / "phase4_models_log.json"
PHASE4_BEST_MODEL    = LOGS_DIR / "phase4_best_model.json"
PHASE5_SHAP_LOG      = LOGS_DIR / "phase5_shap_log.json"
PHASE6A_LOG          = LOGS_DIR / "phase6a_holdout_log.json"
PHASE6B_LOG          = LOGS_DIR / "phase6b_compression_log.json"
PHASE6C_LOG          = LOGS_DIR / "phase6c_roc_log.json"

RANDOM_SEED        = 42
TEST_SIZE          = 0.20
N_VIDEOS_TOTAL     = 1000
N_REAL             = 200
N_FAKE             = 800
N_TEST             = 200
VALIDATED_FEATURES = ["ear_std", "ear_min", "lm_ratio_var"]

GROUND_TRUTH = {
    'rf': {
        'auc'       : 0.6586,
        'f1'        : 0.7800,
        'precision' : 0.8357,
        'recall'    : 0.7312,
        'cm'        : [[17, 23], [43, 117]],
        'note'      : 'Canonical best model. AUC from phase4_models_log.json.',
    },
    'xgb': {
        'auc'       : 0.6280,
        'f1'        : 0.7292,
        'precision' : 0.8203,
        'recall'    : 0.6562,
        'cm'        : [[3, 37], [5, 155]],
        'note'      : 'Degenerate at threshold=0.5. Threshold=0.8 used.',
    },
    'ensemble': {
        'auc'       : 0.6427,
        'f1'        : 0.7609,
        'precision' : 0.8248,
        'recall'    : 0.7063,
        'cm'        : None,
        'note'      : 'Soft-vote ensemble. RF remains canonical single model.',
    },
}

HOLDOUT_MEAN_AUC  = 0.694
COMPRESSION_FLOOR = 0.5719

LABEL_REAL   = "real"
LABEL_FAKE   = "fake"
LABEL_ENCODE = {"real": 0, "fake": 1}

def load_xy(features_csv=None):
    import pandas as pd
    path = features_csv or FEATURES_CSV
    df = pd.read_csv(path)
    X = df[VALIDATED_FEATURES]
    y = (df['class_label'] == LABEL_FAKE).astype(int)
    return df, X, y