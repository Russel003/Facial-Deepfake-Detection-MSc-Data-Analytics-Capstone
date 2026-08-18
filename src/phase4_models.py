import json
import datetime
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
PARQUET_IN   = "/home/hduser/deepfake-data/processed/parquet/feature_table_augmented.parquet"
FEATURES_IN  = "/home/hduser/deepfake-thesis/logs/phase3_surviving_features.json"
LOG_PATH     = "/home/hduser/deepfake-thesis/logs/phase4_models_log.json"
REPORT_PATH  = "/home/hduser/deepfake-thesis/logs/phase4_best_model.json"

# ── CONFIG ────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_FOLDS     = 5
TEST_SIZE   = 0.20   # held-out test split

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "-C", "/home/hduser/deepfake-thesis", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def make_serialisable(obj):
    if isinstance(obj, dict):
        return {k: make_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serialisable(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def eval_on_test(model, X_test, y_test, label):
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    rec     = recall_score(y_test, y_pred)
    cm      = confusion_matrix(y_test, y_pred).tolist()
    print(f"\n  [{label}]")
    print(f"    AUC       : {auc:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    Confusion matrix (TN FP / FN TP):")
    print(f"      {cm[0]}")
    print(f"      {cm[1]}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['real','fake'])}")
    return {"auc": round(auc, 4), "f1": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "confusion_matrix": cm}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print_section("PHASE 4: MODEL TRAINING")

with open(FEATURES_IN) as fh:
    feat_cfg = json.load(fh)
FEATURES = feat_cfg["final_features"]

df = pd.read_parquet(PARQUET_IN)
print(f"\n  Dataset     : {len(df)} rows")
print(f"  Features    : {FEATURES}")
print(f"  Class dist  : {df['class_label'].value_counts().to_dict()}")

X = df[FEATURES].values
y = (df["class_label"] == "fake").astype(int).values   # 1=fake, 0=real

# ── STRATIFIED TRAIN/TEST SPLIT (manual, reproducible) ────────────────────────
rng         = np.random.default_rng(RANDOM_SEED)
indices     = np.arange(len(X))
fake_idx    = indices[y == 1]
real_idx    = indices[y == 0]

# Stratified 80/20 split
n_fake_test = int(len(fake_idx) * TEST_SIZE)
n_real_test = int(len(real_idx) * TEST_SIZE)

rng.shuffle(fake_idx)
rng.shuffle(real_idx)

test_idx    = np.concatenate([fake_idx[:n_fake_test], real_idx[:n_real_test]])
train_idx   = np.concatenate([fake_idx[n_fake_test:], real_idx[n_real_test:]])

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"\n  Train set   : {len(X_train)} samples  "
      f"(fake={y_train.sum()}, real={(y_train==0).sum()})")
print(f"  Test set    : {len(X_test)} samples   "
      f"(fake={y_test.sum()}, real={(y_test==0).sum()})")
print(f"  Split indices saved for reproducibility")

# ── CLASS IMBALANCE NOTE ──────────────────────────────────────────────────────
# 1:4 real:fake ratio. Using class_weight='balanced' in RF and scale_pos_weight
# in XGBoost. This is more principled than oversampling on a small feature set.
scale_pos = int((y_train == 0).sum()) / int(y_train.sum())
print(f"\n  Class imbalance ratio (real:fake) = 1:{1/scale_pos:.1f}")
print(f"  RF uses class_weight='balanced' | XGB scale_pos_weight={scale_pos:.2f}")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# ── MODEL 1: RANDOM FOREST ────────────────────────────────────────────────────
print_section("MODEL 1: Random Forest — GridSearchCV")

rf_grid = {
    "n_estimators":   [100, 300, 500],
    "max_depth":      [3, 5, 10, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features":   ["sqrt", "log2"],
}

rf_base = RandomForestClassifier(
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1
)

rf_search = GridSearchCV(
    rf_base, rf_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_

print(f"\n  Best RF params : {rf_search.best_params_}")
print(f"  Best CV AUC    : {rf_search.best_score_:.4f}")

rf_test_metrics = eval_on_test(rf_best, X_test, y_test, "Random Forest — Test Set")

# Feature importances
rf_importances = dict(zip(FEATURES, rf_best.feature_importances_.round(4).tolist()))
print(f"\n  RF Feature importances: {rf_importances}")

# ── MODEL 2: XGBOOST ──────────────────────────────────────────────────────────
print_section("MODEL 2: XGBoost — GridSearchCV")

xgb_grid = {
    "n_estimators":   [100, 300, 500],
    "max_depth":      [3, 5, 7],
    "learning_rate":  [0.01, 0.05, 0.1],
    "subsample":      [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
}

xgb_base = XGBClassifier(
    scale_pos_weight=scale_pos,
    random_state=RANDOM_SEED,
    use_label_encoder=False,
    eval_metric="auc",
    n_jobs=-1,
    verbosity=0
)

xgb_search = GridSearchCV(
    xgb_base, xgb_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_

print(f"\n  Best XGB params : {xgb_search.best_params_}")
print(f"  Best CV AUC     : {xgb_search.best_score_:.4f}")

xgb_test_metrics = eval_on_test(xgb_best, X_test, y_test, "XGBoost — Test Set")

xgb_importances = dict(zip(FEATURES,
    [round(float(v), 4) for v in xgb_best.feature_importances_]))
print(f"\n  XGB Feature importances: {xgb_importances}")

# ── MODEL 3: SOFT VOTING ENSEMBLE ─────────────────────────────────────────────
print_section("MODEL 3: Soft Voting Ensemble (RF + XGBoost)")

ensemble = VotingClassifier(
    estimators=[("rf", rf_best), ("xgb", xgb_best)],
    voting="soft",
    n_jobs=-1
)
ensemble.fit(X_train, y_train)

ens_test_metrics = eval_on_test(ensemble, X_test, y_test, "Ensemble — Test Set")

# ── CROSS-VALIDATION ON FULL TRAIN SET ────────────────────────────────────────
print_section("5-FOLD CROSS-VALIDATION SUMMARY (train set)")

for name, model in [("RF", rf_best), ("XGB", xgb_best), ("Ensemble", ensemble)]:
    cv_res = cross_validate(model, X_train, y_train,
                            cv=cv, scoring=["roc_auc", "f1"],
                            n_jobs=-1)
    print(f"  {name:<12} AUC={cv_res['test_roc_auc'].mean():.4f} "
          f"(+/-{cv_res['test_roc_auc'].std():.4f})  "
          f"F1={cv_res['test_f1'].mean():.4f} "
          f"(+/-{cv_res['test_f1'].std():.4f})")

# ── PICK BEST MODEL ───────────────────────────────────────────────────────────
print_section("BEST MODEL SELECTION (by test AUC)")

scores = {
    "RandomForest": rf_test_metrics["auc"],
    "XGBoost":      xgb_test_metrics["auc"],
    "Ensemble":     ens_test_metrics["auc"],
}
best_name = max(scores, key=scores.get)
print(f"\n  {scores}")
print(f"  Best model: {best_name} (AUC={scores[best_name]:.4f})")

# ── WRITE LOGS ────────────────────────────────────────────────────────────────
log = {
    "run_timestamp":      datetime.datetime.now().isoformat(),
    "git_commit":         get_git_hash(),
    "random_seed":        RANDOM_SEED,
    "n_folds":            N_FOLDS,
    "features_used":      FEATURES,
    "train_size":         len(X_train),
    "test_size":          len(X_test),
    "train_idx":          train_idx.tolist(),
    "test_idx":           test_idx.tolist(),
    "class_imbalance_ratio": round(scale_pos, 4),
    "rf_best_params":     rf_search.best_params_,
    "rf_cv_auc":          round(rf_search.best_score_, 4),
    "rf_test_metrics":    rf_test_metrics,
    "rf_feature_importances": rf_importances,
    "xgb_best_params":    xgb_search.best_params_,
    "xgb_cv_auc":         round(xgb_search.best_score_, 4),
    "xgb_test_metrics":   xgb_test_metrics,
    "xgb_feature_importances": xgb_importances,
    "ensemble_test_metrics": ens_test_metrics,
    "best_model":         best_name,
    "best_auc":           scores[best_name],
}

Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(LOG_PATH, "w") as fh:
    json.dump(make_serialisable(log), fh, indent=2)

with open(REPORT_PATH, "w") as fh:
    json.dump(make_serialisable({
        "best_model":   best_name,
        "best_auc":     scores[best_name],
        "all_scores":   scores,
        "features":     FEATURES,
        "git_commit":   get_git_hash(),
    }), fh, indent=2)

print(f"\n  Log saved    : {LOG_PATH}")
print(f"  Best model   : {REPORT_PATH}")
print("\nPhase 4 COMPLETE.")

# What it does:
# Trains Random Forest, XGBoost, and a soft-voting ensemble on the 3 surviving
# features from Phase 3 (ear_std, ear_min, lm_ratio_var).
# Uses stratified 80/20 train/test split and 5-fold stratified CV for tuning.
# Handles 1:4 class imbalance via class_weight='balanced' (RF) and
# scale_pos_weight (XGBoost). GridSearchCV optimises for ROC-AUC.
# Logs all hyperparameters, split indices, metrics, and git commit hash to JSON.
# Reports AUC, F1, precision, recall, confusion matrix on held-out test set.
#
# Key risks:
# Only 3 features survived Phase 3. AUC may be modest (~0.60-0.70) due to c40
# compression attenuation. Document this as an empirical finding, not a failure.
# GridSearchCV with full grid may take 15-30 min on CPU. If too slow, reduce
# n_estimators to [100, 300] and max_depth to [3, 5] only.
