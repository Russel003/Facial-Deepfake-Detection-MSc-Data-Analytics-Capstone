import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, DMatrix
import warnings
warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
PARQUET_IN  = "/home/hduser/deepfake-data/processed/parquet/feature_table_augmented.parquet"
FEATURES_IN = "/home/hduser/deepfake-thesis/logs/phase3_surviving_features.json"
PHASE4_LOG  = "/home/hduser/deepfake-thesis/logs/phase4_models_log.json"
REPORT_DIR  = Path("/home/hduser/deepfake-thesis/reports")
LOG_PATH    = "/home/hduser/deepfake-thesis/logs/phase5_shap_log.json"
RANDOM_SEED = 42

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with open(FEATURES_IN) as fh:
    feat_cfg = json.load(fh)
FEATURES = feat_cfg["final_features"]

with open(PHASE4_LOG) as fh:
    phase4 = json.load(fh)
train_idx = np.array(phase4["train_idx"])
test_idx  = np.array(phase4["test_idx"])

df = pd.read_parquet(PARQUET_IN)
X  = df[FEATURES].values
y  = (df["class_label"] == "fake").astype(int).values

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
X_test_df = pd.DataFrame(X_test, columns=FEATURES)

scale_pos = float((y_train == 0).sum()) / float(y_train.sum())

print(f"Features   : {FEATURES}")
print(f"Train size : {len(X_train)} | Test size: {len(X_test)}")

# ── RETRAIN MODELS ────────────────────────────────────────────────────────────
print("\nRetraining RF and XGB with Phase 4 best params...")

rf = RandomForestClassifier(
    n_estimators=phase4["rf_best_params"]["n_estimators"],
    max_depth=phase4["rf_best_params"]["max_depth"],
    min_samples_leaf=phase4["rf_best_params"]["min_samples_leaf"],
    max_features=phase4["rf_best_params"]["max_features"],
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf.fit(X_train, y_train)

xgb = XGBClassifier(
    n_estimators=phase4["xgb_best_params"]["n_estimators"],
    max_depth=phase4["xgb_best_params"]["max_depth"],
    learning_rate=phase4["xgb_best_params"]["learning_rate"],
    subsample=phase4["xgb_best_params"]["subsample"],
    colsample_bytree=phase4["xgb_best_params"]["colsample_bytree"],
    scale_pos_weight=scale_pos,
    random_state=RANDOM_SEED,
    eval_metric="auc",
    verbosity=0,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
print("Models retrained.")

# ── SHAP: RANDOM FOREST ───────────────────────────────────────────────────────
print("\nComputing SHAP values for Random Forest...")
rf_explainer   = shap.TreeExplainer(rf)
rf_shap_values = rf_explainer.shap_values(X_test_df)
rf_sv = rf_shap_values[1] if isinstance(rf_shap_values, list) else rf_shap_values

# Handle 3D output from some SHAP versions: (n_samples, n_features, n_classes)
if rf_sv.ndim == 3:
    rf_sv = rf_sv[:, :, 1]

print(f"  rf_sv shape: {rf_sv.shape}")

# ── FIX: define top_feat HERE after rf_sv is computed ─────────────────────────
top_feat_idx = int(np.argmax([abs(rf_sv[:, i]).mean() for i in range(len(FEATURES))]))
top_feat     = FEATURES[top_feat_idx]
print(f"  Top SHAP feature: {top_feat}")

# Plot 1: RF beeswarm
plt.figure()
shap.summary_plot(rf_sv, X_test_df, feature_names=FEATURES,
                  show=False, plot_type="dot")
plt.title("RF SHAP Summary — Feature Impact on Fake Prediction", fontsize=12)
plt.tight_layout()
out = REPORT_DIR / "shap_rf_summary.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# Plot 2: RF bar
plt.figure()
shap.summary_plot(rf_sv, X_test_df, feature_names=FEATURES,
                  show=False, plot_type="bar")
plt.title("RF SHAP Mean Absolute Feature Importance", fontsize=12)
plt.tight_layout()
out = REPORT_DIR / "shap_rf_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# Plot 3: RF dependence on top feature
feat_vals = X_test_df[top_feat].values
shap_vals = rf_sv[:, top_feat_idx]
n         = min(len(feat_vals), len(shap_vals))

plt.figure(figsize=(7, 5))
plt.scatter(feat_vals[:n], shap_vals[:n], alpha=0.6,
            edgecolors="none", color="#01696f", s=30)
plt.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
plt.xlabel(top_feat, fontsize=11)
plt.ylabel("SHAP value (impact on fake prediction)", fontsize=11)
plt.title(f"RF SHAP Dependence: {top_feat}", fontsize=12)
plt.tight_layout()
out = REPORT_DIR / f"shap_rf_dependence_{top_feat}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── SHAP: XGBOOST (native pred_contribs) ─────────────────────────────────────
print("\nComputing SHAP values for XGBoost (native pred_contribs)...")
dtest           = DMatrix(X_test_df.values, feature_names=FEATURES)
contribs        = xgb.get_booster().predict(dtest, pred_contribs=True)
xgb_shap_values = contribs[:, :-1]   # drop bias column
print(f"  XGB SHAP matrix shape: {xgb_shap_values.shape}")

# Plot 4: XGB beeswarm
plt.figure()
shap.summary_plot(xgb_shap_values, X_test_df, feature_names=FEATURES,
                  show=False, plot_type="dot")
plt.title("XGB SHAP Summary — Feature Impact on Fake Prediction", fontsize=12)
plt.tight_layout()
out = REPORT_DIR / "shap_xgb_summary.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# Plot 5: XGB bar
plt.figure()
shap.summary_plot(xgb_shap_values, X_test_df, feature_names=FEATURES,
                  show=False, plot_type="bar")
plt.title("XGB SHAP Mean Absolute Feature Importance", fontsize=12)
plt.tight_layout()
out = REPORT_DIR / "shap_xgb_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── SHAP MEAN VALUES TABLE ────────────────────────────────────────────────────
print("\n  SHAP mean |values| (RF, fake class):")
rf_mean_shap = {f: round(float(abs(rf_sv[:, i]).mean()), 5)
                for i, f in enumerate(FEATURES)}
for k, v in sorted(rf_mean_shap.items(), key=lambda x: -x[1]):
    print(f"    {k:<30} {v:.5f}")

print("\n  SHAP mean |values| (XGB, native):")
xgb_mean_shap = {f: round(float(abs(xgb_shap_values[:, i]).mean()), 5)
                 for i, f in enumerate(FEATURES)}
for k, v in sorted(xgb_mean_shap.items(), key=lambda x: -x[1]):
    print(f"    {k:<30} {v:.5f}")

# ── LOG ───────────────────────────────────────────────────────────────────────
plots_saved = sorted([str(p) for p in REPORT_DIR.glob("shap_*.png")])
log = {
    "run_timestamp":   datetime.datetime.now().isoformat(),
    "features":        FEATURES,
    "top_shap_feature": top_feat,
    "rf_mean_shap":    rf_mean_shap,
    "xgb_mean_shap":   xgb_mean_shap,
    "xgb_shap_method": "native_pred_contribs",
    "plots_saved":     plots_saved,
}
with open(LOG_PATH, "w") as fh:
    json.dump(log, fh, indent=2)

print(f"\n  Log saved : {LOG_PATH}")
print(f"  Plots     : {plots_saved}")
print("\nPhase 5 SHAP COMPLETE.")

# What it does:
# RF SHAP via shap.TreeExplainer (clean with sklearn RF).
# XGB SHAP via native booster.predict(pred_contribs=True), bypassing the
# '[5E-1]' base_score string bug in SHAP < 0.42.
# top_feat is derived from rf_sv AFTER it is computed, fixing the NameError.
# Produces 5 PNG plots in /reports/ for thesis inclusion.
