import json, datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score, confusion_matrix)
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings("ignore")

PARQUET_IN  = "/home/hduser/deepfake-data/processed/parquet/feature_table_augmented.parquet"
FEATURES_IN = "/home/hduser/deepfake-thesis/logs/phase3_surviving_features.json"
PHASE4_LOG  = "/home/hduser/deepfake-thesis/logs/phase4_models_log.json"
REPORT_DIR  = Path("/home/hduser/deepfake-thesis/reports")
LOG_PATH    = "/home/hduser/deepfake-thesis/logs/phase6c_roc_log.json"
RANDOM_SEED = 42
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(FEATURES_IN) as fh: FEATURES = json.load(fh)["final_features"]
with open(PHASE4_LOG)  as fh: phase4   = json.load(fh)

df = pd.read_parquet(PARQUET_IN)
train_idx = np.array(phase4["train_idx"])
test_idx  = np.array(phase4["test_idx"])
rf_p = phase4["rf_best_params"]; xgb_p = phase4["xgb_best_params"]

X = df[FEATURES].values
y = (df["class_label"] == "fake").astype(int).values
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
scale_pos = float((y_train==0).sum()) / float(y_train.sum())

print("Retraining RF, XGB, Ensemble...")
rf = RandomForestClassifier(
    n_estimators=rf_p["n_estimators"], max_depth=rf_p["max_depth"],
    min_samples_leaf=rf_p["min_samples_leaf"], max_features=rf_p["max_features"],
    class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)

xgb = XGBClassifier(
    n_estimators=xgb_p["n_estimators"], max_depth=xgb_p["max_depth"],
    learning_rate=xgb_p["learning_rate"], subsample=xgb_p["subsample"],
    colsample_bytree=xgb_p["colsample_bytree"], scale_pos_weight=scale_pos,
    random_state=RANDOM_SEED, eval_metric="auc", verbosity=0, n_jobs=-1)
xgb.fit(X_train, y_train)

ensemble = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb)],
                             voting="soft", n_jobs=-1)
ensemble.fit(X_train, y_train)
print("Done.")

models = {"Random Forest": rf, "XGBoost": xgb, "Ensemble": ensemble}
colors = {"Random Forest": "#01696f", "XGBoost": "#964219", "Ensemble": "#7a39bb"}
probs  = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
roc_data = {}
for name, prob in probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc_val = auc(fpr, tpr)
    roc_data[name] = {"auc": round(roc_auc_val, 4)}
    ax.plot(fpr, tpr, color=colors[name], linewidth=2.0,
            label=f"{name}  (AUC = {roc_auc_val:.4f})")
ax.plot([0,1],[0,1], color="#aaaaaa", linewidth=0.8, linestyle="--",
        label="Random (AUC=0.50)")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title(f"ROC Curves — All Models (n={len(y_test)}, "
             f"fake={y_test.sum()}, real={(y_test==0).sum()})", fontsize=10)
ax.legend(fontsize=9, loc="lower right"); ax.grid(True, alpha=0.2)
plt.tight_layout()
out1 = REPORT_DIR / "roc_curves_all_models.png"
plt.savefig(out1, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {out1}")

# Precision-Recall
fig, ax = plt.subplots(figsize=(7, 6))
ax.axhline(y_test.mean(), color="#aaaaaa", linewidth=0.8, linestyle="--",
           label=f"Random baseline (P={y_test.mean():.2f})")
for name, prob in probs.items():
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    ax.plot(rec, prec, color=colors[name], linewidth=2.0,
            label=f"{name}  (AP={ap:.4f})")
ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curves — All Models", fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
plt.tight_layout()
out2 = REPORT_DIR / "pr_curves_all_models.png"
plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {out2}")

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, model) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=["real","fake"], yticklabels=["real","fake"],
                ax=ax, cbar=False, annot_kws={"size": 13})
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"{name}", fontsize=11)
    ax.text(0.5, -0.18, f"AUC = {roc_data[name]['auc']:.4f}",
            transform=ax.transAxes, ha="center", fontsize=9, color="#444444")
plt.suptitle("Confusion Matrices (Held-Out Test Set)", fontsize=12, y=1.02)
plt.tight_layout()
out3 = REPORT_DIR / "confusion_matrices_all_models.png"
plt.savefig(out3, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {out3}")

# Feature importance comparison
x = np.arange(len(FEATURES)); w = 0.35
fig, ax = plt.subplots(figsize=(7, 4))
bars1 = ax.bar(x-w/2, rf.feature_importances_,  w, label="Random Forest",
               color="#01696f", alpha=0.85)
bars2 = ax.bar(x+w/2, xgb.feature_importances_, w, label="XGBoost",
               color="#964219", alpha=0.85)
for bar in list(bars1)+list(bars2):
    ax.annotate(f"{bar.get_height():.3f}",
                xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,3), textcoords="offset points",
                ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(FEATURES, fontsize=11)
ax.set_ylabel("Feature Importance (Gini / Gain)", fontsize=10)
ax.set_title("Feature Importance: RF vs XGBoost", fontsize=10)
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.2)
plt.tight_layout()
out4 = REPORT_DIR / "feature_importance_comparison.png"
plt.savefig(out4, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {out4}")

print("\nMetrics table:")
print(f"  {'Model':<18} {'AUC':>7}  {'AP':>7}")
for name, prob in probs.items():
    print(f"  {name:<18} {roc_data[name]['auc']:>7.4f}  "
          f"{round(average_precision_score(y_test, prob), 4):>7.4f}")

log = {"run_timestamp": datetime.datetime.now().isoformat(),
       "features": FEATURES,
       "roc_data": {k: {"auc": v["auc"]} for k, v in roc_data.items()},
       "plots_saved": [str(out1), str(out2), str(out3), str(out4)]}
with open(LOG_PATH, "w") as fh: json.dump(log, fh, indent=2)
print(f"\nLog saved: {LOG_PATH}")
print("\nPhase 6c ROC Curves COMPLETE.")
