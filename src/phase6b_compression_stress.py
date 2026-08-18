import json, datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

PARQUET_IN  = "/home/hduser/deepfake-data/processed/parquet/feature_table_augmented.parquet"
FEATURES_IN = "/home/hduser/deepfake-thesis/logs/phase3_surviving_features.json"
PHASE4_LOG  = "/home/hduser/deepfake-thesis/logs/phase4_models_log.json"
REPORT_DIR  = Path("/home/hduser/deepfake-thesis/reports")
LOG_PATH    = "/home/hduser/deepfake-thesis/logs/phase6b_compression_log.json"
RANDOM_SEED = 42
N_TRIALS    = 20
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(FEATURES_IN) as fh: FEATURES = json.load(fh)["final_features"]
with open(PHASE4_LOG)  as fh: phase4   = json.load(fh)

df = pd.read_parquet(PARQUET_IN)
train_idx = np.array(phase4["train_idx"])
test_idx  = np.array(phase4["test_idx"])
rf_p      = phase4["rf_best_params"]

X = df[FEATURES].values
y = (df["class_label"] == "fake").astype(int).values
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

rf = RandomForestClassifier(
    n_estimators=rf_p["n_estimators"], max_depth=rf_p["max_depth"],
    min_samples_leaf=rf_p["min_samples_leaf"], max_features=rf_p["max_features"],
    class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)

baseline_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
feat_stds    = X_test.std(axis=0)
print(f"Baseline AUC (zero noise): {baseline_auc:.4f}")
print(f"Feature stds: {dict(zip(FEATURES, feat_stds.round(5)))}")

noise_fractions = np.linspace(0, 2.0, 11)
rng = np.random.default_rng(RANDOM_SEED)
stress_results = []

print(f"\n{'Noise fraction':<20} {'Mean AUC':>10}  {'Std':>8}")
print("-" * 42)

for frac in noise_fractions:
    trial_aucs = []
    for _ in range(N_TRIALS):
        noise  = rng.normal(0, 1, X_test.shape) * (feat_stds * frac)
        y_prob = rf.predict_proba(X_test + noise)[:, 1]
        trial_aucs.append(roc_auc_score(y_test, y_prob))
    mean_auc = float(np.mean(trial_aucs))
    std_auc  = float(np.std(trial_aucs))
    print(f"  sigma={frac:.1f}x feat_std     {mean_auc:.4f}    ({std_auc:.4f})  "
          f"delta={mean_auc - baseline_auc:+.4f}")
    stress_results.append({"noise_fraction": round(frac, 2),
                            "mean_auc": round(mean_auc, 4),
                            "std_auc":  round(std_auc, 4),
                            "delta":    round(mean_auc - baseline_auc, 4)})

# Plot
fracs     = [r["noise_fraction"] for r in stress_results]
means_arr = np.array([r["mean_auc"] for r in stress_results])
stds_arr  = np.array([r["std_auc"]  for r in stress_results])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(fracs, means_arr, color="#01696f", linewidth=2.0,
        marker="o", markersize=5, label="Mean AUC (20 trials)")
ax.fill_between(fracs, means_arr - stds_arr, means_arr + stds_arr,
                alpha=0.15, color="#01696f", label="±1 std dev")
ax.axhline(0.5, color="#aaaaaa", linewidth=0.8, linestyle="--",
           label="Random baseline (AUC=0.50)")
ax.axhline(baseline_auc, color="#964219", linewidth=1.0, linestyle=":",
           label=f"Clean baseline AUC={baseline_auc:.3f}")
ax.set_xlabel("Gaussian noise (multiples of feature std dev)", fontsize=11)
ax.set_ylabel("ROC-AUC", fontsize=11)
ax.set_title("RF Model Robustness: AUC vs Feature Noise Level\n"
             "(Simulating increasing codec compression degradation)", fontsize=11)
ax.set_ylim(0.40, 0.80)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
out = REPORT_DIR / "compression_stress_auc_curve.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {out}")

threshold_frac = next((r["noise_fraction"] for r in stress_results
                       if r["mean_auc"] < 0.55), None)
log = {"run_timestamp": datetime.datetime.now().isoformat(),
       "features": FEATURES,
       "feature_stds": {f: round(float(s), 6) for f, s in zip(FEATURES, feat_stds)},
       "n_trials_per_level": N_TRIALS, "baseline_auc": round(baseline_auc, 4),
       "degradation_threshold_fraction": threshold_frac,
       "stress_results": stress_results, "plot": str(out)}
with open(LOG_PATH, "w") as fh: json.dump(log, fh, indent=2)
print(f"Log saved: {LOG_PATH}")
print("\nPhase 6b Compression Stress COMPLETE.")
