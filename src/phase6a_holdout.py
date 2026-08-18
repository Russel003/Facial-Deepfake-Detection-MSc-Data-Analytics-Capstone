import json, datetime
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import warnings; warnings.filterwarnings("ignore")

PARQUET_IN  = "/home/hduser/deepfake-data/processed/parquet/feature_table_augmented.parquet"
FEATURES_IN = "/home/hduser/deepfake-thesis/logs/phase3_surviving_features.json"
PHASE4_LOG  = "/home/hduser/deepfake-thesis/logs/phase4_models_log.json"
LOG_PATH    = "/home/hduser/deepfake-thesis/logs/phase6a_holdout_log.json"
RANDOM_SEED = 42

with open(FEATURES_IN) as fh: FEATURES = json.load(fh)["final_features"]
with open(PHASE4_LOG)  as fh: phase4   = json.load(fh)

df = pd.read_parquet(PARQUET_IN)
print(f"Loaded: {len(df)} rows | manipulation_type values: {df['manipulation_type'].value_counts().to_dict()}")

real_df     = df[df["class_label"] == "real"].copy()
fake_df     = df[df["class_label"] == "fake"].copy()
manip_types = [t for t in fake_df["manipulation_type"].unique()
               if t not in ("real", "unknown", None, "")]
print(f"Manipulation types: {manip_types}")

rf_p = phase4["rf_best_params"]

def make_rf():
    return RandomForestClassifier(
        n_estimators=rf_p["n_estimators"], max_depth=rf_p["max_depth"],
        min_samples_leaf=rf_p["min_samples_leaf"], max_features=rf_p["max_features"],
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)

baseline_auc = phase4["rf_test_metrics"]["auc"]
print(f"\nBaseline in-distribution AUC (Phase 4): {baseline_auc:.4f}")
print("=" * 60)
print("LEAVE-ONE-MANIPULATION-TYPE-OUT")
print("=" * 60)

results = []
for held_out in manip_types:
    train_types = [t for t in manip_types if t != held_out]
    train_fake  = fake_df[fake_df["manipulation_type"].isin(train_types)]
    test_fake   = fake_df[fake_df["manipulation_type"] == held_out]

    n_real_test  = max(1, int(len(real_df) * len(test_fake) / len(fake_df)))
    real_s       = real_df.sample(frac=1, random_state=RANDOM_SEED)
    train_real   = real_s.iloc[n_real_test:]
    test_real    = real_s.iloc[:n_real_test]

    train_d = pd.concat([train_fake, train_real])
    test_d  = pd.concat([test_fake,  test_real])

    X_tr = train_d[FEATURES].values
    y_tr = (train_d["class_label"] == "fake").astype(int).values
    X_te = test_d[FEATURES].values
    y_te = (test_d["class_label"] == "fake").astype(int).values

    rf = make_rf(); rf.fit(X_tr, y_tr)
    y_prob = rf.predict_proba(X_te)[:, 1]
    y_pred = rf.predict(X_te)
    auc_v  = roc_auc_score(y_te, y_prob)
    f1_v   = f1_score(y_te, y_pred)
    cm     = confusion_matrix(y_te, y_pred).tolist()

    print(f"\n  Held-out : {held_out}")
    print(f"  Train on : {train_types}")
    print(f"  Train size: {len(X_tr)} | Test size: {len(X_te)}")
    print(f"  AUC={auc_v:.4f}  F1={f1_v:.4f}  delta={auc_v-baseline_auc:+.4f}")
    print(f"  CM: {cm}")
    results.append({"held_out_type": held_out, "train_types": train_types,
                    "auc": round(auc_v, 4), "f1": round(f1_v, 4),
                    "delta": round(auc_v - baseline_auc, 4),
                    "confusion_matrix": cm})

mean_auc = float(np.mean([r["auc"] for r in results]))
print("\n" + "=" * 60)
print(f"Mean holdout AUC : {mean_auc:.4f}")
print(f"Generalisation gap: {mean_auc - baseline_auc:+.4f}")
if mean_auc >= baseline_auc - 0.05:
    print("Generalisation: STRONG (gap < 0.05) — features not manipulation-specific.")
elif mean_auc >= baseline_auc - 0.10:
    print("Generalisation: MODERATE (gap 0.05-0.10) — partially method-specific.")
else:
    print("Generalisation: WEAK (gap > 0.10) — features are method-sensitive.")

log = {"run_timestamp": datetime.datetime.now().isoformat(),
       "features": FEATURES, "manip_types": manip_types,
       "baseline_auc": baseline_auc, "mean_holdout_auc": round(mean_auc, 4),
       "generalisation_gap": round(mean_auc - baseline_auc, 4),
       "results": results}
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(LOG_PATH, "w") as fh: json.dump(log, fh, indent=2)
print(f"\nLog saved: {LOG_PATH}")
print("\nPhase 6a Holdout COMPLETE.")
