import os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import shapiro, mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)

# ── CONFIG ────────────────────────────────────────────────────────────────────
FEATURE_CSV  = "/home/hduser/deepfake-data/processed/features/feature_table.csv"
OUT_DIR      = "/home/hduser/deepfake-data/processed/features"
LOG_DIR      = "/home/hduser/deepfake-thesis/logs"
RANDOM_SEED  = 42
P_THRESHOLD  = 0.05
COHEN_D_MIN  = 0.5
SPEARMAN_CUT = 0.90

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv(FEATURE_CSV)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

META_COLS = ["video_id", "manipulation_type", "class_label",
             "total_frames", "frames_sampled", "frames_with_face"]
feature_cols = [c for c in df.columns if c not in META_COLS]
print(f"Feature columns to test: {len(feature_cols)}")
print(feature_cols)

# ── NaN AUDIT ─────────────────────────────────────────────────────────────────
nan_report = df[feature_cols].isnull().sum()
nan_features = nan_report[nan_report > 0]
if len(nan_features) > 0:
    print(f"\nWARNING: NaN values found in {len(nan_features)} features:")
    print(nan_features)
    for col in nan_features.index:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} NaNs with median={median_val:.4f}")
else:
    print("NaN audit: CLEAN")

# ── SPLIT BY CLASS ────────────────────────────────────────────────────────────
real = df[df["class_label"] == "real"][feature_cols]
fake = df[df["class_label"] == "fake"][feature_cols]
print(f"\nReal samples: {len(real)} | Fake samples: {len(fake)}")

# ── COHEN'S D HELPER ─────────────────────────────────────────────────────────
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2)
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return abs((np.mean(a) - np.mean(b)) / pooled_std)

# ── PER-FEATURE STATISTICAL TESTS ────────────────────────────────────────────
results = []

for col in feature_cols:
    r_vals = real[col].values
    f_vals = fake[col].values

    sw_stat_r, sw_p_r = shapiro(r_vals)
    sw_stat_f, sw_p_f = shapiro(f_vals)
    both_normal = (sw_p_r >= P_THRESHOLD) and (sw_p_f >= P_THRESHOLD)

    mw_stat, mw_p = mannwhitneyu(r_vals, f_vals, alternative="two-sided")
    d = cohens_d(r_vals, f_vals)
    retain = (mw_p < P_THRESHOLD) and (d >= COHEN_D_MIN)

    results.append({
        "feature":     col,
        "sw_p_real":   round(sw_p_r, 6),
        "sw_p_fake":   round(sw_p_f, 6),
        "both_normal": bool(both_normal),
        "mw_stat":     round(mw_stat, 2),
        "mw_p":        round(mw_p, 6),
        "cohens_d":    round(d, 4),
        "retain":      bool(retain),
    })
    flag = "KEEP" if retain else "DROP"
    print(f"[{flag}] {col:30s}  mw_p={mw_p:.5f}  d={d:.3f}  normal={both_normal}")

df_results = pd.DataFrame(results)
print(f"\nNaive retained: {df_results['retain'].sum()} / {len(df_results)}")

# ── BENJAMINI-HOCHBERG FDR CORRECTION ────────────────────────────────────────
pvals = df_results["mw_p"].values
reject_bh, pvals_corrected, _, _ = multipletests(pvals, alpha=P_THRESHOLD, method="fdr_bh")

df_results["mw_p_bh_corrected"] = np.round(pvals_corrected, 6)
df_results["retain_bh"] = reject_bh & (df_results["cohens_d"] >= COHEN_D_MIN)

print(f"\nAfter BH FDR correction:")
print(f"  Retained (BH p < 0.05 AND d >= 0.5): {df_results['retain_bh'].sum()} features")
print(df_results[df_results["retain_bh"]][
    ["feature", "mw_p", "mw_p_bh_corrected", "cohens_d"]
].to_string(index=False))

# ── SPEARMAN COLLINEARITY REMOVAL ────────────────────────────────────────────
retained_features = df_results[df_results["retain_bh"]]["feature"].tolist()
print(f"\nSpearman collinearity check on {len(retained_features)} retained features...")

df_retained = df[retained_features]
spearman_corr = df_retained.corr(method="spearman").abs()
upper = spearman_corr.where(
    np.triu(np.ones(spearman_corr.shape), k=1).astype(bool)
)

collinear_pairs = []
to_drop = set()

for col in upper.columns:
    for row in upper.index:
        val = upper.loc[row, col]
        if pd.notna(val) and val > SPEARMAN_CUT:
            collinear_pairs.append({
                "feature_a": row,
                "feature_b": col,
                "spearman_r": round(float(val), 4)
            })
            d_row = float(df_results.loc[df_results["feature"] == row, "cohens_d"].values[0])
            d_col = float(df_results.loc[df_results["feature"] == col, "cohens_d"].values[0])
            drop_target = row if d_row <= d_col else col
            to_drop.add(drop_target)
            print(f"  Collinear: {row} -- {col}  r={val:.3f}  -> DROP {drop_target}")

final_features = [f for f in retained_features if f not in to_drop]
print(f"\nFinal validated feature set ({len(final_features)} features):")
for f in final_features:
    d_val = float(df_results.loc[df_results["feature"] == f, "cohens_d"].values[0])
    p_val = float(df_results.loc[df_results["feature"] == f, "mw_p_bh_corrected"].values[0])
    print(f"  {f:30s}  d={d_val:.3f}  bh_p={p_val:.5f}")

# ── SAVE VALIDATED FEATURE TABLE ─────────────────────────────────────────────
out_cols = ["video_id", "manipulation_type", "class_label"] + final_features
df_validated = df[out_cols].copy()

validated_csv = os.path.join(OUT_DIR, "validated_feature_table.csv")
df_validated.to_csv(validated_csv, index=False)
print(f"\nValidated feature table saved: {validated_csv}")
print(f"Shape: {df_validated.shape}")

# ── SAVE LOGS ─────────────────────────────────────────────────────────────────
log = {
    "timestamp":               datetime.now().isoformat(),
    "random_seed":             RANDOM_SEED,
    "p_threshold":             P_THRESHOLD,
    "cohen_d_min":             COHEN_D_MIN,
    "spearman_cut":            SPEARMAN_CUT,
    "total_features_tested":   len(feature_cols),
    "retained_naive":          int(df_results["retain"].sum()),
    "retained_after_bh":       int(df_results["retain_bh"].sum()),
    "collinear_pairs":         collinear_pairs,
    "collinear_dropped":       list(to_drop),
    "final_feature_count":     len(final_features),
    "final_features":          final_features,
    "per_feature_results":     df_results.to_dict(orient="records"),
}

log_path = os.path.join(LOG_DIR, "phase3_statistical_validation_log.json")
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)
print(f"Log saved: {log_path}")

stats_csv = os.path.join(LOG_DIR, "phase3_feature_stats_table.csv")
df_results.to_csv(stats_csv, index=False)
print(f"Stats table saved: {stats_csv}")

print("\nPhase 3 COMPLETE.")
