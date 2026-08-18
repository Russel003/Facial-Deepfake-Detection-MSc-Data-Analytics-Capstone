"""
Phase 6d: Exploratory Data Analysis — Thesis Figures
Produces 6 publication-ready figures for Chapter 4:
  1. Class distribution (bar chart)
  2. Feature distributions by class (boxplots: real vs fake, 3 features)
  3. Phase 3 Cohen's d ranking (horizontal bar chart, all 23 features)
  4. BH-corrected p-value vs Cohen's d scatter (significance landscape)
  5. Spearman correlation heatmap (3 surviving features)
  6. Feature pair scatter plots (real vs fake, pairwise)
All saved to /reports/ as PNG.
"""
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
PARQUET_IN  = "/home/hduser/deepfake-data/processed/parquet/feature_table_augmented.parquet"
PHASE3_LOG  = "/home/hduser/deepfake-thesis/logs/phase3_stats_log.json"
FEATURES_IN = "/home/hduser/deepfake-thesis/logs/phase3_surviving_features.json"
REPORT_DIR  = Path("/home/hduser/deepfake-thesis/reports")
LOG_PATH    = "/home/hduser/deepfake-thesis/logs/phase6d_eda_log.json"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Thesis color palette (real=teal, fake=terra)
CLR_REAL = "#01696f"
CLR_FAKE = "#964219"
CLR_SURV = "#437a22"   # surviving feature highlight

# ── LOAD ──────────────────────────────────────────────────────────────────────
with open(FEATURES_IN) as fh:
    FEATURES = json.load(fh)["final_features"]

with open(PHASE3_LOG) as fh:
    phase3 = json.load(fh)

df = pd.read_parquet(PARQUET_IN)
print(f"Loaded: {len(df)} rows | Features: {FEATURES}")

real_df = df[df["class_label"] == "real"]
fake_df = df[df["class_label"] == "fake"]

# ── FIGURE 1: CLASS DISTRIBUTION ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
counts  = df["class_label"].value_counts()
bars    = ax.bar(counts.index, counts.values,
                 color=[CLR_REAL, CLR_FAKE],
                 width=0.5, edgecolor="white")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5, str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=11)
ax.set_ylabel("Number of Videos", fontsize=11)
ax.set_title("Class Distribution\n(FaceForensics++ c40 subset)", fontsize=11)
ax.set_ylim(0, max(counts.values) * 1.15)
ax.grid(True, axis="y", alpha=0.2)
plt.tight_layout()
out1 = REPORT_DIR / "eda_class_distribution.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ── FIGURE 2: BOXPLOTS — REAL VS FAKE (3 SURVIVING FEATURES) ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, feat in zip(axes, FEATURES):
    data_real = real_df[feat].dropna().values
    data_fake = fake_df[feat].dropna().values

    bp = ax.boxplot([data_real, data_fake],
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=2,
                                   alpha=0.4, linestyle="none"))

    bp["boxes"][0].set_facecolor(CLR_REAL)
    bp["boxes"][1].set_facecolor(CLR_FAKE)
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_alpha(0.75)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Real", "Fake"], fontsize=10)
    ax.set_title(feat, fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)

    # Annotate Cohen's d
    d_val = None
    for entry in phase3.get("mw_results", []):
        if entry.get("feature") == feat:
            d_val = entry.get("cohens_d")
            break
    if d_val is not None:
        ax.text(0.97, 0.97, f"d = {abs(d_val):.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.7, edgecolor="#cccccc"))

p1 = mpatches.Patch(color=CLR_REAL, alpha=0.75, label="Real")
p2 = mpatches.Patch(color=CLR_FAKE, alpha=0.75, label="Fake")
fig.legend(handles=[p1, p2], loc="upper right",
           fontsize=9, framealpha=0.8)
fig.suptitle("Surviving Feature Distributions: Real vs Fake\n"
             "(Annotated with Cohen's d from Mann-Whitney U test)",
             fontsize=11, y=1.02)
plt.tight_layout()
out2 = REPORT_DIR / "eda_feature_boxplots.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ── FIGURE 3: COHEN'S d RANKING (all 23 features) ─────────────────────────────
if "mw_results" in phase3:
    all_results = sorted(phase3["mw_results"], key=lambda x: abs(x.get("cohens_d", 0)))
    feats_all   = [r["feature"] for r in all_results]
    d_vals      = [abs(r.get("cohens_d", 0)) for r in all_results]
    survives    = [r["feature"] in FEATURES for r in all_results]

    fig, ax = plt.subplots(figsize=(8, 7))
    bar_colors = [CLR_SURV if s else "#bbbbbb" for s in survives]
    bars = ax.barh(feats_all, d_vals, color=bar_colors, edgecolor="white")

    # Threshold line
    ax.axvline(0.2, color="#964219", linewidth=1.0, linestyle="--",
               label="d = 0.2 (small effect)")
    ax.axvline(0.5, color="#a12c7b", linewidth=1.0, linestyle="--",
               label="d = 0.5 (medium effect)")
    ax.set_xlabel("Cohen's d (absolute)", fontsize=10)
    ax.set_title("Cohen's d Effect Size — All 23 Features\n"
                 "(Green = survived BH-corrected p+d filter)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.2)

    p_surv = mpatches.Patch(color=CLR_SURV, label="Survived filter")
    p_drop = mpatches.Patch(color="#bbbbbb", label="Dropped")
    ax.legend(handles=[p_surv, p_drop], fontsize=8, loc="lower right")

    plt.tight_layout()
    out3 = REPORT_DIR / "eda_cohens_d_ranking.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out3}")
else:
    print("[INFO] mw_results not in phase3 log. Skipping Figure 3.")
    out3 = None

# ── FIGURE 4: SIGNIFICANCE LANDSCAPE (p_BH vs Cohen's d) ─────────────────────
if "mw_results" in phase3:
    d_all  = np.array([abs(r.get("cohens_d", 0)) for r in phase3["mw_results"]])
    p_all  = np.array([r.get("p_bh", r.get("p_BH", 1.0))
                       for r in phase3["mw_results"]])
    names_all = [r["feature"] for r in phase3["mw_results"]]
    surv_mask = np.array([n in FEATURES for n in names_all])

    # Clip p=0 to a small value for log scale
    p_plot = np.where(p_all == 0, 1e-8, p_all)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(d_all[~surv_mask], p_plot[~surv_mask],
               color="#cccccc", s=50, zorder=2, label="Dropped features")
    ax.scatter(d_all[surv_mask], p_plot[surv_mask],
               color=CLR_SURV, s=80, zorder=3,
               edgecolors="white", linewidth=0.8,
               label="Surviving features")

    # Label surviving features
    for i, n in enumerate(names_all):
        if n in FEATURES:
            ax.annotate(n, (d_all[i], p_plot[i]),
                        fontsize=8, xytext=(5, -12),
                        textcoords="offset points", color=CLR_SURV)

    ax.axhline(0.05, color="#964219", linewidth=0.8, linestyle="--",
               label="p_BH = 0.05")
    ax.axvline(0.2,  color="#7a39bb", linewidth=0.8, linestyle="--",
               label="d = 0.2")
    ax.set_yscale("log")
    ax.set_xlabel("Cohen's d (absolute)", fontsize=10)
    ax.set_ylabel("BH-corrected p-value (log scale)", fontsize=10)
    ax.set_title("Significance Landscape: All 23 Features\n"
                 "(Bottom-right quadrant = significant AND large effect)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    out4 = REPORT_DIR / "eda_significance_landscape.png"
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out4}")
else:
    print("[INFO] mw_results not in phase3 log. Skipping Figure 4.")
    out4 = None

# ── FIGURE 5: SPEARMAN CORRELATION HEATMAP (3 SURVIVING FEATURES) ────────────
corr = df[FEATURES].corr(method="spearman")

fig, ax = plt.subplots(figsize=(5, 4))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True   # upper triangle off

sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5,
            annot_kws={"size": 11})
ax.set_title("Spearman Correlation\n(Surviving Features — no collinearity > |0.85|)",
             fontsize=10)
plt.tight_layout()
out5 = REPORT_DIR / "eda_correlation_heatmap.png"
plt.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out5}")

# ── FIGURE 6: PAIRWISE SCATTER (REAL vs FAKE) ─────────────────────────────────
n_feat = len(FEATURES)
fig, axes = plt.subplots(n_feat, n_feat, figsize=(10, 9))

for i, fi in enumerate(FEATURES):
    for j, fj in enumerate(FEATURES):
        ax = axes[i][j]
        if i == j:
            # Histogram
            ax.hist(real_df[fi].dropna(), bins=25, color=CLR_REAL,
                    alpha=0.6, density=True)
            ax.hist(fake_df[fi].dropna(), bins=25, color=CLR_FAKE,
                    alpha=0.6, density=True)
        else:
            ax.scatter(real_df[fj].values, real_df[fi].values,
                       color=CLR_REAL, alpha=0.2, s=5)
            ax.scatter(fake_df[fj].values, fake_df[fi].values,
                       color=CLR_FAKE, alpha=0.2, s=5)
        if i == n_feat - 1:
            ax.set_xlabel(fj, fontsize=7)
        if j == 0:
            ax.set_ylabel(fi, fontsize=7)
        ax.tick_params(labelsize=5)

fig.suptitle("Pairwise Feature Scatter: Real (teal) vs Fake (brown)",
             fontsize=11, y=1.01)
plt.tight_layout()
out6 = REPORT_DIR / "eda_pairwise_scatter.png"
plt.savefig(out6, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out6}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
plots = [str(p) for p in [out1, out2, out3, out4, out5, out6] if p is not None]
print(f"\nAll EDA plots saved to {REPORT_DIR}")
for p in plots:
    print(f"  {p}")

log = {
    "run_timestamp": datetime.datetime.now().isoformat(),
    "features":      FEATURES,
    "plots_saved":   plots,
}
with open(LOG_PATH, "w") as fh:
    json.dump(log, fh, indent=2)

print(f"\nLog saved: {LOG_PATH}")
print("\nPhase 6d EDA COMPLETE.")
