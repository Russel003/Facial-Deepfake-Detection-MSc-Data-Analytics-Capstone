"""
SHAP analysis for the final Random Forest model on FaceForensics++.

This focuses on the validated 3-feature subset so that:
- the analysis lines up with the ablation Model A,
- the notebook can regenerate a clean SHAP figure quickly on CPU.
"""

from pathlib import Path

import numpy as np
import shap

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from src.utils.config import (
    VALIDATED_FEATURES,
    RF_PARAMS,
    RANDOM_SEED,
    FIGURES_DIR,
)
from src.data.loading import build_phase4_split


def train_final_rf(feature_columns=None):
    """
    Train the Random Forest on the Phase 4 split using the specified features.

    Returns the fitted model, split arrays, and the feature name list.
    """
    if feature_columns is None:
        feature_columns = VALIDATED_FEATURES

    X_train, y_train, X_test, y_test = build_phase4_split(feature_columns=feature_columns)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    return rf, X_train, y_train, X_test, y_test, feature_columns


def compute_shap_values(rf, X_explain):
    """
    Compute SHAP values for the positive class on the given explanation set.
    """
    explainer = shap.TreeExplainer(rf)
    shap_values_all = explainer.shap_values(X_explain)

    # Binary classification: index 1 corresponds to the positive class
    if isinstance(shap_values_all, list) and len(shap_values_all) == 2:
        shap_values = shap_values_all[1]
    else:
        # Fallback for different shap versions
        shap_values = shap_values_all

    return shap_values


def plot_shap_summary(shap_values, X_explain, feature_names, out_bar_path, out_beeswarm_path):
    """
    Save SHAP summary bar and beeswarm plots for the given SHAP values.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Bar plot: global importance ranking
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_bar_path, dpi=150)
    plt.close()
    print(f"SHAP bar plot saved: {out_bar_path}")

    # Beeswarm plot: distribution of effects per feature
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_beeswarm_path, dpi=150)
    plt.close()
    print(f"SHAP beeswarm plot saved: {out_beeswarm_path}")


def run_shap_analysis(n_samples: int = 200):
    """
    End-to-end SHAP analysis for the final RF model on a test subset.

    - trains RF on the validated 3-feature subset,
    - samples up to n_samples rows from the test set,
    - computes SHAP for the positive class,
    - saves bar and beeswarm plots,
    - prints mean absolute SHAP values for quick inspection.
    """
    rf, X_train, y_train, X_test, y_test, feature_names = train_final_rf()

    rng = np.random.default_rng(RANDOM_SEED)
    if len(X_test) > n_samples:
        idx = rng.choice(len(X_test), size=n_samples, replace=False)
        X_explain = X_test[idx]
    else:
        X_explain = X_test

    print(f"Using {X_explain.shape[0]} test samples for SHAP.")

    shap_values = compute_shap_values(rf, X_explain)

    bar_path = FIGURES_DIR / "shap_final_rf_bar.png"
    beeswarm_path = FIGURES_DIR / "shap_final_rf_beeswarm.png"
    plot_shap_summary(shap_values, X_explain, feature_names, bar_path, beeswarm_path)

    mean_abs = np.abs(shap_values).mean(axis=0).flatten()
    ranking = sorted(
        [(name, float(val)) for name, val in zip(feature_names, mean_abs)],
        key=lambda x: x[1],
        reverse=True,
    )

    print("\nMean |SHAP| per feature (descending):")
    for name, val in ranking:
        print(f"  {name:<20} {val:.6f}")

    return {
        "feature_names": feature_names,
        "mean_abs_shap": ranking,
        "bar_path": bar_path,
        "beeswarm_path": beeswarm_path,
    }


if __name__ == "__main__":
    run_shap_analysis()
