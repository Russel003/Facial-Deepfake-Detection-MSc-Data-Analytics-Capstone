"""
app.py — Streamlit Demo: Facial Deepfake Detector
MSc Data Analytics Capstone · CCT College Dublin · 2026

Pages:
  1. Watch & Guess          — video comparison hook
  2. Guess With Features    — feature-level quiz
  3. Model Predictions      — live inference on test set
  4. Feature Importance     — SHAP explainability
  5. Model Performance      — full results + robustness
  6. About This Research    — context, ethics, limitations

Run: streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# ── Path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.config import (
    DATA_ROOT,
    FEATURE_TABLE_CSV,
    LABEL_COL,
    LOGS_DIR,
    POSITIVE_CLASS,
    RANDOM_SEED,
    VALIDATED_FEATURES,
    REPORTS_DIR,
)
from src.data.loading import build_phase4_split, load_feature_table
from src.models.evaluation import bootstrap_confidence_intervals, compute_binary_metrics

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Facial Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load phase 4 log once ─────────────────────────────────────────────────────
PHASE4_LOG = LOGS_DIR / "phase4_models_log.json"

def _load_rf_params() -> dict:
    if PHASE4_LOG.exists():
        return json.loads(PHASE4_LOG.read_text())["rf_best_params"]
    return {"n_estimators": 500, "max_depth": 3, "min_samples_leaf": 10, "max_features": "sqrt"}

def _load_ci() -> str:
    if PHASE4_LOG.exists():
        d  = json.loads(PHASE4_LOG.read_text())
        ci = d.get("rf_ci", {})
        lo = ci.get("auc_lower", 0.57)
        hi = ci.get("auc_upper", 0.75)
        return f"{lo:.2f}–{hi:.2f}"
    return "0.57–0.75"

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on FaceForensics++ data…")
def get_model():
    rf = RandomForestClassifier(
        **_load_rf_params(),
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    X_train, y_train, X_test, y_test = build_phase4_split(feature_columns=VALIDATED_FEATURES)
    rf.fit(X_train, y_train)
    return rf, X_train, X_test, y_test

@st.cache_data(show_spinner="Loading feature table…")
def get_data() -> pd.DataFrame:
    return load_feature_table()

rf_model, X_train_cached, X_test_cached, y_test_cached = get_model()
df = get_data()

# ── Video paths ───────────────────────────────────────────────────────────────
_DEMO = REPO / "demo_videos"
_FFPP = DATA_ROOT / "ffpp"

if (_DEMO / "real").exists():
    REAL_VIDS = _DEMO / "real"
    FAKE_VIDS = {
        "Deepfakes":      _DEMO / "fake" / "Deepfakes",
        "Face2Face":      _DEMO / "fake" / "Face2Face",
        "FaceSwap":       _DEMO / "fake" / "FaceSwap",
        "NeuralTextures": _DEMO / "fake" / "NeuralTextures",
    }
else:
    REAL_VIDS = _FFPP / "original_sequences" / "youtube" / "c40" / "videos"
    FAKE_VIDS = {
        "Deepfakes":      _FFPP / "manipulated_sequences" / "Deepfakes"      / "c40" / "videos",
        "Face2Face":      _FFPP / "manipulated_sequences" / "Face2Face"       / "c40" / "videos",
        "FaceSwap":       _FFPP / "manipulated_sequences" / "FaceSwap"        / "c40" / "videos",
        "NeuralTextures": _FFPP / "manipulated_sequences" / "NeuralTextures"  / "c40" / "videos",
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Facial Deepfake Detector")
st.sidebar.markdown(f"""
**MSc Data Analytics Capstone**
CCT College Dublin · 2026

**Student:** Russel Prashant Shah
**Supervisor:** Dr. Kislay

---
| | |
|---|---|
| Model | Random Forest (500 trees) |
| Features | 3 physiological signals |
| Dataset | FaceForensics++ c40 · 1,000 videos |
| Test AUC | **0.6586** |
| CV AUC | **0.6777** (5-fold) |
| 95% CI | {_load_ci()} |
| Holdout AUC | **0.694** (mean, 4 types) |

---
*Proof of concept only. Not for deployment.*
*SHAP explanations satisfy EU AI Act Article 13.*
""")

page = st.sidebar.radio("Navigation", [
    "🎬 Watch & Guess",
    "📊 Guess With Features",
    "🤖 Model Predictions",
    "🔍 Feature Importance (SHAP)",
    "📈 Model Performance",
    "📄 About This Research",
])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — WATCH & GUESS
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🎬 Watch & Guess":
    st.title("Watch the Videos — Which is Real?")
    st.markdown("Watch both clips and decide which one is real. The model uses only 3 numbers to make the same call.")

    real_vids = sorted(REAL_VIDS.glob("*.mp4")) if REAL_VIDS.exists() else []
    fake_vids = []
    for label, d in FAKE_VIDS.items():
        if d.exists():
            for v in sorted(d.glob("*.mp4"))[:50]:
                fake_vids.append((v, label))

    if not real_vids or not fake_vids:
        st.info(
            "Video files not found on this machine. "
            "Set the DEEPFAKE_DATA_ROOT environment variable to your FaceForensics++ "
            "directory and restart. Showing feature-level evidence below."
        )
        st.markdown("---")
        st.subheader("Feature Distributions — Real vs Fake")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, feat in zip(axes, VALIDATED_FEATURES):
            real_vals = df[df[LABEL_COL] == "real"][feat]
            fake_vals = df[df[LABEL_COL] == "fake"][feat]
            ax.boxplot([real_vals, fake_vals], labels=["Real", "Fake"], patch_artist=True)
            ax.set_title(feat)
        plt.suptitle("These are the three signals the model uses", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(
            "ear_std: eye aperture variability across frames. "
            "ear_min: minimum eye opening (blink depth). "
            "lm_ratio_var: facial landmark geometry variance."
        )
    else:
        rng       = np.random.RandomState(st.session_state.get("video_seed", 99))
        real_path = real_vids[rng.choice(len(real_vids))]
        fake_path, fake_type = fake_vids[rng.choice(len(fake_vids))]
        swap      = st.session_state.get("swap", False)

        left_path,  left_label  = (real_path, "real") if not swap else (fake_path, "fake")
        right_path, right_label = (fake_path, "fake") if not swap else (real_path, "real")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Video A")
            st.video(str(left_path))
        with col2:
            st.subheader("Video B")
            st.video(str(right_path))

        guess = st.radio("Which video is REAL?", ["Video A", "Video B"], horizontal=True)
        c1, c2 = st.columns(2)
        reveal = c1.button("Reveal Answer", type="primary")
        if c2.button("Load Different Pair"):
            st.session_state["video_seed"] = st.session_state.get("video_seed", 99) + 7
            st.session_state["swap"]       = not st.session_state.get("swap", False)
            st.rerun()

        if reveal:
            correct = (
                (guess == "Video A" and left_label == "real") or
                (guess == "Video B" and right_label == "real")
            )
            if correct:
                st.success("Correct! You identified the real video.")
            else:
                st.error("Incorrect. The other video was real.")
            st.info(f"Real: {real_path.name}  |  Fake: {fake_path.name} (type: {fake_type})")
            row = df[df["video_id"].astype(str) == str(real_path.stem)]
            if not row.empty:
                proba = rf_model.predict_proba(row[VALIDATED_FEATURES].values)[0][1]
                pred  = "Fake" if proba >= 0.5 else "Real"
                st.metric("Model prediction (real video)", pred, delta=f"Fake probability: {proba:.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GUESS WITH FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Guess With Features":
    st.title("Can You Tell Real from Fake Using Only the Features?")
    st.markdown(
        "No video — just three numbers per clip. "
        "These are the same signals the model uses. "
        "Real faces tend to show higher eye variability and deeper blinks."
    )

    sample = df.sample(10, random_state=42)[
        ["video_id", LABEL_COL] + VALIDATED_FEATURES
    ].reset_index(drop=True)

    rename = {
        "ear_std":      "Eye Variation (ear_std)",
        "ear_min":      "Min Eye Opening (ear_min)",
        "lm_ratio_var": "Geometry Stability (lm_ratio_var)",
    }
    st.dataframe(
        sample[["video_id"] + VALIDATED_FEATURES].rename(columns=rename)
            .style.format("{:.5f}", subset=list(rename.values())),
        use_container_width=True,
    )
    st.caption(
        "Higher ear_std = more variable eye opening. "
        "Lower ear_min = deeper blinks. "
        "Higher lm_ratio_var = more facial geometry movement."
    )

    guesses: dict[int, str] = {}
    cols = st.columns(5)
    for i, row in sample.iterrows():
        guesses[i] = cols[i % 5].radio(f"Video {row['video_id']}", ["Real", "Fake"], key=f"g_{i}")

    if st.button("Reveal Answers + Model Predictions", type="primary"):
        proba       = rf_model.predict_proba(sample[VALIDATED_FEATURES].values)[:, 1]
        pred_labels = ["Fake" if p >= 0.5 else "Real" for p in proba]
        true_labels = [v.capitalize() for v in sample[LABEL_COL].tolist()]
        results, your_score, model_score = [], 0, 0
        for i, row in sample.iterrows():
            you_ok    = guesses[i] == true_labels[i]
            model_ok  = pred_labels[i] == true_labels[i]
            your_score  += int(you_ok)
            model_score += int(model_ok)
            conf = proba[i] if pred_labels[i] == "Fake" else 1 - proba[i]
            results.append({
                "Video":            row["video_id"],
                "True Label":       true_labels[i],
                "Your Guess":       f"{'✅' if you_ok   else '❌'} {guesses[i]}",
                "Model Prediction": f"{'✅' if model_ok  else '❌'} {pred_labels[i]}",
                "Model Confidence": f"{conf:.1%}",
            })
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Your Score",  f"{your_score}/10")
        c2.metric("Model Score", f"{model_score}/10")
        if your_score < model_score:
            st.warning("The model outperformed you — high-quality deepfakes fool humans consistently.")
        elif your_score == model_score:
            st.info("You matched the model. Both approaches face the same compression floor.")
        else:
            st.success("You beat the model. Note: the model uses only 3 features and no visual information.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Predictions":
    st.title("Model Predictions on the Hold-Out Test Set")
    st.markdown("Trained on 800 videos. Evaluated on 200 it never saw during training.")

    proba  = rf_model.predict_proba(X_test_cached)[:, 1]
    preds  = (proba >= 0.5).astype(int)
    y_true = np.array(y_test_cached)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Prediction Table (first 50)")
        st.dataframe(pd.DataFrame({
            "True Label":       ["Fake" if y == 1 else "Real" for y in y_true],
            "Predicted":        ["Fake" if p == 1 else "Real" for p in preds],
            "Fake Probability": np.round(proba, 4),
            "Correct":          preds == y_true,
        }).head(50), use_container_width=True)
    with col2:
        st.subheader("Summary Metrics")
        metrics = compute_binary_metrics(y_true, proba)
        ci      = bootstrap_confidence_intervals(y_true, proba, seed=RANDOM_SEED)
        st.metric("AUC",       f"{metrics['auc']:.4f}")
        st.caption(f"95% CI: [{ci['auc']['lower']:.4f}, {ci['auc']['upper']:.4f}]")
        st.metric("F1 Score",  f"{metrics['f1']:.4f}")
        st.metric("Precision", f"{metrics['precision']:.4f}")
        st.metric("Recall",    f"{metrics['recall']:.4f}")
        st.metric("Accuracy",  f"{metrics['accuracy']:.4f}")

    st.subheader("Fake Probability Distribution")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(proba[y_true == 0], bins=20, alpha=0.6, label="Real videos",  color="#2196F3")
    ax.hist(proba[y_true == 1], bins=20, alpha=0.6, label="Fake videos",  color="#F44336")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision threshold (0.5)")
    ax.set_xlabel("Predicted Fake Probability")
    ax.set_ylabel("Count")
    ax.set_title("Score separation — real vs fake")
    ax.legend()
    st.pyplot(fig)
    plt.close()
    st.caption(
        "The score overlap is expected for a 3-feature CPU-only model on c40 compression. "
        "Perfect separation would require deep visual features."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance (SHAP)":
    st.title("Why Does the Model Make These Decisions?")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) assigns each feature a contribution score "
        "for every individual prediction. This satisfies **EU AI Act Article 13** transparency "
        "requirements for high-risk AI systems. The model here is a research prototype — "
        "SHAP is used to show which physiological signals drove each decision."
    )

    for caption, fname in [
        ("SHAP Summary — Random Forest (beeswarm)", "shap_rf_summary.png"),
        ("SHAP Bar Chart — Mean |SHAP| per feature", "shap_rf_bar.png"),
    ]:
        fpath = REPORTS_DIR / fname
        if fpath.exists():
            st.image(str(fpath), caption=caption, use_column_width=True)

    n     = min(200, len(X_test_cached))
    X_exp = np.array(X_test_cached[:n])
    with st.spinner("Computing live SHAP values…"):
        explainer = shap.TreeExplainer(rf_model)
        shap_out  = explainer.shap_values(X_exp)
        sv = np.array(shap_out[1] if isinstance(shap_out, list) else shap_out)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mean |SHAP| per Feature")
        mean_shap = np.abs(sv).mean(axis=0)
        shap_df   = pd.DataFrame({
            "Feature":     VALIDATED_FEATURES,
            "Mean |SHAP|": [float(v) for v in mean_shap],
        }).sort_values("Mean |SHAP|", ascending=False)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(shap_df["Feature"].tolist(), shap_df["Mean |SHAP|"].tolist(),
                color=["#01696f", "#964219", "#a12c7b"])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature importance by SHAP")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()
        st.caption(
            "lm_ratio_var consistently ranks first in both RF and XGBoost. "
            "This confirms that facial geometry variance is the primary signal, "
            "with eye-aperture features (ear_std, ear_min) providing secondary support."
        )
    with col2:
        st.subheader("What Each Feature Measures")
        st.markdown("""
**`lm_ratio_var`** — Facial Geometry Variance

Variation in facial landmark distances across 100 sampled frames.
The dominant SHAP feature in both models.
Captures spatial artefacts from synthetic face generation.

**`ear_std`** — Eye Aperture Variability

Standard deviation of eye opening across the clip.
Deepfakes show unnaturally consistent eye openness.
Cohen's d = 0.45 — the strongest individual effect size.

**`ear_min`** — Minimum Eye Opening

Smallest eye aperture recorded in the clip.
Linked to natural blink depth that deepfakes suppress.
Cohen's d = 0.28.
        """)

    st.subheader("SHAP Values — First 50 Test Samples")
    sv_df = pd.DataFrame(sv[:50].round(5), columns=VALIDATED_FEATURES)
    sv_df.insert(0, "True Label", ["Fake" if y == 1 else "Real"
                                   for y in np.array(y_test_cached[:n])[:50]])
    st.dataframe(sv_df, use_container_width=True)
    st.caption("Positive SHAP = pushes toward Fake. Negative SHAP = pushes toward Real.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("Full Model Performance Summary")

    ph6a_path = LOGS_DIR / "phase6a_holdout_log.json"
    ph6b_path = LOGS_DIR / "phase6b_compression_log.json"

    ph4 = json.loads(PHASE4_LOG.read_text())
    rf  = ph4["rf_test_metrics"]
    xgb = ph4["xgb_test_metrics"]
    ens = ph4["ensemble_test_metrics"]

    st.subheader("All Three Models — Test Set (n=200)")
    st.dataframe(pd.DataFrame([
        {"Model": "Random Forest",     "AUC": rf["auc"],  "F1": rf["f1"],
         "Precision": rf["precision"], "Recall": rf["recall"],
         "Note": "Best model. Stable, interpretable, CPU-friendly."},
        {"Model": "XGBoost",           "AUC": xgb["auc"], "F1": xgb["f1"],
         "Precision": xgb["precision"],"Recall": xgb["recall"],
         "Note": "Gradient boosting. Slightly lower AUC."},
        {"Model": "Ensemble (RF+XGB)", "AUC": ens["auc"], "F1": ens["f1"],
         "Precision": ens["precision"],"Recall": ens["recall"],
         "Note": "Did not outperform RF alone — weak diversity gain at 3 features."},
    ]), use_container_width=True)
    st.caption(
        "RF wins despite the ensemble, which is unusual. "
        "With only 3 features, RF and XGBoost share similar error structure, "
        "so the ensemble gains little diversity."
    )

    st.subheader("Robustness Test 1 — Manipulation Hold-Out")
    st.markdown(
        "Trained on 3 manipulation types, tested on the 4th. "
        "Repeated for all 4 types. Mean holdout AUC (0.694) exceeds the within-distribution "
        "baseline (0.6586), showing the features transfer across manipulation methods."
    )
    if ph6a_path.exists():
        p6a  = json.loads(ph6a_path.read_text())
        rows = [{"Held-Out Type": r["held_out_type"], "AUC": r["auc"],
                 "F1": r["f1"], "Delta vs Baseline": r["delta"]}
                for r in p6a["results"]]
        rows.append({
            "Held-Out Type": "MEAN",
            "AUC": p6a["mean_holdout_auc"],
            "F1": "None",
            "Delta vs Baseline": p6a["generalisation_gap"],
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(
            "FaceSwap is hardest (AUC 0.62). This is expected: FaceSwap alters facial geometry "
            "structurally, reducing the discriminative signal available to eye-aperture features. "
            "Deepfakes is easiest (AUC 0.79) because identity-level synthesis produces "
            "more consistent eye-blink artefacts."
        )

    st.subheader("Robustness Test 2 — Compression Stress Test")
    st.markdown(
        "Gaussian noise added at increasing multiples of each feature's standard deviation. "
        "20 trials per noise level. This simulates progressive codec degradation."
    )
    if ph6b_path.exists():
        p6b    = json.loads(ph6b_path.read_text())
        stress = p6b["stress_results"]
        fracs  = [r["noise_fraction"] for r in stress]
        aucs   = np.array([r["mean_auc"] for r in stress])
        stds   = np.array([r["std_auc"]  for r in stress])
        b_auc  = p6b["baseline_auc"]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(fracs, aucs, marker="o", color="#01696f", linewidth=2, label="Mean AUC (20 trials)")
        ax.fill_between(fracs, aucs - stds, aucs + stds, alpha=0.15, color="#01696f", label="±1 std")
        ax.axhline(0.5,   color="gray",    linestyle="--", linewidth=1,   label="Random chance (0.5)")
        ax.axhline(b_auc, color="#964219", linestyle=":",  linewidth=1.2, label=f"Baseline AUC ({b_auc:.4f})")
        ax.set_xlabel("Noise fraction (× feature std)")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.40, 0.75)
        ax.set_title("AUC under increasing compression noise")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()
        st.caption(
            f"AUC drops from {b_auc:.4f} at zero noise to 0.5719 at 2x std. "
            "The decay is gradual, not catastrophic. "
            "The model retains above-chance performance across the full noise range tested."
        )

    st.subheader("Honest Assessment")
    st.warning(
        "Single dataset only (FaceForensics++ c40). Cross-dataset generalisation is unverified.\n\n"
        "No demographic labels in FaceForensics++. Fairness across skin tones and demographics "
        "cannot be quantified. This is a known limitation, not an evaluated contribution.\n\n"
        "AUC 0.6586 is below deep-learning benchmarks (~0.90+). This is expected for a "
        "CPU-only 3-feature model and is consistent with the proof-of-concept framing.\n\n"
        "Gaussian noise approximates but does not replicate real codec compression artefacts.\n\n"
        "Social media footage is likely harder than controlled FaceForensics benchmark clips."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT THIS RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄 About This Research":
    st.title("About This Research")
    st.caption("MSc Data Analytics Capstone · CCT College Dublin · 2026")

    st.markdown("""
### Research Question
> *Can interpretable, engineered facial and temporal features extracted from
> FaceForensics++ c40 support deepfake detection under constrained, CPU-only compute?*

### What This System Is

This is a proof of concept. It demonstrates that three physiologically grounded video-level
features carry discriminative signal for deepfake detection, even under c40 heavy compression.
It is not a deployment-ready forensic tool. AUC 0.6586 is above chance on a hard benchmark,
but it is not sufficient to serve as standalone evidence in any real-world context.

The contribution is not peak accuracy. It is interpretability, portability, and auditability
in a pipeline that runs without GPU hardware and produces a SHAP explanation for every prediction.

### Why These Three Features

27 candidate features were engineered from 68-point dlib facial landmarks sampled across
100 frames per video. A statistical selection pipeline reduced this to three:

- **ear_std** passed Mann-Whitney U (BH-corrected p = 0.0) with Cohen's d = 0.45.
  It carries the strongest individual effect size.
- **ear_min** passed with p = 0.0017 and Cohen's d = 0.28.
- **lm_ratio_var** was added from Phase 5 SHAP analysis as the dominant model signal.

The full 27-feature model improves AUC by approximately 0.02. That gain sits within the
bootstrap confidence interval [0.57, 0.75] and does not justify the loss of interpretability.
Three features with documented selection logic are more defensible than 27 features
with weaker individual evidence.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### Pipeline Summary

| Phase | What it does |
|---|---|
| Phase 2 | dlib landmark extraction — 68 points, 100 frames/video, ~4.5 min CPU |
| Phase 3 | Mann-Whitney U + Cohen's d + BH correction — 3 features from 27 |
| Phase 4 | RF + XGBoost + Ensemble — 5-fold CV, RF wins (AUC 0.6586) |
| Phase 5 | SHAP TreeExplainer — per-prediction attribution |
| Phase 6a | Leave-one-manipulation-out — mean AUC 0.694 |
| Phase 6b | Gaussian noise stress test — AUC holds above 0.57 at 2× std |
        """)
    with col2:
        st.markdown("""
### Canonical Results

| Metric | Value |
|---|---|
| Videos | 1,000 (200 real, 800 fake) |
| Manipulation types | Deepfakes, Face2Face, FaceSwap, NeuralTextures |
| Train / Test | 800 / 200 (stratified, video-level) |
| CV | 5-fold GridSearchCV |
| RF Test AUC | 0.6586 |
| RF CV AUC | 0.6777 |
| Bootstrap 95% CI | 0.57–0.75 |
| Holdout AUC (mean) | 0.694 |
| Compression floor AUC | 0.5719 (at noise 2.0×) |
| AUC drop | 8.67 pp across noise range |
        """)

    st.markdown("---")

    st.markdown("""
### Ethical and Regulatory Position

This pipeline was designed with EU AI Act Article 13 in mind. Every prediction carries
a SHAP attribution that identifies which feature drove the decision and by how much.
This supports the transparency requirement for high-risk AI applications.

Two firm limitations apply to any ethical use of this system:

1. **No demographic evaluation.** FaceForensics++ has no demographic labels.
   Fairness across skin tones, age groups, or genders was not and cannot be
   assessed with this dataset. Any operational deployment would require demographic
   auditing before use.

2. **Single benchmark only.** All results are internal to FaceForensics++ c40.
   Cross-dataset generalisation to social media footage, webcam recordings, or
   other synthesis methods is unverified.

### What Would Be Needed for Deployment

This system would require the following before any real-world use:

- Multi-dataset training and cross-dataset validation.
- Demographic subgroup auditing across skin tone, gender, and age.
- Adversarial testing against state-of-the-art generation methods (e.g., diffusion-based synthesis).
- Integration with a human review process — no automated forensic decision should rest
  on a single model with AUC 0.66.
    """)

    st.info(
        "This app supports reproducibility and communication. "
        "It does not change model performance or strengthen the empirical results. "
        "The thesis and notebooks contain the full experimental record."
    )

    st.markdown("---")
    st.caption("Russel Prashant Shah · Supervisor: Dr. Kislay · CCT College Dublin · 2026")