# Facial Deepfake Detection — Interpretable CPU-Only Pipeline

**MSc Data Analytics · CCT College Dublin · May 2026**
**Student:** Russel Prashant Shah · **Supervisor:** Kislay Raj

---

## Research Question

Can a small set of engineered, physiologically grounded facial features detect
deepfakes under heavy compression FaceForensics (c40) compression, using only CPU-based classifiers,
while remaining interpretable and auditable under EU AI Act Article 13?

---

## What This Project Is

This is a proof of concept. It demonstrates that three video-level geometric
features carry discriminative signal for deepfake detection under FaceForensics++
c40 heavy compression. It is not a forensic-grade tool. AUC 0.6586 is above
chance on the hardest compression tier of FaceForensics++. It is not sufficient
for real-world deployment as standalone evidence.

The contribution is interpretability, portability, and auditability — a pipeline
that runs without GPU hardware and produces a SHAP explanation for every prediction.

---

## Canonical Results

| Model               | AUC        | F1     | Precision | Recall |
|---------------------|------------|--------|-----------|--------|
| Random Forest       | **0.6586** | 0.7800 | 0.8357    | 0.7312 |
| XGBoost             | 0.6280     | 0.7292 | 0.8203    | 0.6562 |
| Soft-Voting Ensemble| 0.6427     | 0.7609 | 0.8248    | 0.7063 |

- **Test set:** 200 videos · 160 fake · 40 real (stratified, video-level split)
- **CV AUC:** 0.6777 (5-fold, RF)
- **Bootstrap 95% CI:** 0.57–0.75
- **RF Confusion Matrix:** TN=17 · FP=23 · FN=43 · TP=117

### Generalisation — Leave-One-Manipulation-Out

| Held-Out Type  | AUC    | F1     |
|----------------|--------|--------|
| Deepfakes      | 0.7858 | 0.8446 |
| Face2Face      | 0.7253 | 0.7799 |
| NeuralTextures | 0.6487 | 0.7062 |
| FaceSwap       | 0.6162 | 0.6115 |
| **Mean**       | **0.694** | — |

Mean holdout AUC (0.694) exceeds the within-distribution baseline (0.6586).
FaceSwap is the hardest type — structural geometry changes suppress the
discriminative signal available to eye-aperture features.

### Compression Robustness

AUC declines from 0.6586 (no noise) to 0.5719 (noise = 2.0×).
Decay is gradual, not catastrophic. Total drop: 8.67 percentage points.

---

## The Three Features

27 candidate features were engineered from 68-point dlib facial landmarks
sampled across 100 frames per video. Three survived Mann-Whitney U
(Benjamini-Hochberg corrected), Cohen's d threshold (d > 0.2), and
Spearman collinearity screening:

| Feature        | Cohen's d | Mean SHAP (RF) | Role                              |
|----------------|-----------|----------------|-----------------------------------|
| `ear_std`      | 0.4542    | 0.04916        | Eye aperture variability across frames |
| `ear_min`      | 0.2843    | 0.03023        | Minimum eye opening in the clip   |
| `lm_ratio_var` | dominant  | 0.08881        | Facial landmark geometry variance |

`lm_ratio_var` is the top SHAP feature in both RF and XGBoost.
The full 27-feature model improves AUC by ~0.02 — a gain that sits within
the bootstrap CI and does not justify the interpretability cost.

---

## Pipeline Phases

| Phase    | Script                          | What it does                                              | Runtime (CPU) |
|----------|---------------------------------|-----------------------------------------------------------|---------------|
| Phase 2  | `src/phase2_etl.py`             | dlib landmark extraction · 68 points · 100 frames/video   | ~8 min        |
| Phase 3  | `src/phase3_feature_validation.py` | Mann-Whitney U · Cohen's d · BH correction · 3 from 27   | < 2 min       |
| Phase 4  | `src/phase4_models.py`          | RF + XGBoost + Ensemble · 5-fold GridSearchCV             | < 6 min       |
| Phase 5  | `src/phase5_shap.py`            | SHAP TreeExplainer · per-prediction attribution           | < 3 min       |
| Phase 6a | `src/phase6a_holdout.py`        | Leave-one-manipulation-out evaluation                     | < 2 min       |
| Phase 6b | `src/phase6b_compression_stress.py` | Gaussian noise stress test · 20 trials/level          | < 2 min       |
| Phase 6c | `src/phase6c_roc_curves.py`     | ROC + PR curves for all models                            | < 1 min       |
| Phase 6d | `src/phase6d_eda.py`            | EDA figures · class distribution · correlation heatmap    | < 1 min       |

**Total end-to-end (excluding ETL): under 20 minutes on CPU.**

---

## Repository Structure

```text
capstone-msc-da-feb-25-cohort-Russel003/
├── app.py # Streamlit demo (6 pages)
├── requirements.txt
├── .gitignore
├── Thesis-FacialDeepFakeDetection_2025004.docx # Final thesis submission
│
├── data/
│ └── features/
│ ├── feature_table_augmented.csv # 1000 videos · 30 columns · 255 KB
│ └── dataset_manifest.json
│
├── logs/ # All experiment logs (13 files)
│ ├── phase2_etl_log.json
│ ├── phase3_feature_stats_table.csv
│ ├── phase3_statistical_validation_log.json
│ ├── phase3_surviving_features.json
│ ├── phase3_validation_log.json
│ ├── phase4_best_model.json
│ ├── phase4_models_log.json
│ ├── phase5_shap_log.json
│ ├── phase6a_holdout_log.json
│ ├── phase6b_compression_log.json
│ ├── phase6c_roc_log.json
│ ├── phase6d_eda_log.json
│ └── video_count_report.json
│
├── notebooks/
│ ├── NB1_Dataset_and_Feature_Selection_2025004.ipynb
│ └── NB2_Model_Results_SHAP_Robustness_2025004.ipynb
│
├── reports/ # All 17 generated figures
│ ├── shap_rf_summary.png
│ ├── shap_rf_bar.png
│ ├── shap_xgb_summary.png
│ ├── shap_xgb_bar.png
│ ├── roc_curves_all_models.png
│ ├── pr_curves_all_models.png
│ ├── confusion_matrices_all_models.png
│ ├── compression_stress_auc_curve.png
│ ├── eda_class_distribution.png
│ ├── eda_cohens_d_ranking.png
│ ├── eda_feature_boxplots.png
│ ├── eda_correlation_heatmap.png
│ └── ...
│
└── src/
├── utils/
│ ├── config.py # All paths · constants · hyperparameters
│ └── logging_utils.py
├── phase2_etl.py
├── phase3_feature_validation.py
├── phase4_models.py
├── phase5_shap.py
├── phase6a_holdout.py
├── phase6b_compression_stress.py
├── phase6c_roc_curves.py
└── phase6d_eda.py
```

---

## Dataset

**FaceForensics++** (Rössler et al., 2019)

Raw videos are **not included** in this repository. Download from the official source:

**[https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)**

This project uses the **c40 (heavy compression)** variant only.

- 1,000 videos total: 200 real · 800 fake
- 4 manipulation types: Deepfakes · Face2Face · FaceSwap · NeuralTextures
- Class imbalance: 4:1 (fake:real)
- Only `data/features/feature_table_augmented.csv` is committed to this repo

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/CCT-Dublin/capstone-msc-da-feb-25-cohort-Russel003
cd capstone-msc-da-feb-25-cohort-Russel003
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks (no raw videos needed)

Launch Jupyter from the **project root**:

```bash
jupyter notebook
```

Run in order:
1. `notebooks/NB1_Dataset_and_Feature_Selection_2025004.ipynb`
2. `notebooks/NB2_Model_Results_SHAP_Robustness_2025004.ipynb`

Both notebooks load from `data/features/feature_table_augmented.csv`
and the `logs/` directory. Raw videos are not required.

### 5. Run individual phase scripts

```bash
python -m src.phase4_models
python -m src.phase6a_holdout
python -m src.phase6b_compression_stress
```

### 6. Run the Streamlit demo

```bash
streamlit run app.py
```

The app loads from committed logs and CSV. Raw videos are optional.
If `DEEPFAKE_DATA_ROOT` is not set, the app shows feature-level
evidence and box plots. To enable the video demo:

```bash
# Windows
set DEEPFAKE_DATA_ROOT=E:\path\to\ffpp

# macOS / Linux
export DEEPFAKE_DATA_ROOT=/path/to/ffpp

streamlit run app.py
```

---

## Platform Notes and Known Issues

The pipeline was developed on a Linux VM (Ubuntu 22.04) and tested on
macOS (Apple Silicon, M-series).

### dlib on macOS Apple Silicon (M1 / M2 / M3)

dlib 20.x fails to build on Apple Silicon due to a missing `fp.h` header.
The build error looks like:

    fatal error: 'fp.h' file not found

Fix:

    pip install dlib==19.24.6

Or use the conda-forge pre-compiled binary:

    conda install -c conda-forge dlib -y

Verify:

    python -c "import dlib; print(dlib.__version__)"

### dlib on Windows

Requires CMake and Visual Studio Build Tools:

1. Install CMake from https://cmake.org/download — tick "Add to PATH"
2. Install Visual Studio Build Tools, select "Desktop development with C++"
3. Then run: pip install dlib==19.24.6


### Disk space

Full environment needs ~ 4-5 GB.

### dlib is only needed for ETL

dlib is required only for src/phase2_etl.py (landmark extraction from raw
videos). The notebooks and Streamlit app load from the committed CSV and
do not require dlib at all.



---

## Ethical and Regulatory Position

This pipeline was designed with EU AI Act Article 13 transparency in mind.
Every prediction carries a SHAP attribution identifying which feature drove
the decision and by how much. This supports the transparency requirement
for high-risk AI systems.

Two firm limitations apply to any use of this system:

1. **No demographic evaluation.** FaceForensics++ has no demographic labels.
   Fairness across skin tones, age groups, or genders was not and cannot be
   assessed with this dataset. This is a limitation of the benchmark, not a
   design choice. Any deployment would require demographic auditing first.

2. **Single benchmark only.** All results are internal to FaceForensics++ c40.
   Cross-dataset generalisation to social media footage or other synthesis
   methods is unverified.

---

## Experimental Work

An earlier phase of this project explored distributed feature extraction
using Apache Spark (PySpark) on a Linux VM (Ubuntu 22.04, Hadoop 3.3).
The Spark pipeline extracted facial landmarks at scale across the full
FaceForensics++ corpus before the final CPU-only design was adopted.

This work informed two methodological decisions:
- Frame-level feature aggregation was validated at scale before
  the video-level tabular approach was finalised
- The ETL phase runtime (~8 minutes on CPU) was benchmarked against
  the Spark baseline to confirm CPU-only was viable for 1,000 videos

The Spark scripts are retained in `src/` as part of the research record.
They require a configured Hadoop/Spark environment and are not part of
the reproducible pipeline.


---

## Honest Limitations

- Single dataset (FaceForensics++ c40). No cross-dataset validation.
- AUC 0.6586 is above chance on a hard benchmark. It is not sufficient for
  real-world deployment as a sole piece of evidence.
- FaceSwap AUC drops to 0.6162 because structural geometry changes suppress
  the discriminative signal available to eye-aperture features.
- Gaussian noise is a proxy for codec degradation, not a direct simulation
  of social media compression pipelines.
- No GPU hardware was used. Deep learning benchmarks (~0.90+ AUC) are not
  a fair comparison for a CPU-only 3-feature model on c40.
- Tested on Linux (Ubuntu 22.04) and macOS Apple Silicon. Windows
  reproducibility requires manual dlib build setup (see Platform Notes).

---

## Citation

If you reference this work, please cite:

> Shah, R. P. (2026). *Interpretable CPU-Only Deepfake Detection Using
> Engineered Video-Level Features on FaceForensics++ c40.*
> MSc Data Analytics Dissertation, CCT College Dublin.

---

## License

For academic use only. Raw FaceForensics++ data is subject to the
[FaceForensics++ licence terms](https://github.com/ondyari/FaceForensics).
