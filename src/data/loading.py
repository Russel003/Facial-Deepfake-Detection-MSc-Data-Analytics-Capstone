"""
src/data/loading.py

Data loading utilities for the deepfake detection thesis pipeline.
All paths resolve exclusively through src/utils/config.py.
No absolute paths anywhere in this file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import (
    ALL_FEATURES,
    FEATURE_TABLE_CSV,
    FEATURE_TABLE_PARQUET,
    ID_COL,
    LABEL_COL,
    LOGS_DIR,
    POSITIVE_CLASS,
)

logger = logging.getLogger(__name__)

# Phase 4 log path resolves from LOGS_DIR in config.
# This constant is intentionally module-level so callers can override it
# in tests without patching the config.
PHASE4_LOG_PATH: Path = LOGS_DIR / "phase4_models_log.json"


def load_feature_table(from_csv: bool = True) -> pd.DataFrame:
    """
    Load the augmented feature table.

    from_csv=True  — uses the committed CSV under data/features.
                     Works on any machine without the VM data directory.
    from_csv=False — reads the parquet file at FEATURE_TABLE_PARQUET.
                     Requires DEEPFAKE_DATA_ROOT to be set (VM only).
    """
    if from_csv:
        if not FEATURE_TABLE_CSV.exists():
            raise FileNotFoundError(
                f"Feature CSV not found: {FEATURE_TABLE_CSV}\n"
                "Ensure the repository is cloned correctly. "
                "The CSV must be committed under data/features/."
            )
        df = pd.read_csv(FEATURE_TABLE_CSV)
        logger.info("Loaded feature table from CSV: %s  shape=%s", FEATURE_TABLE_CSV, df.shape)
    else:
        if not FEATURE_TABLE_PARQUET.exists():
            raise FileNotFoundError(
                f"Parquet not found: {FEATURE_TABLE_PARQUET}\n"
                "Set the DEEPFAKE_DATA_ROOT env var to the external data directory, "
                "or use from_csv=True to run from the committed CSV."
            )
        import pyarrow.parquet as pq
        df = pq.read_table(FEATURE_TABLE_PARQUET).to_pandas()
        logger.info(
            "Loaded feature table from Parquet: %s  shape=%s",
            FEATURE_TABLE_PARQUET,
            df.shape,
        )

    required = {ID_COL, "manipulation_type", LABEL_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Feature table is missing required columns: {missing}\n"
            f"Columns present: {sorted(df.columns.tolist())}"
        )

    return df


def load_phase4_indices(
    log_path: Path = PHASE4_LOG_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read train/test positional row indices from the Phase 4 log.

    These are integer row indices into the feature table as it existed
    when Phase 4 was run. Row order must not change between runs.
    Use _verify_row_order() to confirm this assumption before indexing.
    """
    if not log_path.exists():
        raise FileNotFoundError(
            f"Phase 4 log not found: {log_path}\n"
            "Copy logs/phase4_models_log.json from the VM into the repo logs/ directory, "
            "or regenerate Phase 4 to create a fresh log."
        )

    with open(log_path, "r") as f:
        payload = json.load(f)

    train_idx = payload.get("train_idx")
    test_idx  = payload.get("test_idx")

    if train_idx is None or test_idx is None:
        raise ValueError(
            "train_idx or test_idx missing from Phase 4 log.\n"
            f"Keys found in log: {sorted(payload.keys())}"
        )

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def _verify_row_order(df: pd.DataFrame, log_path: Path = PHASE4_LOG_PATH) -> None:
    """
    Confirm that the video_id order in df matches the order recorded in
    the Phase 4 log under the 'video_ids' key.

    Protects against silent corruption when the CSV is re-exported or re-sorted
    after Phase 4 indices were saved. If the log has no 'video_ids' key,
    the check degrades to a warning rather than a hard failure so existing
    logs without this key do not break the pipeline.
    """
    with open(log_path, "r") as f:
        payload = json.load(f)

    recorded_ids: list | None = payload.get("video_ids")

    if recorded_ids is None:
        logger.warning(
            "Phase 4 log does not contain 'video_ids'. "
            "Row order cannot be verified. "
            "If the CSV has been re-sorted or re-exported since Phase 4 ran, "
            "the train/test split is invalid. "
            "Add 'video_ids' to the log or re-run Phase 4 to resolve this."
        )
        return

    current_ids = df[ID_COL].tolist()
    if current_ids != recorded_ids:
        mismatches = sum(a != b for a, b in zip(current_ids, recorded_ids))
        raise ValueError(
            f"Row order mismatch: {mismatches} video_id positions differ between "
            "the current feature table and the Phase 4 log.\n"
            "The CSV was likely re-sorted or re-exported after Phase 4 ran.\n"
            "Re-run Phase 4 with the current feature table to restore a valid split."
        )

    logger.info("Row order verified against Phase 4 log. %d videos match.", len(current_ids))


def build_phase4_split(
    feature_columns: Optional[list[str]] = None,
    from_csv: bool = True,
    verify_order: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X_train, y_train, X_test, y_test) using the Phase 4 split.

    Parameters
    ----------
    feature_columns : list[str] or None
        Columns to use as predictors. Defaults to ALL_FEATURES from config.
        Pass VALIDATED_FEATURES for Model A or SHAP_TOP8_FEATURES for Model B.
    from_csv : bool
        Load from committed CSV (True) or VM parquet (False).
    verify_order : bool
        Verify video_id row order against the Phase 4 log before indexing.
        Set False only when the log is known to lack 'video_ids'.

    Returns
    -------
    X_train, y_train, X_test, y_test as numpy float64 / int arrays.
    y is binary: 1 = POSITIVE_CLASS (fake), 0 = real.
    """
    df = load_feature_table(from_csv=from_csv)
    train_idx, test_idx = load_phase4_indices()

    if verify_order:
        _verify_row_order(df)

    if feature_columns is None:
        feature_columns = ALL_FEATURES

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Requested feature columns not in table: {missing_cols}\n"
            f"Available numeric columns: {sorted(df.columns.tolist())}"
        )

    n_rows = len(df)
    for name, arr in [("train_idx", train_idx), ("test_idx", test_idx)]:
        if len(arr) == 0:
            raise ValueError(f"{name} is empty. Phase 4 log may be corrupt.")
        if arr.max() >= n_rows:
            raise IndexError(
                f"{name} contains out-of-bounds index {arr.max()} "
                f"but feature table has only {n_rows} rows.\n"
                "The saved split was recorded against a different feature table."
            )

    X = df[feature_columns].to_numpy(dtype=float)
    y = (df[LABEL_COL].values == POSITIVE_CLASS).astype(int)

    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        logger.warning(
            "%d NaN value(s) detected in feature matrix before split. "
            "Impute missing values or investigate the ETL output before training.",
            nan_count,
        )

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    logger.info(
        "Phase 4 split loaded  train=%d  test=%d  features=%d  "
        "train_fake_rate=%.3f  test_fake_rate=%.3f",
        len(train_idx),
        len(test_idx),
        len(feature_columns),
        y_train.mean(),
        y_test.mean(),
    )

    return X_train, y_train, X_test, y_test
