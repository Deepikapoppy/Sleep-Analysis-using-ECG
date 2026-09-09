"""
=============================================================================
phase4b_ml_classification_testing/p4b_predict.py
Core logic: load the pretrained RF bundle, align a test session's Phase 3
features to the model's exact training columns, impute, predict.

No training happens here. No ground truth is used or expected — this is
the true deployment path: a device session was never part of the pooled
training set, so there's nothing to score predictions against.
=============================================================================
"""

import logging
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


# ─────────────────────────────────────────────────────────────────────────────
#  Load the pretrained bundle
# ─────────────────────────────────────────────────────────────────────────────

def load_model_bundle(model_path: str, logger: logging.Logger) -> Dict:
    """
    Load {"model", "imputer", "feature_cols"} exactly as saved by
    p4b_ml_classify.py's save_outputs(). Raises FileNotFoundError with a
    clear message if the .pkl doesn't exist — this is a hard prerequisite,
    not something to degrade gracefully around.
    """
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            f"Run phase4b_ml_classification/p4b_ml_classify.py (training) "
            f"first, or check the model_path you passed in — it must point "
            f"at a phase4b_rf_model.pkl under results/<training_dataset>/"
            f"consolidated/, where <training_dataset> is 'hmc' or 'slpdb' "
            f"(whichever the model was trained on) — never 'device', since "
            f"device sessions are never part of the pooled training set."
        )
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    for key in ("model", "imputer", "feature_cols"):
        if key not in bundle:
            raise ValueError(
                f"Model bundle at {model_path} is missing key '{key}' — "
                f"not a valid phase4b_rf_model.pkl (expected "
                f"{{'model','imputer','feature_cols'}})."
            )

    logger.info(
        f"Loaded model bundle: {model_path}  "
        f"({len(bundle['feature_cols'])} feature columns, "
        f"model={type(bundle['model']).__name__})"
    )
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
#  Align a test session's features to the model's training columns
# ─────────────────────────────────────────────────────────────────────────────

def align_features(df: pd.DataFrame,
                   feat_cols: List[str],
                   logger: logging.Logger) -> Tuple[np.ndarray, Dict]:
    """
    Reindex df to EXACTLY feat_cols, in that order — the single step
    people get wrong most often when deploying a trained sklearn model.

    Columns present in feat_cols but absent from this session's Phase 3
    output (e.g. a feature that silently failed to compute for a very
    short/sparse session) become all-NaN columns here — the model's own
    imputer will median-fill them using the TRAINING distribution, exactly
    as it would for a missing value it saw during training.

    Returns
    -------
    X          : np.ndarray, shape (n_epochs, len(feat_cols)) — inf already
                 converted to NaN, ready for imputer.transform()
    align_info : dict — diagnostics about how well this session's columns
                 matched the model's expected columns (for the meta JSON)
    """
    missing_cols = [c for c in feat_cols if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in feat_cols
                    and c not in ("epoch_idx", "is_bad", "n_rr",
                                  "detection_method", "sqi_flag")]

    if missing_cols:
        logger.warning(
            f"{len(missing_cols)}/{len(feat_cols)} model feature(s) not "
            f"present in this session's Phase 3 output — will be "
            f"median-filled from the TRAINING distribution: "
            f"{missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}"
        )
    if extra_cols:
        logger.info(
            f"{len(extra_cols)} column(s) in this session's features are "
            f"not used by the model (dropped, not an error): "
            f"{extra_cols[:5]}{'...' if len(extra_cols) > 5 else ''}"
        )

    X_df = df.reindex(columns=feat_cols)
    X    = X_df.values.astype(float)
    X[np.isinf(X)] = np.nan

    n_nan_before = int(np.isnan(X).sum())
    align_info = {
        "n_feat_cols_expected": len(feat_cols),
        "n_missing_cols"      : len(missing_cols),
        "missing_cols"        : missing_cols,
        "n_extra_cols_dropped": len(extra_cols),
        "n_nan_cells_pre_impute": n_nan_before,
        "pct_nan_cells_pre_impute": round(
            100.0 * n_nan_before / max(X.size, 1), 2
        ),
    }
    logger.info(
        f"Feature alignment: {X.shape[0]} epochs x {X.shape[1]} columns, "
        f"{align_info['pct_nan_cells_pre_impute']}% NaN before imputation"
    )
    return X, align_info


# ─────────────────────────────────────────────────────────────────────────────
#  Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict_session(X: np.ndarray,
                    bundle: Dict,
                    logger: logging.Logger) -> np.ndarray:
    """
    Impute (using the model's fitted TRAINING imputer, not a fresh one —
    this session's own NaN pattern must never influence its own fill
    values) and predict.

    Returns
    -------
    preds : np.ndarray of int, shape (n_epochs,) — stage codes 0-4
    """
    imputer = bundle["imputer"]
    model   = bundle["model"]

    X_imputed = imputer.transform(X)
    preds     = model.predict(X_imputed)

    from collections import Counter
    counts = Counter(preds)
    logger.info(
        "Predicted stage distribution: "
        + str({STAGE_NAMES.get(k, str(k)): int(v)
              for k, v in sorted(counts.items())})
    )
    return preds.astype(int)
