"""
=============================================================================
phase4b_ml_classification/p4b_predict.py
Inference-only: load the pre-trained RF model (from p4b_ml_classify.py) and
predict a hypnogram for ONE subject, WITHOUT retraining.

Use this for:
  - a genuinely new/held-out subject (no PSG needed to run this step)
  - re-scoring a subject after re-running Phase 3 with updated features

Prerequisites
─────────────
  - phase4b_rf_model.pkl must exist (produced once by p4b_ml_classify.py)
  - Phase 3 must be complete for this subject (phase3_hrv_features.csv)
  - PSG (psg_hypnogram.npy) is OPTIONAL here — only needed if you also
    want accuracy/kappa printed for this subject; not needed to just
    produce a hypnogram.
=============================================================================
"""

import os
import sys
import json
import pickle
import logging

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_sleep_classification.p4_load_phase3 import load_features
from phase4_sleep_classification.p4_classify import smooth_hypnogram
from phase0_dataset_management.p0_config import make_subject_dirs

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def load_pretrained_model(dataset: str, config: dict):
    consolidated_dir = os.path.join(config.get("results_dir", "results"),
                                     dataset, "consolidated")
    model_path = os.path.join(consolidated_dir, "phase4b_rf_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model at {model_path} — run p4b_ml_classify.py "
            f"(training) at least once before using this inference script."
        )
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["imputer"], bundle["feature_cols"]


def predict_subject(record_name: str, config: dict, logger: logging.Logger):
    """
    Predict RF hypnogram for one subject using the pre-trained model.
    Does NOT retrain. Returns (predictions, output_dir).
    """
    model, imputer, feat_cols = load_pretrained_model(config["dataset"], config)

    output_dir = make_subject_dirs(
        record_name, config.get("results_dir", "results"), config["dataset"]
    )
    cfg = dict(config)
    cfg["record_name"] = record_name
    cfg["output_dir"]  = output_dir

    df = load_features(cfg, logger)

    # Guard: if any training feature column is missing for this subject
    # (e.g. Phase 3 config changed), fill with NaN so imputer can handle it
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        logger.warning(f"{record_name}: {len(missing)} training features "
                        f"missing from this subject's CSV — filling as NaN: "
                        f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        for c in missing:
            df[c] = np.nan

    X = df[feat_cols].values.astype(float)
    X[np.isinf(X)] = np.nan   # same fix as training: ratio features can
                              # overflow to +/-inf for individual epochs
    X = imputer.transform(X)
    preds_raw = model.predict(X)
    preds = smooth_hypnogram(preds_raw, kernel_size=5, min_run_epochs=2, logger=None)

    np.save(os.path.join(output_dir, "phase4b_ecg_hypnogram_rf_raw.npy"), preds_raw)
    out_path = os.path.join(output_dir, "phase4b_ecg_hypnogram_rf.npy")
    np.save(out_path, preds)
    logger.info(f"{record_name}: predicted {len(preds)} epochs → {out_path}")

    # Optional: if PSG exists for this subject, report accuracy/kappa
    psg_path = os.path.join(output_dir, "psg_hypnogram.npy")
    if os.path.exists(psg_path):
        psg = np.load(psg_path)
        n = min(len(psg), len(preds))
        acc_raw    = accuracy_score(psg[:n], preds_raw[:n])
        kappa_raw  = cohen_kappa_score(psg[:n], preds_raw[:n])
        acc        = accuracy_score(psg[:n], preds[:n])
        kappa      = cohen_kappa_score(psg[:n], preds[:n])
        logger.info(f"{record_name}: raw acc={acc_raw:.3f} kappa={kappa_raw:.3f}  →  "
                    f"smoothed acc={acc:.3f} kappa={kappa:.3f}")
    else:
        logger.info(f"{record_name}: no PSG found — hypnogram saved, "
                     f"no accuracy computed (expected for a truly new subject)")

    return preds, output_dir


if __name__ == "__main__":
    from phase0_dataset_management.p0_config  import CONFIG
    from phase0_dataset_management.p0_logging import setup_logger

    logger = setup_logger("phase4b_predict", log_dir="logs",
                           log_file="phase4b_predict.log")

    # Set this to the subject you want to score with the pre-trained model
    RECORD_NAME = "REPLACE_WITH_SUBJECT_ID"
    predict_subject(RECORD_NAME, CONFIG, logger)