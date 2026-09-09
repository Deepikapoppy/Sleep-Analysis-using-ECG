"""
phase4b_ml_classification_testing/  —  RF Sleep Stage PREDICTION, DEVICE-ONLY
══════════════════════════════════════════════════════════════════════════
This is the true deployment path — loads your PRETRAINED
RandomForestClassifier bundle (produced by
phase4b_ml_classification/p4b_ml_classify.py during training on hmc/slpdb)
and predicts sleep stages for a device test session directly from Phase 3
HRV features. No training happens here, no ground truth is used or
expected.

NOTE: phase4_sleep_classification_testing (the rule-based scorer testing
package) is NOT part of this path — it's an independent, optional baseline
you can run separately if you want to eyeball rule-based vs. RF predictions
on the same session, but Phase 4b never calls into it.

Module layout
─────────────
  p4b_logging.py   — logger setup
  p4b_predict.py   — load_model_bundle, align_features, predict_session
  p4b_pipeline.py  — master orchestrator — run_phase4b_predict_record()

Prerequisites
─────────────
1. Phase 3 (phase3_hrv_features_testing) complete for the session
   (phase3_hrv_features.csv must exist in output_dir).
2. A trained phase4b_rf_model.pkl (from running training's
   p4b_ml_classify.py on your labeled hmc/slpdb data) — you must pass its
   path explicitly, it is never derived from config["dataset"]="device".
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .p4b_logging import setup_logger, log_phase_header, log_step
from .p4b_predict  import (
    load_model_bundle, align_features, predict_session, STAGE_NAMES
)
from .p4b_pipeline import run_phase4b_predict_record

__all__ = [
    "setup_logger", "log_phase_header", "log_step",
    "load_model_bundle", "align_features", "predict_session", "STAGE_NAMES",
    "run_phase4b_predict_record",
]
