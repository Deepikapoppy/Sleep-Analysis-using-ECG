"""
phase3_hrv_features_testing/  —  Phase 3 HRV Feature Extraction, DEVICE-ONLY
══════════════════════════════════════════════════════════════════════════
slpdb/hmc handling intentionally removed — production code for those
datasets lives in phase3_hrv_features/. Requires phase2_rpeak_rr_testing to
have already produced phase2_rr_full.json for the session under
config["output_dir"].

Module layout
─────────────
  p3_logging.py          Logging setup
  p3_load_phase2.py       Load Phase 2 RR results + Phase 1 meta/SQI
  p3_sqi_gate.py          SQI Gate (4 features) — always applied first
  p3_tier1_features.py    T1 Primary Screamer (18 features)
  p3_tier2_features.py    T2 Confirmatory (19 features)
  p3_tier3_features.py    T3 Resolver (29 features)
  p3_postprocess.py       Derived features, rolling median smooth, z-scores
  p3_plots.py             Diagnostic plots per tier
  p3_save.py              Save CSV, meta JSON, SQLite update (optional)
  p3_pipeline.py          Master orchestrator — run_phase3_record()
══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .p3_logging       import setup_logger, log_phase_header, log_step
from .p3_load_phase2   import (
    load_phase2_results, load_phase1_meta, load_phase1_sqi, load_raw_ecg_epochs
)
from .p3_sqi_gate      import (
    apply_sqi_gate, null_spectral_on_bad_sqi,
    SQI_GATE_COLS, SPECTRAL_FEATURE_COLS
)
from .p3_tier1_features import compute_tier1
from .p3_tier2_features import compute_tier2
from .p3_tier3_features import compute_tier3
from .p3_postprocess    import (
    apply_spectral_rolling_median, null_vlf,
    compute_derived_features, add_zscore_features
)
from .p3_save           import save_phase3_results
from .p3_pipeline       import run_phase3_record

__all__ = [
    "setup_logger", "log_phase_header", "log_step",
    "load_phase2_results", "load_phase1_meta", "load_phase1_sqi",
    "load_raw_ecg_epochs",
    "apply_sqi_gate", "null_spectral_on_bad_sqi",
    "SQI_GATE_COLS", "SPECTRAL_FEATURE_COLS",
    "compute_tier1", "compute_tier2", "compute_tier3",
    "apply_spectral_rolling_median", "null_vlf",
    "compute_derived_features", "add_zscore_features",
    "save_phase3_results",
    "run_phase3_record",
]
