"""
phase2_rpeak_rr_testing/
══════════════════════════════════════════════════════════════════════
DEVICE-ONLY testing package. slpdb/hmc handling intentionally removed —
production code for those datasets lives in phase2_rpeak_rr/. Requires
phase1_preprocessing_testing to have already produced Phase 1 outputs for
the session (preprocessed_epochs.npy, bad_epoch_mask.npy, phase1_meta.json,
etc.) under config["output_dir"].

FILE                    STEP   RESPONSIBILITY
──────────────────────────────────────────────────────────────────────
p2_logging.py           —      setup_logger, log_phase_header, log_step
p2_load_phase1.py       1      load Phase-1 .npy + .json outputs from disk
p2_rpeak_detection.py   2      calibrated cascade: neurokit → elgendi2010
                                → scipy_prominence safety net
p2_rr_intervals.py      3      RR computation + 2-tier artifact correction
p2_process_epochs.py    4      loop all epochs → per-epoch result rows
p2_plots.py             5      7 diagnostic plots incl. method comparison
p2_save.py              6      .csv / .npy / .json + SQLite phase2_done=1
                                (optional — db_row is None handled gracefully)
p2_pipeline.py          ALL    run_phase2_record()  ← MASTER ENTRY POINT
p2_consolidated_csv.py  —      build_consolidated_csv() across all sessions
phase2_done.py          —      one-off reset utility (isolated test DB only)
══════════════════════════════════════════════════════════════════════

3 R-peak methods (calibrated cascade, unchanged from training):
    neurokit             — NeuroKit2 default (gradient + threshold)
    elgendi2010          — Two-moving-average (best on noisy sleep ECG)
    scipy_prominence     — No external dep, always available (safety net)

Data flow from Phase 1 (phase1_preprocessing_testing):
    preprocessed_epochs.npy  → loaded by p2_load_phase1
    bad_epoch_mask.npy        → loaded by p2_load_phase1
    clean_ecg_full.npy        → loaded for comparison plots
    raw_ecg_ds_full.npy       → loaded for comparison plots
    phase1_meta.json          → fs, wavelet, polarity propagated to phase2
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .p2_logging       import setup_logger, log_phase_header, log_step
from .p2_load_phase1   import (
    load_phase1_outputs, load_raw_ecg_for_viz, load_clean_ecg_for_viz
)
from .p2_rpeak_detection import detect_rpeaks_all_methods, calibrate_subject
from .p2_rr_intervals    import (
    compute_rr, correct_rr_artifacts,
    apply_patient_rr_filter, build_tachogram
)
from .p2_process_epochs  import process_all_epochs
from .p2_save            import save_phase2_results
from .p2_pipeline        import run_phase2_record
from .p2_consolidated_csv import build_consolidated_csv

__all__ = [
    "setup_logger", "log_phase_header", "log_step",
    "load_phase1_outputs", "load_raw_ecg_for_viz", "load_clean_ecg_for_viz",
    "detect_rpeaks_all_methods", "calibrate_subject",
    "compute_rr", "correct_rr_artifacts",
    "apply_patient_rr_filter", "build_tachogram",
    "process_all_epochs",
    "save_phase2_results",
    "run_phase2_record",
    "build_consolidated_csv",
]
