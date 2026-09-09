"""
phase2_rpeak_rr/
══════════════════════════════════════════════════════════════════════
FILE                    STEP   RESPONSIBILITY
──────────────────────────────────────────────────────────────────────
p2_logging.py           —      setup_logger, log_phase_header, log_step
p2_load_phase1.py       1      load Phase-1 .npy + .json outputs from disk
p2_rpeak_detection.py   2      5 parallel detectors + best-method selection
p2_rr_intervals.py      3      RR computation + 2-tier artifact correction
p2_process_epochs.py    4      loop all epochs → per-epoch result rows
p2_plots.py             5      7 plots incl. all-5-methods comparison
p2_save.py              6      .csv / .npy / .json + SQLite phase2_done=1
p2_pipeline.py          ALL    run_phase2_record()  ← MASTER ENTRY POINT
══════════════════════════════════════════════════════════════════════

5 R-peak methods (cascade):
    neurokit             — NeuroKit2 default (gradient + threshold)
    elgendi2010          — Two-moving-average (best on noisy sleep ECG)
    scipy_prominence     — No external dep, always available

Data flow from Phase 1:
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
from .p2_rpeak_detection import detect_rpeaks_all_methods
from .p2_rr_intervals    import (
    compute_rr, correct_rr_artifacts,
    apply_patient_rr_filter, build_tachogram
)
from .p2_process_epochs  import process_all_epochs
from .p2_save            import save_phase2_results
from .p2_pipeline        import run_phase2_record

__all__ = [
    "setup_logger", "log_phase_header", "log_step",
    "load_phase1_outputs", "load_raw_ecg_for_viz", "load_clean_ecg_for_viz",
    "detect_rpeaks_all_methods",
    "compute_rr", "correct_rr_artifacts",
    "apply_patient_rr_filter", "build_tachogram",
    "process_all_epochs",
    "save_phase2_results",
    "run_phase2_record",
]
