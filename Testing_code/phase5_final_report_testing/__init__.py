"""
phase5_final_report_testing/  —  Final Report from Pretrained-Model Predictions
════════════════════════════════════════════════════════════════════════════
Combines training's Phase 5 (PSG hypnogram reporting) + Phase 6 (evaluation)
into ONE deployment-side phase, minus everything that needs PSG ground
truth (accuracy, kappa, confusion matrix, per-stage F1) — device sessions
never carry PSG labels, so that comparison is impossible by definition,
not merely skipped.

Instead, this reports on the Phase 4b RF model's PREDICTED hypnogram:
hypnogram plot, stage distribution, SQI overlay, HRV-by-stage boxplots,
and a final summary dashboard.

Module layout
─────────────
  p5_logging.py      — logger setup
  p5_load_inputs.py  — load Phase 1 meta/SQI, Phase 3 features,
                       Phase 4b predictions
  p5_plots.py        — 5 plots, all built on the predicted stage
  p5_save.py         — duration summary CSV, phase5_meta.json, SQLite update
  p5_pipeline.py     — master orchestrator — run_phase5_record()

Prerequisites
─────────────
Phase 1 testing and Phase 4b testing must be complete for the session.
Phase 3 testing is optional (HRV summary plot skipped gracefully if absent).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .p5_logging     import setup_logger, log_phase_header, log_step
from .p5_load_inputs import load_all
from .p5_plots        import (
    STAGE_NAMES, STAGE_COLORS,
    plot_predicted_hypnogram, plot_stage_distribution,
    plot_sqi_overlay, plot_hrv_summary, plot_final_summary,
)
from .p5_save          import save_duration_summary, save_phase5_meta, update_sqlite
from .p5_pipeline      import run_phase5_record

__all__ = [
    "setup_logger", "log_phase_header", "log_step",
    "load_all",
    "STAGE_NAMES", "STAGE_COLORS",
    "plot_predicted_hypnogram", "plot_stage_distribution",
    "plot_sqi_overlay", "plot_hrv_summary", "plot_final_summary",
    "save_duration_summary", "save_phase5_meta", "update_sqlite",
    "run_phase5_record",
]
