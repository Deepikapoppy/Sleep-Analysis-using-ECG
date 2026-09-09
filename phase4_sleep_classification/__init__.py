"""
phase4_sleep_classification
===========================
Rule-Based Sleep Stage Classification from ECG-derived HRV features.

Module layout
─────────────
  p4_logging      — logger setup (mirrors Phase 3 style)
  p4_load_phase3  — load Phase 3 CSV + meta JSON
  p4_preprocess   — clip_outliers, normalise_features (robust 5–95 pct)
  p4_scoring      — score_wake/n1/n2/n3/rem + autonomic-storm helper
  p4_classify     — classify_epochs, smooth_hypnogram
  p4_plots        — 5 diagnostic plots (scores, hypnogram, duration, v2)
  p4_save         — persist npy / CSV / JSON, update SQLite phase4_done=1
  p4_pipeline     — master orchestrator (run_phase4_record + __main__)

Prerequisites
─────────────
Phase 3 (p3_pipeline.py) must be complete so that
phase3_hrv_features.csv and phase3_meta.json exist in output_dir.
"""

from phase4_sleep_classification.p4_pipeline import run_phase4_record

__all__ = ["run_phase4_record"]
