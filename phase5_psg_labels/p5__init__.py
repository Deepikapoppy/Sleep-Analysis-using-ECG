"""
phase5_psg_labels
=================
Load PSG gold-standard sleep stage labels from SLPDB (.st) and HMC (text)
annotation files, align them to ECG epoch indices, and save the aligned
hypnogram for Phase 6 evaluation.

Module layout
─────────────
  p5_logging          logger setup (mirrors Phase 3/4 style)
  p5_load_inputs      load_phase1_meta, load_phase3_meta, load_phase1_sqi,
                      load_psg_annotations, get_original_fs
  p5_build_hypnogram  map_stage, build_psg_hypnogram
                      (FIX-3 annotation offset + pandas-safe ffill/bfill)
  p5_plots            5 diagnostic plots
                        1 — PSG hypnogram
                        2 — Stage distribution bar + pie
                        3 — Raw annotation timeline
                        4 — PSG hypnogram + Phase 1 SQI overlay
                        5 — HRV feature boxplots by PSG stage
  p5_save             save_psg, save_phase5_meta,
                      collect_duration_row, write_duration_summary,
                      update_sqlite  (phase5_done=1)
  p5_pipeline         run_phase5_record + __main__ entry point

Prerequisites
─────────────
Phase 1 complete   — phase1_meta.json must exist (provides n_epochs).
Phase 3 optional   — phase3_hrv_features.csv used for HRV-by-stage plot.
Phase 4 optional   — phase4_ecg_hypnogram.csv used for duration summary.
"""

from phase5_psg_labels.p5_pipeline import run_phase5_record

__all__ = ["run_phase5_record"]
