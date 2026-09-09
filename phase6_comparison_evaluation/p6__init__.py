"""
phase6_comparison_evaluation
=============================
Compare the ECG-derived hypnogram (Phase 4) against the PSG gold-standard
hypnogram (Phase 5): compute accuracy / Cohen's kappa / per-stage P,R,F1,
generate 9 diagnostic plots, and save evaluation results.

This is the FINAL phase of the pipeline (renamed from the old "Phase 7").

Module layout
─────────────
  p6_logging          logger setup (mirrors Phase 3/4/5 style)
  p6_load_inputs      load_pipeline_metadata, apply_final_smoothing,
                      load_all, align
  p6_metrics          _kappa_interpretation, compute_metrics
  p6_plots_core       4 core comparison plots
                        1 — Hypnogram comparison + agreement timeline
                        2 — Confusion matrix (raw + normalised)
                        3 — Per-stage Precision/Recall/F1 bar chart
                        4 — Stage duration comparison (PSG vs ECG)
                        + Final results summary dashboard
  p6_plots_hrv        4 HRV / signal-quality diagnostic plots
                        5 — HRV features by PSG stage (core + v2)
                        6 — SQI vs epoch-agreement analysis
                        7 — v2 feature comparison (PSG vs ECG stage)
                        8 — R-peak detection method vs accuracy
  p6_save             save_metrics, save_phase6_meta,
                      combine_hrv_and_summary, update_sqlite
                      (phase6_done=1), _get_overall_dir
  p6_pipeline         run_phase6_record, plot_multi_subject_summary,
                      __main__ entry point

Prerequisites
─────────────
Phase 3 complete — phase3_hrv_features.csv must exist.
Phase 4 complete — ecg_hypnogram_smooth.npy / ecg_hypnogram_raw.npy must exist.
Phase 5 complete — psg_hypnogram.npy and phase5_meta.json must exist.
"""

from phase6_comparison_evaluation.p6_pipeline import run_phase6_record

__all__ = ["run_phase6_record"]
