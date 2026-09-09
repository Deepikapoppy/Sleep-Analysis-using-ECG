"""
=============================================================================
phase6_comparison_evaluation/p6_pipeline.py
Master orchestrator for Phase 6 — Hypnogram Comparison & Evaluation
(ECG-Derived Hypnogram vs PSG Gold Standard).

Execution order
───────────────
1  p6_load_inputs  → phase5_meta (DWT + v2 provenance), ECG hypnogram
                      (Phase 4), PSG hypnogram (Phase 5), HRV features
                      (Phase 3); align all to a common epoch count
2  p6_metrics      → accuracy, Cohen's kappa, per-stage P/R/F1,
                      confusion matrix
3  p6_plots_core   → hypnogram comparison, confusion matrix, per-stage
                      metrics bar, duration comparison, final summary
   p6_plots_hrv    → HRV-by-PSG-stage, SQI-agreement, v2 feature
                      comparison, detection-method accuracy
4  p6_save         → metrics JSON/CSV/report + phase6_meta.json
                      + combined report PNG + SQLite phase6_done=1

Prerequisites
─────────────
Phase 3 output (phase3_hrv_features.csv) must exist.
Phase 4 output (ecg_hypnogram_smooth.npy / ecg_hypnogram_raw.npy) must exist.
Phase 5 output (psg_hypnogram.npy, phase5_meta.json) must exist.

Phase 6 is the FINAL phase of the pipeline.
=============================================================================
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase6_comparison_evaluation.p6_logging      import setup_logger, log_phase_header, log_step
from phase6_comparison_evaluation.p6_load_inputs  import (
    load_pipeline_metadata, load_all, align,
)
from phase6_comparison_evaluation.p6_metrics      import (
    compute_metrics, _kappa_interpretation,
)
from phase6_comparison_evaluation.p6_plots_core   import (
    plot_hypnogram_comparison, plot_confusion_matrix, plot_metrics_bar,
    plot_stage_duration_comparison, plot_final_summary,
)
from phase6_comparison_evaluation.p6_plots_hrv    import (
    plot_hrv_by_psg_stage, plot_sqi_agreement_analysis,
    plot_v2_feature_comparison, plot_detection_method_accuracy,
)
from phase6_comparison_evaluation.p6_save         import (
    save_metrics, save_phase6_meta, combine_hrv_and_summary,
    update_sqlite, _get_overall_dir,
)
from Database.db_manager                          import mark_phase_failed, get_record, DB_PATH


def run_phase6_record(config: dict,
                       logger: logging.Logger,
                       db_path: str = DB_PATH) -> dict:
    """
    Run the full Phase 6 pipeline for config['record_name'].

    Returns
    -------
    dict with keys: subject, overall_acc, cohen_kappa,
                    kappa_interpretation, n_epochs
    """
    rec_name  = config["record_name"]
    dataset   = config.get("dataset", "slpdb")
    db_row    = get_record(rec_name, dataset, db_path)
    record_id = db_row["record_id"] if db_row else None

    try:
        # ── Step 1: Load inputs ───────────────────────────────────────────────
        log_step(logger, 1, "Load pipeline metadata + ECG/PSG hypnograms + HRV features")
        pipeline_meta         = load_pipeline_metadata(config, logger)
        ecg_hyp, psg_hyp, hrv = load_all(config, logger)
        ecg_hyp, psg_hyp, hrv = align(ecg_hyp, psg_hyp, hrv, logger)

        log_phase_header(logger, rec_name, dataset, len(psg_hyp))

        # ── Step 2: Compute metrics ───────────────────────────────────────────
        log_step(logger, 2, "Compute evaluation metrics (accuracy, kappa, per-stage P/R/F1)")
        metrics, cm, y_true, y_pred, stages = compute_metrics(ecg_hyp, psg_hyp, logger)

        # ── Step 3: Plots ─────────────────────────────────────────────────────
        log_step(logger, 3, "Generate diagnostic plots (9)")
        plot_hypnogram_comparison(ecg_hyp, psg_hyp, config, logger)
        plot_confusion_matrix(cm, stages, config, logger)
        plot_metrics_bar(metrics, config, logger)
        plot_hrv_by_psg_stage(hrv, psg_hyp, config, logger)
        plot_stage_duration_comparison(ecg_hyp, psg_hyp, config, logger)
        plot_final_summary(metrics, ecg_hyp, psg_hyp, config, logger,
                           pipeline_meta=pipeline_meta)
        plot_sqi_agreement_analysis(ecg_hyp, psg_hyp, config, logger)
        plot_v2_feature_comparison(hrv, ecg_hyp, psg_hyp, config, logger)
        plot_detection_method_accuracy(ecg_hyp, psg_hyp, config, logger)

        # ── Step 4: Save ──────────────────────────────────────────────────────
        log_step(logger, 4, "Save metrics + meta + combined report + SQLite")
        save_metrics(metrics, cm, stages, config, logger)
        save_phase6_meta(metrics, config, logger, pipeline_meta=pipeline_meta)
        combine_hrv_and_summary(config, logger)
        update_sqlite(metrics, config, logger, db_path=db_path)

        logger.info(f"✓ Phase 6 complete — {rec_name} | epochs={metrics['n_epochs']}")
        return {
            "subject": rec_name,
            **{k: metrics[k] for k in
               ("overall_acc", "cohen_kappa", "kappa_interpretation", "n_epochs")},
        }

    except Exception as exc:
        logger.error(f"✗ Phase 6 FAILED — {rec_name}: {exc}")
        if record_id:
            mark_phase_failed(record_id, 6, str(exc), db_path)
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-subject summary  (cross-subject accuracy / kappa bar charts)
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_subject_summary(all_metrics: list, config: dict,
                                logger: logging.Logger) -> None:
    """
    Cross-subject summary bar chart of accuracy and kappa.
    Called after processing all subjects; writes to results/<dataset>/plots/overall/.
    """
    if len(all_metrics) < 2:
        return

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = [m["subject"]      for m in all_metrics]
    accs     = [m["overall_acc"]  for m in all_metrics]
    kappas   = [m["cohen_kappa"]  for m in all_metrics]

    overall_dir = _get_overall_dir(config)
    x = np.arange(len(subjects))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(subjects) * 1.2 + 4), 6))
    fig.suptitle("Cross-Subject Evaluation Summary — All Records",
                 fontsize=14, fontweight='bold')

    axes[0].bar(x, accs, width=w * 2, color='#3498db', edgecolor='black', alpha=0.85)
    axes[0].axhline(np.mean(accs), color='navy', linestyle='--', linewidth=1.5,
                    label=f'Mean={np.mean(accs):.3f}')
    for xi, v in zip(x, accs):
        axes[0].text(xi, v + 0.005, f'{v:.2f}', ha='center', fontsize=8)
    axes[0].set_xticks(x);  axes[0].set_xticklabels(subjects, rotation=30, ha='right')
    axes[0].set_ylim([0, 1.1]);  axes[0].set_ylabel("Overall Accuracy")
    axes[0].set_title("Overall Accuracy per Subject");  axes[0].legend(fontsize=8)
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].bar(x, kappas, width=w * 2, color='#9b59b6', edgecolor='black', alpha=0.85)
    axes[1].axhline(np.mean(kappas), color='purple', linestyle='--', linewidth=1.5,
                    label=f'Mean κ={np.mean(kappas):.3f}')
    for xi, v in zip(x, kappas):
        interp = _kappa_interpretation(v)
        axes[1].text(xi, v + 0.005, f'{v:.2f}\n({interp[:3]})',
                     ha='center', fontsize=7)
    axes[1].set_xticks(x);  axes[1].set_xticklabels(subjects, rotation=30, ha='right')
    axes[1].set_ylim([-0.1, 1.1]);  axes[1].set_ylabel("Cohen's Kappa (κ)")
    axes[1].set_title("Cohen's Kappa per Subject");  axes[1].legend(fontsize=8)
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(overall_dir, "phase6_multi_subject_summary.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Multi-subject summary saved -> {out}")

    # Save CSV
    df_ms = pd.DataFrame(all_metrics)
    df_ms.to_csv(os.path.join(overall_dir, "phase6_multi_subject_metrics.csv"), index=False)
    logger.info(f"Multi-subject metrics CSV saved -> {overall_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from phase0_dataset_management.p0_config import CONFIG, make_subject_dirs
    from Database.db_manager import get_all_records

    logger      = setup_logger("phase6", log_dir="logs", log_file="phase6.log")
    records     = get_all_records(CONFIG["dataset"])
    all_metrics = []

    for row in records:
        CONFIG["record_name"] = row["record_name"]
        CONFIG["output_dir"]  = make_subject_dirs(
            row["record_name"],
            CONFIG.get("results_dir", "results"),
            CONFIG["dataset"],
        )
        try:
            result = run_phase6_record(CONFIG, logger)
            all_metrics.append(result)
        except Exception as exc:
            logger.error(f"Skipping {row['record_name']}: {exc}")
            continue

    if all_metrics:
        logger.info("Generating cross-subject summary...")
        plot_multi_subject_summary(all_metrics, CONFIG, logger)

    logger.info("Phase 6 — all subjects done.")
    logger.info("ALL 6 PHASES COMPLETE — Results in results/")
