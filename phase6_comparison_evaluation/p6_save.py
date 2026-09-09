"""
=============================================================================
phase6_comparison_evaluation/p6_save.py
Step 4 — Persist Phase 6 outputs and update SQLite.

Functions
─────────
  save_metrics            phase6_metrics.json + phase6_confusion_matrix.csv
                          + phase6_final_report.txt (printed to stdout too)
  save_phase6_meta        phase6_meta.json — evaluation results + Phase 1
                          DWT / Phase 3 v2 provenance, for downstream use
  combine_hrv_and_summary stacks phase3_all_hrv_features.png +
                          phase6_final_summary.png into combined_report.png
  update_sqlite           processing_status.phase6_done = 1 + evaluation
                          summary columns
  _get_overall_dir        results/<dataset>/plots/overall/ helper
=============================================================================
"""

import os
import sys
import json
import logging
import sqlite3

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Database.db_manager import get_record, mark_phase_done, DB_PATH
from phase4_sleep_classification.p4_plots import STAGE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_overall_dir(config: dict) -> str:
    overall = os.path.join(
        config.get("results_dir", "results"),
        config.get("dataset", "slpdb"),
        "plots", "overall",
    )
    os.makedirs(overall, exist_ok=True)
    return overall


# ─────────────────────────────────────────────────────────────────────────────
#  Save metrics JSON / confusion matrix CSV / final report TXT
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, cm, present_stages, config: dict,
                 logger: logging.Logger) -> None:
    out = config["output_dir"]

    def _to_serializable(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)):  return o.tolist()
        try:
            import pandas as _pd
            if isinstance(o, _pd.Timestamp): return o.isoformat()
        except Exception:
            pass
        raise TypeError(f"Object of type {o.__class__.__name__} not serialisable")

    metrics["present_stages"] = [int(s) for s in metrics.get("present_stages", [])]

    with open(os.path.join(out, "phase6_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=_to_serializable)
    logger.info("Metrics JSON saved.")

    stage_labels = [STAGE_NAMES.get(s, str(s)) for s in present_stages]
    df_cm        = pd.DataFrame(cm, index=stage_labels, columns=stage_labels)
    df_cm.index.name = "True \\ Pred"
    cm_path = os.path.join(out, "phase6_confusion_matrix.csv")
    df_cm.to_csv(cm_path)
    logger.info(f"Confusion matrix CSV -> {cm_path}")

    report_lines = [
        "=" * 60,
        "SLEEP ECG ANALYSIS — FINAL EVALUATION REPORT",
        f"Record : {config['record_name']}",
        "=" * 60,
        f"Overall Accuracy : {metrics['overall_acc']*100:.2f}%",
        f"Cohen's Kappa    : {metrics['cohen_kappa']:.4f}"
        f"  ({metrics.get('kappa_interpretation', '')})",
        f"Valid Epochs     : {metrics['n_epochs']}",
        "",
        "Per-Stage Metrics:",
        f"{'Stage':10s}  {'Precision':10s}  {'Recall':10s}  {'F1':10s}  {'Support':8s}",
        "-" * 55,
    ]
    for name, v in metrics["per_stage"].items():
        report_lines.append(
            f"{name:10s}  {v['precision']:10.4f}  {v['recall']:10.4f}"
            f"  {v['f1']:10.4f}  {v['support']:8d}"
        )
    report_lines += [
        "",
        "Confusion Matrix (rows=True PSG, cols=Predicted ECG):",
        df_cm.to_string(),
        "=" * 60,
    ]
    report_path = os.path.join(out, "phase6_final_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Final report -> {report_path}")
    print("\n" + "\n".join(report_lines))


# ─────────────────────────────────────────────────────────────────────────────
#  Save Phase 6 metadata
# ─────────────────────────────────────────────────────────────────────────────

def save_phase6_meta(metrics: dict, config: dict, logger: logging.Logger,
                     pipeline_meta: dict = None) -> dict:
    """Write phase6_meta.json bundling evaluation results + DWT provenance."""
    meta6 = {
        "overall_acc"          : metrics["overall_acc"],
        "cohen_kappa"          : metrics["cohen_kappa"],
        "kappa_interpretation" : metrics.get("kappa_interpretation", ""),
        "n_epochs"             : metrics["n_epochs"],
        "per_stage_f1"         : {s: v["f1"] for s, v in metrics["per_stage"].items()},
        # Upstream DWT provenance
        "phase1_dwt_wavelet"   : (pipeline_meta or {}).get("phase1_dwt_wavelet",   "db4"),
        "phase1_dwt_level"     : (pipeline_meta or {}).get("phase1_dwt_level",      5),
        "phase1_mean_sqi"      : (pipeline_meta or {}).get("phase1_mean_sqi",       None),
        "phase3_n_features"    : (pipeline_meta or {}).get("phase3_n_features",     None),
    }
    out = os.path.join(config["output_dir"], "phase6_meta.json")
    with open(out, "w") as f:
        json.dump(meta6, f, indent=2)
    logger.info(f"Phase 6 meta saved -> {out}")
    return meta6


# ─────────────────────────────────────────────────────────────────────────────
#  Combined HRV + final summary report PNG
# ─────────────────────────────────────────────────────────────────────────────

def combine_hrv_and_summary(config: dict, logger: logging.Logger) -> None:
    from PIL import Image, ImageDraw, ImageFont

    hrv_path     = os.path.join(config["output_dir"], "phase3_all_hrv_features.png")
    summary_path = os.path.join(config["output_dir"], "phase6_final_summary.png")

    if not os.path.exists(hrv_path):
        logger.warning(f"combine_hrv_and_summary: missing {hrv_path} — skipping.")
        return
    if not os.path.exists(summary_path):
        logger.warning(f"combine_hrv_and_summary: missing {summary_path} — skipping.")
        return

    hrv_img     = Image.open(hrv_path).convert("RGB")
    summary_img = Image.open(summary_path).convert("RGB")

    TARGET_W  = hrv_img.width
    scale     = TARGET_W / summary_img.width
    summary_r = summary_img.resize(
        (TARGET_W, int(summary_img.height * scale)), Image.LANCZOS
    )

    LABEL_H = 60
    GAP     = 16
    total_h = LABEL_H + hrv_img.height + GAP + LABEL_H + summary_r.height

    canvas = Image.new("RGB", (TARGET_W, total_h), color=(255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36
        )
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([0, 0, TARGET_W, LABEL_H], fill=(220, 235, 255))
    draw.text((TARGET_W // 2, LABEL_H // 2),
              f"All HRV Features — {config['record_name']}",
              font=font, fill=(20, 40, 120), anchor="mm")
    canvas.paste(hrv_img, (0, LABEL_H))

    y2 = LABEL_H + hrv_img.height + GAP
    draw.rectangle([0, y2, TARGET_W, y2 + LABEL_H], fill=(220, 255, 230))
    draw.text((TARGET_W // 2, y2 + LABEL_H // 2),
              f"Sleep ECG Analysis — Final Results Summary — {config['record_name']}",
              font=font, fill=(20, 90, 30), anchor="mm")
    canvas.paste(summary_r, (0, y2 + LABEL_H))

    out = os.path.join(config["output_dir"], "combined_report.png")
    canvas.save(out, dpi=(150, 150))
    logger.info(f"Combined report saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  SQLite update
# ─────────────────────────────────────────────────────────────────────────────

def update_sqlite(metrics: dict, config: dict, logger: logging.Logger,
                  db_path: str = DB_PATH) -> None:
    """Mark phase6_done = 1 and write evaluation summary columns."""
    try:
        db_row = get_record(config["record_name"],
                            config.get("dataset", "slpdb"), db_path)
        if db_row:
            mark_phase_done(db_row["record_id"], 6, config["output_dir"], db_path)
            per_stage_f1 = json.dumps(
                {s: v["f1"] for s, v in metrics["per_stage"].items()}
            )
            conn = sqlite3.connect(db_path)
            conn.execute("""
                UPDATE processing_status
                   SET phase6_overall_acc          = ?,
                       phase6_cohen_kappa          = ?,
                       phase6_kappa_interpretation = ?,
                       phase6_n_epochs             = ?,
                       phase6_per_stage_f1         = ?
                 WHERE record_id = ?
            """, (
                metrics["overall_acc"],
                metrics["cohen_kappa"],
                metrics.get("kappa_interpretation", ""),
                metrics["n_epochs"],
                per_stage_f1,
                db_row["record_id"],
            ))
            conn.commit()
            conn.close()
            logger.info("SQLite: phase6_done=1, phase6 evaluation summary columns written.")
    except Exception as exc:
        logger.warning(f"SQLite update failed: {exc}")
