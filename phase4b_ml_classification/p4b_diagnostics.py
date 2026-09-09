"""
=============================================================================
phase4b_ml_classification/p4b_diagnostics.py
Per-subject data-quality diagnostic — run this BEFORE trusting Phase 4b's
accuracy/kappa numbers, especially for subjects with near-zero kappa.

Checks three things per subject, then cross-references against the
phase4b_overall_acc / phase4b_cohen_kappa already saved in SQLite:

  1. Epoch alignment  — does len(phase3_hrv_features.csv) match
                         len(psg_hypnogram.npy) BEFORE truncation?
                         A big mismatch means p4b_ml_classify.py's
                         `n = min(len(df), len(psg))` silently truncated
                         and probably paired the wrong epochs together.
  2. PSG imbalance    — what fraction of this subject's night is a single
                         dominant stage? >85% single-stage makes kappa
                         near 0 almost inevitable even with a good model.
  3. Signal quality   — mean SQI / bad-epoch fraction from Phase 3. Poor
                         quality ECG for this subject → poor features →
                         poor predictions, independent of the model itself.

Output: results/<dataset>/consolidated/phase4b_diagnostics.csv, sorted
worst-kappa-first, plus a printed summary of the flagged subjects.
=============================================================================
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.db_manager import get_all_records, get_connection, DB_PATH
from phase4_sleep_classification.p4_load_phase3 import load_features
from phase0_dataset_management.p0_config import make_subject_dirs

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


# ─────────────────────────────────────────────────────────────────────────────
#  Pull already-saved RF results from SQLite (from your last p4b_ml_classify.py run)
# ─────────────────────────────────────────────────────────────────────────────

def _load_phase4b_results(dataset: str, db_path: str = DB_PATH) -> dict:
    """Returns {record_name: {'acc': ..., 'kappa': ...}} from processing_status."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT r.record_name, ps.phase4b_overall_acc, ps.phase4b_cohen_kappa
               FROM processing_status ps
               JOIN records r ON r.record_id = ps.record_id
               JOIN datasets d ON d.dataset_id = r.dataset_id
               WHERE d.name = ? AND ps.phase4b_done = 1""",
            (dataset,),
        ).fetchall()
    return {r["record_name"]: {"acc": r["phase4b_overall_acc"],
                                "kappa": r["phase4b_cohen_kappa"]}
            for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
#  Per-subject diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_subject(row: dict, config: dict, logger: logging.Logger) -> dict:
    subj = row["record_name"]
    output_dir = make_subject_dirs(
        subj, config.get("results_dir", "results"), config["dataset"]
    )
    cfg = dict(config)
    cfg["record_name"] = subj
    cfg["output_dir"]  = output_dir

    feat_path = os.path.join(output_dir, "phase3_hrv_features.csv")
    psg_path  = os.path.join(output_dir, "psg_hypnogram.npy")

    result = {"subject": subj}

    if not os.path.exists(feat_path):
        result["status"] = "MISSING_PHASE3"
        return result
    if not os.path.exists(psg_path):
        result["status"] = "MISSING_PHASE5"
        return result

    df  = load_features(cfg, logger)
    psg = np.load(psg_path)

    n_feat = len(df)
    n_psg  = len(psg)
    len_diff     = abs(n_feat - n_psg)
    len_diff_pct = 100.0 * len_diff / max(n_feat, n_psg, 1)

    # PSG class balance (on the RAW psg array, before any truncation)
    psg_counts = pd.Series(psg).value_counts(normalize=True)
    dominant_stage      = int(psg_counts.idxmax())
    dominant_stage_frac = float(psg_counts.max())

    # Signal quality from Phase 3, if columns present
    mean_sqi      = float(df["overall_sqi"].mean()) if "overall_sqi" in df.columns else np.nan
    bad_epoch_frac = float(df["is_bad"].mean()) if "is_bad" in df.columns else np.nan

    result.update({
        "status":               "OK",
        "n_feat_epochs":        n_feat,
        "n_psg_epochs":         n_psg,
        "len_diff":             len_diff,
        "len_diff_pct":         round(len_diff_pct, 2),
        "dominant_psg_stage":   STAGE_NAMES.get(dominant_stage, dominant_stage),
        "dominant_psg_frac":    round(dominant_stage_frac, 3),
        "mean_sqi":             round(mean_sqi, 3) if np.isfinite(mean_sqi) else None,
        "bad_epoch_frac":       round(bad_epoch_frac, 3) if np.isfinite(bad_epoch_frac) else None,
    })

    # Flags — the actual "why might this subject be bad" reasons
    flags = []
    if len_diff_pct > 5:
        flags.append(f"EPOCH_MISALIGNMENT ({len_diff} epochs, {len_diff_pct:.1f}%)")
    if dominant_stage_frac > 0.85:
        flags.append(f"SEVERE_PSG_IMBALANCE ({result['dominant_psg_stage']}="
                      f"{dominant_stage_frac*100:.0f}%)")
    if np.isfinite(mean_sqi) and mean_sqi < 0.5:
        flags.append(f"LOW_SQI ({mean_sqi:.2f})")
    if np.isfinite(bad_epoch_frac) and bad_epoch_frac > 0.3:
        flags.append(f"HIGH_BAD_EPOCH_FRAC ({bad_epoch_frac*100:.0f}%)")
    result["flags"] = "; ".join(flags) if flags else "none"

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Run across all subjects, join with SQLite RF results
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(config: dict, logger: logging.Logger,
                     db_path: str = DB_PATH) -> pd.DataFrame:
    records = get_all_records(config["dataset"])
    rf_results = _load_phase4b_results(config["dataset"], db_path)

    rows = []
    for row in records:
        d = diagnose_subject(row, config, logger)
        rf = rf_results.get(row["record_name"], {})
        d["rf_acc"]   = rf.get("acc")
        d["rf_kappa"] = rf.get("kappa")
        rows.append(d)

    df = pd.DataFrame(rows)

    # Worst kappa first (NaN kappa — no RF result — sinks to bottom)
    df = df.sort_values("rf_kappa", ascending=True, na_position="last")

    consolidated_dir = os.path.join(config.get("results_dir", "results"),
                                     config["dataset"], "consolidated")
    os.makedirs(consolidated_dir, exist_ok=True)
    out_path = os.path.join(consolidated_dir, "phase4b_diagnostics.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Diagnostics saved → {out_path}")

    # Print the 15 worst subjects for a quick look
    print("\n" + "=" * 100)
    print("15 WORST-KAPPA SUBJECTS — check the 'flags' column for likely cause")
    print("=" * 100)
    cols = ["subject", "rf_acc", "rf_kappa", "len_diff_pct",
            "dominant_psg_stage", "dominant_psg_frac", "mean_sqi",
            "bad_epoch_frac", "flags"]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df[cols].head(15).to_string(index=False))
    print("=" * 100)

    # Quick correlation check: does ANY flag correlate with low kappa?
    flagged = df[df["flags"] != "none"]
    unflagged = df[df["flags"] == "none"]
    if len(flagged) > 0 and len(unflagged) > 0:
        print(f"\nMean kappa — flagged subjects   ({len(flagged)}): "
              f"{flagged['rf_kappa'].mean():.3f}")
        print(f"Mean kappa — unflagged subjects ({len(unflagged)}): "
              f"{unflagged['rf_kappa'].mean():.3f}")
        print("(if flagged << unflagged, the low-kappa subjects are largely "
              "explained by data issues, not the model itself)\n")

    return df


if __name__ == "__main__":
    from phase0_dataset_management.p0_config  import CONFIG
    from phase0_dataset_management.p0_logging import setup_logger

    logger = setup_logger("phase4b_diagnostics", log_dir="logs",
                           log_file="phase4b_diagnostics.log")
    run_diagnostics(CONFIG, logger)
