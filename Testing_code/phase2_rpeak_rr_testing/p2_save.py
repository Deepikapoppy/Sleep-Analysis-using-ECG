"""
=============================================================================
phase2_rpeak_rr_testing/p2_save.py
Step 6 — Save all Phase-2 outputs to disk + update SQLite (optional).
IDENTICAL logic to the training pipeline — record_name is a device session
id instead of an slpdb/hmc record name; everything else is unchanged.

Files written to <output_dir>/
    phase2_rr_epoch_summary.csv   — per-epoch RR summary
    tachogram_time.npy             — beat timestamps (sec)
    tachogram_rr.npy               — RR intervals (ms)
    r_peak_times_abs.npy           — absolute R-peak sample positions
    phase2_rr_full.json            — full per-epoch results
    phase2_metrics.json            — summary stats + Phase 1 DWT metadata

SQLite update:
    processing_status.phase2_done = 1
    processing_status.phase2_method_used
    processing_status.phase2_mean_hr
    processing_status.phase2_valid_epochs
=============================================================================
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Database.db_manager import get_record, mark_phase_done, mark_phase_failed, DB_PATH


def save_phase2_results(results: list,
                         tachogram_t: np.ndarray,
                         tachogram_rr: np.ndarray,
                         method_counts: dict,
                         config: dict,
                         logger: logging.Logger,
                         meta: Optional[dict] = None,
                         db_path: str = DB_PATH) -> dict:
    """
    Persist all Phase-2 artefacts and update SQLite.
    Returns metrics dict.
    """
    out = config["output_dir"]
    os.makedirs(out, exist_ok=True)

    # ── CSV summary ────────────────────────────────────────────────────────────
    rows = [{k: v for k, v in r.items() if k not in ("rr_ms",)}
            for r in results]
    pd.DataFrame(rows).to_csv(
        os.path.join(out, "phase2_rr_epoch_summary.csv"), index=False
    )

    # ── numpy arrays ──────────────────────────────────────────────────────────
    np.save(os.path.join(out, "tachogram_time.npy"), tachogram_t)
    np.save(os.path.join(out, "tachogram_rr.npy"),   tachogram_rr)

    all_rp_abs = []
    for r in results:
        all_rp_abs.extend(r.get("r_peak_samples_abs", []))
    np.save(os.path.join(out, "r_peak_times_abs.npy"),
            np.array(all_rp_abs, dtype=np.int64))
    logger.info(f"Saved {len(all_rp_abs)} absolute R-peak samples")

    # ── full JSON ─────────────────────────────────────────────────────────────
    with open(os.path.join(out, "phase2_rr_full.json"), "w") as f:
        json.dump(results, f)

    # ── metrics JSON ──────────────────────────────────────────────────────────
    valid = [r for r in results if not r["is_bad"] and r["n_rr"] > 0]

    # Which method won the most epochs
    top_method = (max(method_counts, key=method_counts.get)
                  if method_counts else "unknown")
    mean_hr = (round(float(np.nanmean([r["mean_hr"] for r in valid])), 2)
               if valid else float("nan"))

    metrics = {
        "record"              : config.get("record_name"),
        "dataset"             : config.get("dataset", "device"),
        "total_epochs"        : len(results),
        "valid_epochs"        : len(valid),
        "mean_hr_overall"     : mean_hr,
        "top_method"          : top_method,
        "detection_methods"   : method_counts,
        "mean_quality"        : round(float(
                                    np.nanmean([r["quality"] for r in results])
                                ), 3),
        # Phase 1 DWT metadata for full traceability
        "phase1_preprocessing"      : (meta or {}).get("preprocessing",  "DWT"),
        "phase1_dwt_wavelet"        : (meta or {}).get("dwt_wavelet",    "db4"),
        "phase1_dwt_level"          : (meta or {}).get("dwt_level",       5),
        "phase1_dwt_approx_zeroed"  : (meta or {}).get("dwt_approx_zeroed", True),
        "phase1_dwt_pli_zeroed"     : (meta or {}).get("dwt_pli_zeroed",    True),
        "phase1_dwt_soft_thresh"    : (meta or {}).get("dwt_soft_thresh",   True),
        "phase1_polarity_inverted"  : (meta or {}).get("polarity_inverted", False),
    }
    with open(os.path.join(out, "phase2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Phase 2 metrics: top_method={top_method}, "
                f"mean_hr={mean_hr:.1f}, valid_epochs={len(valid)}")

    # ── SQLite update ─────────────────────────────────────────────────────────
    try:
        db_row = get_record(config["record_name"],
                            config.get("dataset", "device"), db_path)
        if db_row:
            mark_phase_done(db_row["record_id"], 2, out, db_path)
            logger.info("SQLite: phase2_done=1")
    except Exception as exc:
        logger.warning(f"SQLite update failed: {exc}")

    return metrics
