"""
=============================================================================
phase3_hrv_features_testing/p3_pipeline.py
Master orchestrator for Phase 3, DEVICE sessions only.
slpdb/hmc handling has been intentionally REMOVED from this testing
package — production code for those datasets lives in
phase3_hrv_features/p3_pipeline.py. config["record_name"] here is always a
device session id (e.g. "ADM937394258") and config["dataset"] is always
"device". Requires Phase 2 (phase2_rpeak_rr_testing) to have already run
successfully for the session — reads phase2_rr_full.json, phase1_meta.json,
phase1_sqi.csv, and preprocessed_epochs.npy from config["output_dir"].

Execution order (unchanged step order/logic from training)
──────────────────────────────────────────────────────────
1  p3_load_phase2      → load Phase 2 RR results, Phase 1 meta, SQI, ECG epochs
2  p3_sqi_gate         → merge SQI columns, flag bad epochs
3  p3_tier1_features   → 18 T1 Primary Screamer features (all RR-derived)
4  p3_tier2_features   → 19 T2 Confirmatory features (RR + first EDR)
5  p3_tier3_features   → 29 T3 Resolver features (full morph + CRC)
6  p3_postprocess      → spectral smoothing, VLF→NaN, derived, z-scores
7  p3_sqi_gate         → null spectral features on bad-SQI epochs
8  p3_plots            → 5 diagnostic plots (SQI + tier1 + tier2 + tier3 + all)
9  p3_save             → CSV + meta JSON + SQLite phase3_done=1 (optional)
=============================================================================
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .p3_logging       import setup_logger, log_phase_header, log_step
from .p3_load_phase2   import (
    load_phase2_results, load_phase1_meta, load_phase1_sqi, load_raw_ecg_epochs
)
from .p3_sqi_gate      import (
    apply_sqi_gate, null_spectral_on_bad_sqi, SPECTRAL_FEATURE_COLS
)
from .p3_tier1_features import compute_tier1
from .p3_tier2_features import compute_tier2
from .p3_tier3_features import compute_tier3
from .p3_postprocess    import (
    apply_spectral_rolling_median, null_vlf,
    compute_derived_features, add_zscore_features
)
from .p3_plots          import (
    plot_sqi_gate, plot_tier1, plot_tier2, plot_tier3, plot_all_features
)
from .p3_save           import save_phase3_results
from Database.db_manager import mark_phase_failed, get_record, DB_PATH


def _build_feature_matrix(results: list,
                            ecg_epochs,
                            fs: float,
                            n_epochs: int,
                            epoch_sec: float,
                            logger: logging.Logger) -> pd.DataFrame:
    """
    Per-epoch feature extraction loop: SQI + T1 + T2 + T3.
    IDENTICAL logic to the training pipeline — fully signal-agnostic.
    """
    rows = []
    logger.info(f"Extracting 71 features for {len(results)} epochs...")

    for i, row in enumerate(results):
        rr_ms  = row.get("rr_ms", [])
        n_rr   = int(row.get("n_rr", len(rr_ms)))
        r_peaks_abs = row.get("r_peak_samples_abs", [])

        # Per-epoch ECG array (if available)
        ecg_ep = None
        r_peaks_local = []
        if ecg_epochs is not None and i < len(ecg_epochs):
            ecg_ep = ecg_epochs[i]
            # Convert absolute indices to epoch-local indices
            epoch_offset = i * int(fs * epoch_sec)
            r_peaks_local = [int(p) - epoch_offset for p in r_peaks_abs
                              if 0 <= int(p) - epoch_offset < len(ecg_ep)]

        # Base row
        feat = {
            "epoch_idx"       : row["epoch_idx"],
            "is_bad"          : row["is_bad"],
            "n_rr"            : n_rr,
            "detection_method": row.get("method", "unknown"),
        }

        # ── T1 Primary Screamer ───────────────────────────────────────────────
        t1 = compute_tier1(rr_ms) if n_rr >= 4 else {}
        feat.update(t1)

        # ── T2 Confirmatory ───────────────────────────────────────────────────
        t2 = compute_tier2(
            rr_ms,
            t1_feats  = t1,
            ecg_epoch = ecg_ep,
            r_peaks   = r_peaks_local,
            fs        = fs,
            epoch_idx = row["epoch_idx"],
            n_epochs  = n_epochs,
            epoch_sec = epoch_sec,
        ) if n_rr >= 4 else {}
        feat.update(t2)

        # ── T3 Resolver ───────────────────────────────────────────────────────
        t3 = compute_tier3(
            rr_ms,
            ecg_epoch = ecg_ep,
            r_peaks   = r_peaks_local,
            fs        = fs,
        ) if n_rr >= 4 else {}
        feat.update(t3)

        rows.append(feat)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(results)} epochs...")

    df = pd.DataFrame(rows)
    logger.info(f"Raw feature matrix shape: {df.shape}")
    return df


def run_phase3_record(config: dict,
                       logger: logging.Logger,
                       db_path: str = DB_PATH) -> dict:
    """
    Run full Phase 3 pipeline for config['record_name'] (a device session id).

    Returns
    -------
    metrics : dict with epoch counts per tier
    """
    rec_name = config["record_name"]
    dataset  = config.get("dataset", "device")

    db_row    = get_record(rec_name, dataset, db_path)   # None if not registered — fine
    record_id = db_row["record_id"] if db_row else None

    try:
        # ── Step 1: Load inputs ───────────────────────────────────────────────
        log_step(logger, 1, "Load Phase 2 results + Phase 1 meta")
        results     = load_phase2_results(config, logger, db_path)
        phase1_meta = load_phase1_meta(config, logger)
        sqi_df      = load_phase1_sqi(config, logger)
        ecg_epochs  = load_raw_ecg_epochs(config, logger)

        fs        = float(phase1_meta.get("fs", config.get("target_fs", 125)))
        epoch_sec = float(config.get("epoch_sec", 30.0))
        n_epochs  = len(results)

        log_phase_header(logger, rec_name, dataset, n_epochs)

        # ── Step 2: Build feature matrix (T1 + T2 + T3) ──────────────────────
        log_step(logger, 2, "Extract T1 (18) + T2 (19) + T3 (29) features")
        df = _build_feature_matrix(
            results, ecg_epochs, fs, n_epochs, epoch_sec, logger
        )

        # ── Step 3: Apply SQI gate ────────────────────────────────────────────
        log_step(logger, 3, "Apply SQI Gate")
        df = apply_sqi_gate(df, sqi_df, logger)

        # ── Step 4: Post-processing ───────────────────────────────────────────
        log_step(logger, 4, "Post-processing (spectral smooth, VLF, derived, z-scores)")
        df = apply_spectral_rolling_median(df, logger)
        df = null_vlf(df, logger)
        df = compute_derived_features(df, logger)
        df = add_zscore_features(df, logger)

        # ── Step 5: Null spectral on bad-SQI epochs ───────────────────────────
        log_step(logger, 5, "Null spectral features on bad-SQI epochs")
        df = null_spectral_on_bad_sqi(df, SPECTRAL_FEATURE_COLS, logger)

        # ── Step 6: Plots ─────────────────────────────────────────────────────
        log_step(logger, 6, "Generate diagnostic plots")
        plot_sqi_gate(df, config, logger)
        plot_tier1(df, config, logger)
        plot_tier2(df, config, logger)
        plot_tier3(df, config, logger)
        plot_all_features(df, config, logger)

        # ── Step 7: Save ──────────────────────────────────────────────────────
        log_step(logger, 7, "Save outputs + update SQLite (optional)")
        metrics = save_phase3_results(df, config, logger,
                                       phase1_meta=phase1_meta,
                                       db_path=db_path)

        logger.info(
            f"✓ Phase 3 complete — {rec_name} | "
            f"epochs={metrics['n_epochs']} | "
            f"T1={metrics['n_t1_cols']} | T2={metrics['n_t2_cols']} | "
            f"T3={metrics['n_t3_cols']} | z-scores={metrics['n_zscore_cols']}"
        )
        return metrics

    except Exception as exc:
        logger.error(f"✗ Phase 3 FAILED — {rec_name}: {exc}")
        if record_id:
            mark_phase_failed(record_id, 3, str(exc), db_path)
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from phase0_dataset_management_testing import CONFIG, scan_datasets, make_subject_dirs
    from Database.db_manager import get_all_records

    TEST_DB_PATH = "test_pipeline.db"   # isolated from your production DB

    logger = setup_logger("phase3_test", log_dir="logs", log_file="phase3_test.log")

    # Prefer sessions already registered in the isolated test DB; fall back
    # to a fresh disk scan if nothing's registered yet — same fallback
    # pattern as phase1/phase2 testing packages.
    db_records = get_all_records(CONFIG["dataset"], db_path=TEST_DB_PATH)
    session_ids = ([row["record_name"] for row in db_records] if db_records
                   else scan_datasets(CONFIG, logger))

    for session_id in session_ids:
        CONFIG["record_name"] = session_id
        CONFIG["output_dir"]  = make_subject_dirs(
            session_id, CONFIG["results_dir"], CONFIG["dataset"]
        )
        try:
            run_phase3_record(CONFIG, logger, db_path=TEST_DB_PATH)
        except Exception as exc:
            # e.g. a session whose Phase 2 never completed — skip, don't halt.
            logger.error(f"Skipping {session_id}: {exc}")
            continue
