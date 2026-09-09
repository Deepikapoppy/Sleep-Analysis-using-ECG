"""
=============================================================================
phase3_hrv_features/p3_pipeline.py
Master orchestrator for Phase 3.

Execution order
───────────────
1  p3_load_phase2      → load Phase 2 RR results, Phase 1 meta, SQI, ECG epochs
2  p3_sqi_gate         → merge SQI columns, flag bad epochs
3  p3_tier1_features   → 18 T1 Primary Screamer features (all RR-derived)
4  p3_tier2_features   → 19 T2 Confirmatory features (RR + first EDR)
5  p3_tier3_features   → 29 T3 Resolver features (full morph + CRC)
6  p3_postprocess      → spectral smoothing, VLF→NaN, derived, z-scores
7  p3_sqi_gate         → null spectral features on bad-SQI epochs
8  p3_plots            → 5 diagnostic plots (SQI + tier1 + tier2 + tier3 + all)
9  p3_save             → CSV + meta JSON + SQLite phase3_done=1

Complexity note
───────────────
71 features sounds heavy, but the pipeline is O(N_epochs) — each epoch is
processed independently. The expensive T3 features (MSE, CRC, morphology)
are computed per-epoch but take ~5–15 ms each on 30 s / 125 Hz data.
For 360 epochs (3-hr recording): ~60–90 seconds total on a modern CPU.
=============================================================================
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase3_hrv_features.p3_logging       import setup_logger, log_phase_header, log_step
from phase3_hrv_features.p3_load_phase2   import (
    load_phase2_results, load_phase1_meta, load_phase1_sqi, load_raw_ecg_epochs
)
from phase3_hrv_features.p3_sqi_gate      import (
    apply_sqi_gate, null_spectral_on_bad_sqi, SPECTRAL_FEATURE_COLS
)
from phase3_hrv_features.p3_tier1_features import compute_tier1
from phase3_hrv_features.p3_tier2_features import compute_tier2
from phase3_hrv_features.p3_tier3_features import compute_tier3
from phase3_hrv_features.p3_postprocess    import (
    apply_spectral_rolling_median, null_vlf,
    compute_derived_features, add_zscore_features
)
from phase3_hrv_features.p3_plots          import (
    plot_sqi_gate, plot_tier1, plot_tier2, plot_tier3, plot_all_features
)
from phase3_hrv_features.p3_save           import save_phase3_results
from Database.db_manager                   import mark_phase_failed, get_record, DB_PATH


def _build_feature_matrix(results: list,
                            ecg_epochs,
                            fs: float,
                            n_epochs: int,
                            epoch_sec: float,
                            logger: logging.Logger) -> pd.DataFrame:
    """
    Per-epoch feature extraction loop: SQI + T1 + T2 + T3.
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
    Run full Phase 3 pipeline for config['record_name'].

    Returns
    -------
    metrics : dict with epoch counts per tier
    """
    rec_name = config["record_name"]
    dataset  = config.get("dataset", "slpdb")

    db_row    = get_record(rec_name, dataset, db_path)
    record_id = db_row["record_id"] if db_row else None

    try:
        # ── Step 1: Load inputs ───────────────────────────────────────────────
        log_step(logger, 1, "Load Phase 2 results + Phase 1 meta")
        results     = load_phase2_results(config, logger, db_path)
        phase1_meta = load_phase1_meta(config, logger)
        sqi_df      = load_phase1_sqi(config, logger)
        ecg_epochs  = load_raw_ecg_epochs(config, logger)

        fs        = float(phase1_meta.get("fs", 125))
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
        log_step(logger, 7, "Save outputs + update SQLite")
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
    from phase0_dataset_management.p0_config import CONFIG, make_subject_dirs
    from Database.db_manager import get_all_records

    logger  = setup_logger("phase3", log_dir="logs", log_file="phase3.log")
    records = get_all_records(CONFIG["dataset"])

    for row in records:
        CONFIG["record_name"] = row["record_name"]
        CONFIG["output_dir"]  = make_subject_dirs(
            row["record_name"],
            CONFIG.get("results_dir", "results"),
            CONFIG["dataset"]
        )
        try:
            run_phase3_record(CONFIG, logger)
        except Exception as exc:
            logger.error(f"Skipping {row['record_name']}: {exc}")
            continue
