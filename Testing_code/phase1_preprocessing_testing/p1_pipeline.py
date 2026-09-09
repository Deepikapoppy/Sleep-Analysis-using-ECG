"""
=============================================================================
phase1_preprocessing_testing/p1_pipeline.py
Orchestrates all Phase-1 steps for one DEVICE session.
slpdb/hmc handling has been intentionally REMOVED from this testing
package — production code for those datasets lives in
phase1_preprocessing/p1_pipeline.py. config["record_name"] here is always
a device session id (e.g. "ADM937394258"), and config["dataset"] is always
"device" (as set by phase0_dataset_management_testing.p0_config.CONFIG).

Marks phase1_done in SQLite on success/failure IF the session was registered
via phase0_dataset_management_testing.p0_register_records — this is
OPTIONAL for one-off test recordings; db_row is None is handled gracefully
(record_id stays None, DB writes are skipped) exactly like Phase 0 testing
does. Always pass an explicit db_path (e.g. "test_pipeline.db") to keep test
runs isolated from your production DB.

Full pipeline (unchanged step order/logic from training)
──────────────────────────────────────────────────────
Step 1  p1_load_ecg             → load + concatenate device session chunks
Step 2  p1_downsample           → resample to target_fs (default 125 Hz)
Step 3  p1_dwt_denoise          → DWT full pipeline:
          Step 3a  p1_baseline_wander    → zero cA_level
          Step 3b  p1_pli_removal        → zero cD1
          Step 3c  p1_soft_threshold     → soft-threshold D2–D5 + IDWT
Step 4  p1_polarity_correction  → global polarity fix (BUG-5)
Step 5  p1_zscore_normalisation → z-score normalise
Step 6  p1_segmentation         → 30-sec epochs
Step 7  p1_sqi                  → SQI + bad epoch detection
Step 8  p1_plots                → 6 diagnostic plots
Step 9  p1_save                 → .npy + .csv + .json
=============================================================================
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .p1_logging            import setup_logger, log_phase_header, log_step
from .p1_load_ecg           import load_ecg, extract_recording_start_time
from .p1_downsample         import downsample
from .p1_dwt_denoise        import dwt_filter
from .p1_polarity_correction import detect_and_apply_global_polarity
from .p1_zscore_normalisation import normalize
from .p1_segmentation       import segment_epochs
from .p1_sqi                import compute_sqi_all_epochs, detect_bad_epochs
from .p1_plots               import (
    plot_preprocessing_stages,
    plot_dwt_coefficients,
    plot_raw_vs_dwt_cleaned,
    plot_sqi_overview,
    plot_epoch_quality,
    plot_spectral_comparison,
)
from .p1_save                import save_preprocessed

from Database.db_manager import get_record, mark_phase_done, mark_phase_failed, DB_PATH


def run_phase1_record(config: dict,
                      logger: logging.Logger,
                      db_path: str = DB_PATH) -> dict:
    """
    Run full Phase-1 pipeline for config['record_name'] (a device session id).

    Returns
    -------
    meta : dict — same content as phase1_meta.json
    """
    rec_name  = config["record_name"]
    dataset   = config.get("dataset", "device")
    wavelet   = config.get("dwt_wavelet", "db4")
    level     = config.get("dwt_level",   5)

    db_row    = get_record(rec_name, dataset, db_path)   # None if not registered — fine
    record_id = db_row["record_id"] if db_row else None

    try:
        log_phase_header(logger, rec_name, dataset,
                         fs_orig=0, fs_target=config["target_fs"],
                         wavelet=wavelet, level=level)

        # ── Step 1: Load + concatenate device session chunks ────────────────
        log_step(logger, 1, "Load raw ECG (device session)")
        ecg_raw, fs_orig, record = load_ecg(config, logger)
        rec_start = extract_recording_start_time(config, record, logger)

        # ── Step 2: Downsample ────────────────────────────────────────────────
        log_step(logger, 2, f"Downsample to {config['target_fs']} Hz")
        ecg_ds, fs = downsample(ecg_raw, fs_orig, config["target_fs"], logger)

        # ── Step 3: DWT Denoising (3a + 3b + 3c inside dwt_filter) ───────────
        log_step(logger, 3, "DWT Denoising (Baseline + PLI + Soft-Threshold)")
        ecg_dwt, coeffs_raw, coeffs_proc = dwt_filter(
            ecg_ds, fs, wavelet=wavelet, level=level, logger=logger
        )

        # ── Step 4: Global Polarity Correction ────────────────────────────────
        log_step(logger, 4, "Global Polarity Correction")
        ecg_dwt, polarity_inverted = detect_and_apply_global_polarity(
            ecg_dwt, logger
        )

        # ── Step 5: Z-score Normalisation ────────────────────────────────────
        log_step(logger, 5, "Z-score Normalisation")
        ecg_norm = normalize(ecg_dwt, logger=logger)

        # ── Step 6: Epoch Segmentation ────────────────────────────────────────
        log_step(logger, 6, "Epoch Segmentation (30 sec)")
        epochs, _spe = segment_epochs(
            ecg_norm, fs, config["epoch_sec"], logger
        )

        # Device sessions can be shorter than one epoch (e.g. a single
        # ~29 s chunk) — segment_epochs() correctly returns 0 epochs in
        # that case, but detect_bad_epochs/compute_sqi_all_epochs/
        # save_preprocessed all divide by len(epochs) unconditionally and
        # would otherwise crash with a bare ZeroDivisionError further down.
        # Fail fast here with an actionable message instead.
        if len(epochs) == 0:
            n_samples = len(ecg_norm)
            duration_s = n_samples / fs
            raise ValueError(
                f"Session '{rec_name}' produced 0 epochs after segmentation: "
                f"{n_samples} samples @ {fs} Hz = {duration_s:.1f}s, which is "
                f"shorter than one {config['epoch_sec']}s epoch. This session "
                f"has too few chunks to score — combine with more chunks for "
                f"this admissionId, or skip it."
            )

        # ── Step 7: SQI + Bad Epoch Detection ────────────────────────────────
        log_step(logger, 7, "SQI Computation + Bad Epoch Detection")
        bad_mask = detect_bad_epochs(epochs, logger=logger)
        sqi_df   = compute_sqi_all_epochs(epochs, fs, logger)

        # ── Step 8: Plots ─────────────────────────────────────────────────────
        log_step(logger, 8, "Generating Diagnostic Plots")
        stages = [
            (f"1. Raw (downsampled to {config['target_fs']} Hz)", ecg_ds),
            (f"2. After DWT Denoising ({wavelet}, L={level})",    ecg_dwt),
            ("3. Z-score Normalised",                             ecg_norm),
        ]
        plot_preprocessing_stages(stages, fs, config, logger)
        plot_dwt_coefficients(ecg_ds, coeffs_raw, coeffs_proc,
                              fs, wavelet, level, config, logger)
        plot_raw_vs_dwt_cleaned(ecg_ds, ecg_norm, fs, config, logger)
        plot_sqi_overview(sqi_df, bad_mask, config, logger)
        plot_epoch_quality(epochs, bad_mask, config, logger)
        plot_spectral_comparison(ecg_ds, ecg_norm, fs, config, logger,
                                 wavelet=wavelet, level=level)

        # ── Step 9: Save ──────────────────────────────────────────────────────
        log_step(logger, 9, "Save Outputs")
        meta = save_preprocessed(
            epochs, bad_mask, ecg_norm, ecg_ds, sqi_df, fs, config, logger,
            polarity_inverted    = polarity_inverted,
            dwt_wavelet          = wavelet,
            dwt_level            = level,
            recording_start_time = rec_start,
        )

        logger.info(
            f"✓ Phase 1 complete — {rec_name} | "
            f"{meta['n_epochs']} epochs | "
            f"{meta['n_bad_epochs']} bad | "
            f"mean SQI={meta['mean_sqi']} | "
            f"DWT={wavelet} L={level}"
        )

        if record_id:
            mark_phase_done(record_id, 1, config["output_dir"], db_path)

        return meta

    except Exception as exc:
        logger.error(f"✗ Phase 1 FAILED — {rec_name}: {exc}")
        if record_id:
            mark_phase_failed(record_id, 1, str(exc), db_path)
        raise


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from phase0_dataset_management_testing import CONFIG, scan_datasets, make_subject_dirs
    from Database.db_manager import get_all_records

    logger = setup_logger("phase1_test", log_dir="logs", log_file="phase1_test.log")

    # Prefer sessions already registered in the (isolated) test DB; fall back
    # to a fresh disk scan if nothing's registered yet, so this still works
    # standalone without requiring Phase 0 registration to have run first.
    records = get_all_records(CONFIG["dataset"], db_path="test_pipeline.db")
    if records:
        session_ids = [row["record_name"] for row in records]
    else:
        session_ids = scan_datasets(CONFIG, logger)

    for session_id in session_ids:
        CONFIG["record_name"] = session_id
        CONFIG["output_dir"]  = make_subject_dirs(
            session_id, CONFIG["results_dir"], CONFIG["dataset"]
        )
        run_phase1_record(CONFIG, logger, db_path="test_pipeline.db")
