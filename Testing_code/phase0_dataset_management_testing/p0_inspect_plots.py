"""
=============================================================================
phase0_dataset_management_testing/p0_inspect_plots.py
DEVICE-ONLY. Load a single session, generate the Phase-0 inspection plot:
  - Raw ECG overview (first 60 s + full-session amplitude envelope)
No PSG stage distribution plot — device format never has ground truth, so
that plot (and slpdb/hmc loading) has been intentionally removed from this
testing package.

Marks phase0_done in SQLite when successful (and record_id is available —
sessions run without DB registration are still fully supported).
=============================================================================
"""

import os
import json
import logging
import numpy as np

from .p0_config  import CONFIG, get_record_path, make_subject_dirs, TEST_DB_PATH
from .p0_logging import setup_logger
from .p0_metadata_extraction import _fs_from_chunk
from .p0_scan_datasets import build_device_manifest, get_device_chunk_files
from Database.db_manager import (
    get_record, mark_phase_done, mark_phase_failed
)


# ─────────────────────────────────────────────────────────────────────────────
#  Load — one device session
# ─────────────────────────────────────────────────────────────────────────────

def load_record(config, logger):
    """
    Load one device session: concatenate all ECG chunk files (sorted by
    window_start_ms, across every hour-folder for this admissionId) into
    one continuous signal. Gaps beyond device_gap_tolerance_ms are
    NaN-padded (not stitched) so Phase 1's SQI / bad-epoch logic can flag
    them downstream. Returns ann=[] since this format never carries
    ground truth.
    """
    session_id = config["record_name"]
    ecg_key    = config.get("device_ecg_key", "ECG_CH_A")
    gap_tol    = config.get("device_gap_tolerance_ms", 500)
    fs_tol_pct = config.get("device_fs_tolerance_pct", 2.0)

    chunks = get_device_chunk_files(config, session_id, logger)
    if not chunks:
        raise FileNotFoundError(
            f"[DEVICE] No chunk files found for session '{session_id}'. "
            f"Check local_test_device_path / device_id_key in CONFIG."
        )

    logger.info(f"[DEVICE] Loading session '{session_id}': {len(chunks)} chunk(s)")

    per_chunk_fs = []
    pieces       = []
    n_gaps       = 0

    for i, c in enumerate(chunks):
        with open(c["path"], "r", encoding="utf-8") as f:
            full = json.load(f)
        # Every registered chunk is a packetized list (see p0_scan_datasets.py
        # module docstring) — a session's chunks were already filtered to
        # that shape by build_device_manifest(), so no other case is expected.
        signal = []
        for record in full:
            if not isinstance(record, dict):
                continue
            value = record.get("value")
            if not isinstance(value, list) or not value:
                continue
            packet = value[0] if isinstance(value[0], list) else value
            if isinstance(packet, list):
                signal.extend(packet)
        seg = np.asarray(signal, dtype="float64")
        fs_c = _fs_from_chunk(c)
        if fs_c:
            per_chunk_fs.append(fs_c)

        if i > 0:
            prev   = chunks[i - 1]
            gap_ms = c["window_start_ms"] - prev["window_end_ms"]
            if gap_ms > gap_tol and per_chunk_fs:
                fs_est    = per_chunk_fs[-1]
                n_missing = int(round((gap_ms / 1000) * fs_est))
                if n_missing > 0:
                    pieces.append(np.full(n_missing, np.nan))
                    n_gaps += 1
                    logger.warning(
                        f"  Gap {gap_ms:.0f} ms between "
                        f"{os.path.basename(prev['path'])} → "
                        f"{os.path.basename(c['path'])} — "
                        f"padded {n_missing} NaN sample(s)"
                    )
        pieces.append(seg)

    signal = np.concatenate(pieces) if pieces else np.array([])
    if not per_chunk_fs:
        raise ValueError(f"[DEVICE] Could not determine fs for session '{session_id}'")

    fs_median = float(np.median(per_chunk_fs))
    outliers  = [f for f in per_chunk_fs
                if abs(f - fs_median) / fs_median * 100 > fs_tol_pct]
    if outliers:
        logger.warning(
            f"  {len(outliers)}/{len(per_chunk_fs)} chunk(s) deviate "
            f">{fs_tol_pct}% from median fs ({fs_median:.2f} Hz)"
        )

    if np.isnan(signal).any():
        signal = np.where(np.isfinite(signal), signal, np.nanmedian(signal))

    class _DeviceRecord:
        pass
    rec          = _DeviceRecord()
    rec.fs       = fs_median
    rec.sig_len  = len(signal)
    rec.sig_name = [ecg_key]
    rec.p_signal = signal.reshape(-1, 1)

    logger.info(
        f"  Session '{session_id}': {len(signal)} samples @ {fs_median:.2f} Hz, "
        f"{n_gaps} gap(s) padded, no ground truth"
    )
    return rec, []          # ann = [] — device format never has PSG labels


# ─────────────────────────────────────────────────────────────────────────────
#  Plot — Raw ECG overview
# ─────────────────────────────────────────────────────────────────────────────

def plot_raw_ecg_overview(record, config, logger):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(f"Matplotlib unavailable, skipping raw ECG plot: {exc}")
        return

    fs      = record.fs
    raw_ecg = record.p_signal[:, 0].flatten()
    n_show  = min(int(60 * fs), len(raw_ecg))
    t       = np.arange(n_show) / fs

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(f"DEVICE — {config['record_name']} | Raw ECG Overview",
                 fontsize=14, fontweight="bold")

    axes[0].plot(t, raw_ecg[:n_show], color="#2c3e50", linewidth=0.6)
    axes[0].set_title("Raw ECG — First 60 Seconds")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Amplitude (mV)")
    axes[0].grid(True, alpha=0.3)

    chunk    = int(fs * 10)
    n_chunks = len(raw_ecg) // chunk
    envelope = [np.max(np.abs(raw_ecg[i*chunk:(i+1)*chunk]))
                for i in range(n_chunks)]
    t_env = np.arange(n_chunks) * 10 / 3600

    axes[1].fill_between(t_env, 0, envelope, alpha=0.6, color="#3498db")
    axes[1].set_title("Full Session — ECG Amplitude Envelope (10-sec chunks)")
    axes[1].set_xlabel("Time (hours)"); axes[1].set_ylabel("|Amplitude| max")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase0_raw_ecg_overview.png")
    try:
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved → {out}")
    except Exception as exc:
        logger.warning(f"Failed to save raw ECG plot: {exc}")
    finally:
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Run Phase 0 for a single session
# ─────────────────────────────────────────────────────────────────────────────

def run_phase0_record(config: dict, logger: logging.Logger,
                      db_path: str = TEST_DB_PATH) -> None:
    rec_name  = config["record_name"]
    db_row    = get_record(rec_name, "device", db_path)   # None if not registered — fine
    record_id = db_row["record_id"] if db_row else None

    try:
        record, ann = load_record(config, logger)
        plot_raw_ecg_overview(record, config, logger)

        logger.info(
            f"  No ground-truth annotations for '{rec_name}' "
            f"(device format) — no PSG distribution plot to generate"
        )

        logger.info(f"Phase 0 Done — {rec_name}")
        if record_id:
            mark_phase_done(record_id, 0, config["output_dir"], db_path)
    except Exception as exc:
        logger.error(f"Phase 0 FAILED for {rec_name}: {exc}")
        if record_id:
            mark_phase_failed(record_id, 0, str(exc), db_path)
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .p0_scan_datasets    import scan_datasets
    from .p0_register_records import register_all_records

    logger  = setup_logger("phase0_inspect", log_dir="logs")
    records = scan_datasets(CONFIG, logger)

    register_all_records(CONFIG, records, logger, db_path=CONFIG["db_path"])

    for rec in records:
        CONFIG["record_name"] = rec
        CONFIG["output_dir"]  = make_subject_dirs(
            rec, CONFIG["results_dir"], CONFIG["dataset"]
        )
        run_phase0_record(CONFIG, logger, db_path=CONFIG["db_path"])