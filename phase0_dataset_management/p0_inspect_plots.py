"""
=============================================================================
phase0_dataset_management/p0_inspect_plots.py
Load a single record, generate Phase-0 inspection plots:
  - Raw ECG overview (first 60 s + full-night amplitude envelope)
  - PSG sleep-stage distribution (bar + pie)
Marks phase0_done in SQLite when successful.
=============================================================================
"""

import os
import sys
import json
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0_dataset_management.p0_config  import CONFIG, get_record_path, make_subject_dirs
from phase0_dataset_management.p0_logging import setup_logger
from phase0_dataset_management.p0_metadata_extraction import (
    load_edf_signal, parse_hmc_annotations, _get_hmc_ann_path
)
from Database.db_manager import (
    get_record, mark_phase_done, mark_phase_failed, DB_PATH
)


# ─────────────────────────────────────────────────────────────────────────────
#  Load helpers (returns minimal record object + annotation)
# ─────────────────────────────────────────────────────────────────────────────

def _load_slpdb(config, logger):
    import wfdb
    record_path = get_record_path(config)
    logger.info(f"[SLPDB] Loading: {record_path}")
    record = wfdb.rdrecord(record_path)
    ann    = wfdb.rdann(record_path, "st")
    return record, ann


def _load_hmc(config, logger):
    edf_path = get_record_path(config) + ".edf"
    logger.info(f"[HMC] Loading EDF: {edf_path}")
    keywords = config.get("hmc_ecg_keywords", ["ECG", "EKG"])
    signal, fs, channel_names, ecg_idx, _ = load_edf_signal(
        edf_path, keywords, logger
    )
    signal = np.where(np.isfinite(signal), signal, np.nanmedian(signal))

    class _HMCRecord:
        pass
    rec           = _HMCRecord()
    rec.fs        = fs
    rec.sig_len   = len(signal)
    rec.sig_name  = channel_names
    rec.p_signal  = signal.reshape(-1, 1)

    ann_path = _get_hmc_ann_path(get_record_path(config))
    ann      = parse_hmc_annotations(ann_path, logger) if ann_path else []
    return rec, ann


def load_record(config, logger):
    if config.get("dataset") == "hmc":
        return _load_hmc(config, logger)
    return _load_slpdb(config, logger)


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1 — Raw ECG overview
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

    dataset = config.get("dataset", "slpdb").upper()
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(f"{dataset} — {config['record_name']} | Raw ECG Overview",
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
    axes[1].set_title("Full Night — ECG Amplitude Envelope (10-sec chunks)")
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
#  Plot 2 — PSG stage distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_psg_distribution(ann, config, logger):
    from collections import Counter
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(f"Matplotlib unavailable, skipping PSG plot: {exc}")
        return

    dataset      = config.get("dataset", "slpdb")
    stage_labels = config["stage_labels"]
    stage_colors = config["stage_colors"]

    if dataset == "hmc":
        stage_map  = config["hmc_stage_map"]
        raw_labels = [r[2].strip().upper() for r in ann]
    else:
        stage_map  = config["stage_map"]
        raw_labels = [lbl.strip().upper() for lbl in ann.aux_note]

    numeric = []
    for lbl in raw_labels:
        matched = stage_map.get(lbl, -1)
        if matched == -1:
            for k, v in stage_map.items():
                if k in lbl:
                    matched = v
                    break
        numeric.append(matched)

    counts       = Counter(numeric)
    valid_stages = sorted([s for s in counts if s >= 0])
    labels_text  = [stage_labels[s] for s in valid_stages]
    values       = [counts[s] for s in valid_stages]
    colors       = [stage_colors[s] for s in valid_stages]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"PSG Stage Distribution — {config['record_name']}",
                 fontsize=13, fontweight="bold")

    bars = axes[0].bar(labels_text, values, color=colors,
                       edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(val), ha="center", va="bottom", fontsize=10)
    axes[0].set_title("Epoch Count per Stage")
    axes[0].set_ylabel("Number of 30-sec Epochs")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].pie(values, labels=labels_text, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "black", "linewidth": 0.8})
    axes[1].set_title("Stage Proportion")

    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase0_psg_stage_distribution.png")
    try:
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved → {out}")
    except Exception as exc:
        logger.warning(f"Failed to save PSG distribution plot: {exc}")
    finally:
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Run Phase 0 for a single record
# ─────────────────────────────────────────────────────────────────────────────

def run_phase0_record(config: dict, logger: logging.Logger,
                      db_path: str = DB_PATH) -> None:
    rec_name  = config["record_name"]
    dataset   = config.get("dataset", "slpdb")
    db_row    = get_record(rec_name, dataset, db_path)
    record_id = db_row["record_id"] if db_row else None

    try:
        record, ann = load_record(config, logger)
        plot_raw_ecg_overview(record, config, logger)
        plot_psg_distribution(ann, config, logger)
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
    from phase0_dataset_management.p0_scan_datasets    import scan_datasets
    from phase0_dataset_management.p0_register_records import register_all_records

    logger  = setup_logger("phase0_inspect", log_dir="logs")
    records = scan_datasets(CONFIG, logger)
    register_all_records(CONFIG, records, logger)

    for rec in records:
        CONFIG["record_name"] = rec
        CONFIG["output_dir"]  = make_subject_dirs(
            rec, CONFIG["results_dir"], CONFIG["dataset"]
        )
        run_phase0_record(CONFIG, logger)
