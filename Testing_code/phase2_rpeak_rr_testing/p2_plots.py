"""
=============================================================================
phase2_rpeak_rr/p2_plots.py
All Phase-2 visualisation functions.

Plots produced
──────────────
1. phase2_method_comparison.png      — all 3 methods on same epoch
2. phase2_rpeak_examples.png         — best-method R-peaks on N epochs
3. phase2_raw_vs_preprocessed.png    — raw | clean | R-peaks 3-panel
4. phase2_full_night_tachogram.png   — RR intervals + HR over night
5. phase2_hr_distribution.png        — HR histogram + time series
6. phase2_detection_quality.png      — quality score per epoch
7. phase2_method_breakdown.png       — bar chart: which method won most

R-peak colour: ORANGE (#f39c12) throughout.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .p2_rpeak_detection import (
    detect_rpeaks_all_methods, RPEAK_COLOR, RPEAK_EDGE
)

# Cascade order — used for consistent plot ordering
METHOD_PRIORITY = ["neurokit", "elgendi2010", "scipy_prominence"]

_METHOD_COLORS = {
    "neurokit"         : "#2ecc71",
    "elgendi2010"      : "#3498db",
    "scipy_prominence" : "#e74c3c",
}


def _save(fig, out_dir: str, filename: str, logger: logging.Logger) -> None:
    path = os.path.join(out_dir, filename)
    try:
        plt.tight_layout()
    except Exception:
        pass
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved → {path}")
    except Exception as exc:
        logger.warning(f"Failed to save {filename}: {exc}")
    finally:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  1. NEW — All 5 methods comparison on same epoch(s)
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_comparison(epochs: np.ndarray,
                            results: list,
                            fs: float,
                            config: dict,
                            logger: logging.Logger,
                            n_epochs: int = 3) -> None:
    """
    For N example epochs: plot all 3 detectors on the SAME ECG trace.
    Each method gets its own subplot row, coloured distinctly.
    Lets you visually compare where each detector places R-peaks.
    """
    good = [r for r in results if not r["is_bad"] and r["n_peaks"] > 3][:n_epochs]
    if not good:
        logger.warning("No good epochs for method comparison plot.")
        return

    methods = list(METHOD_PRIORITY)   # consistent order
    n_rows  = len(methods)

    for ep_num, row in enumerate(good):
        ep  = epochs[row["epoch_idx"]]
        t   = np.arange(len(ep)) / fs
        
        # Use stored peaks from pipeline — NO re-detection
        stored_all_peaks = row.get("all_peaks", {})
        
        fig, axes = plt.subplots(n_rows, 1, figsize=(18, 2.8 * n_rows),
                                 sharex=True)
        fig.suptitle(
            f"All 3 R-Peak Methods — {config['record_name']}  "
            f"Epoch {row['epoch_idx']}  |  Best: {row['method']}",
            fontsize=12, fontweight="bold"
            )
        
        for ax, method in zip(axes, methods):
            peaks = np.array(stored_all_peaks.get(method, []), dtype=int)  # ← fix
            color = _METHOD_COLORS.get(method, "#7f8c8d")
            ax.plot(t, ep, color="#2c3e50", linewidth=0.7, alpha=0.8)
            if len(peaks) > 0:
                ax.scatter(peaks / fs, ep[peaks],
                           color=color, s=60, zorder=5,
                           edgecolors="black", linewidths=0.5,
                           label=f"{method}  ({len(peaks)} peaks)")
            else:
                ax.text(0.5, 0.5, f"{method}: no peaks detected",
                        transform=ax.transAxes, ha="center",
                        fontsize=9, color="red")
            ax.set_ylabel("Amp", fontsize=8)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.2)
            if method == row["method"]:
                ax.set_facecolor("#f0fff0")
                ax.set_title("★ BEST METHOD", fontsize=8, color="green")
                
        axes[-1].set_xlabel("Time (s)")
        fname = f"phase2_method_comparison_ep{row['epoch_idx']}.png"
        _save(fig, config["output_dir"], fname, logger)


# ─────────────────────────────────────────────────────────────────────────────
#  2. R-peak examples (best method)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rpeak_examples(epochs: np.ndarray,
                         results: list,
                         fs: float,
                         config: dict,
                         logger: logging.Logger,
                         n_show: int = 4) -> None:
    good = [r for r in results if not r["is_bad"] and r["n_peaks"] > 3]
    if not good:
        return

    fig, axes = plt.subplots(n_show, 1, figsize=(16, 3.5 * n_show))
    if n_show == 1:
        axes = [axes]
    fig.suptitle(f"R-Peak Detection (Best Method) — {config['record_name']}",
                 fontsize=13, fontweight="bold")

    for ax, row in zip(axes, good[:n_show]):
        ep    = epochs[row["epoch_idx"]]
        t     = np.arange(len(ep)) / fs

        # ── CHANGE: use stored best peaks, no re-detection ──
        epoch_offset = row["epoch_idx"] * int(fs * config["epoch_sec"])
        stored_abs   = np.array(row.get("r_peak_samples_abs", []), dtype=int)
        peaks        = stored_abs - epoch_offset
        # ────────────────────────────────────────────────────

        ax.plot(t, ep, color="#2c3e50", linewidth=0.7, label="DWT-Cleaned ECG")
        if len(peaks):
            ax.scatter(peaks / fs, ep[peaks],
                       color=RPEAK_COLOR, s=55, zorder=5,
                       edgecolors=RPEAK_EDGE, linewidths=0.8,
                       label=f"R-peaks ({len(peaks)}) — {row['method']}")
        ax.set_title(
            f"Epoch {row['epoch_idx']}  |  HR={row['mean_hr']:.1f} bpm  |  "
            f"Method: {row['method']}  |  RR valid: {row['n_rr']}",
            fontsize=9
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Norm. Amplitude")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)

    _save(fig, config["output_dir"], "phase2_rpeak_examples.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Raw | Clean | R-peaks comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_raw_clean_rpeaks(epochs, results, fs, config, logger,
                           raw_ecg_ds=None, clean_ecg_full=None,
                           n_panels: int = 3) -> None:
    good = [r for r in results if not r["is_bad"] and r["n_peaks"] > 3][:n_panels]
    if not good:
        return

    spe  = epochs.shape[1]
    fig, axes = plt.subplots(len(good), 3, figsize=(22, 4 * len(good)))
    if len(good) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Raw ECG  |  DWT-Cleaned  |  R-Peak Detection — {config['record_name']}",
        fontsize=13, fontweight="bold"
    )
    for ci, ct in enumerate(["Raw ECG (125 Hz, pre-DWT)",
                               "DWT-Cleaned + Z-scored",
                               "DWT-Cleaned + R-peaks (orange)"]):
        axes[0, ci].set_title(ct, fontsize=10, fontweight="bold")

    for ri, row in enumerate(good):
        ep_idx = row["epoch_idx"]
        ep     = epochs[ep_idx]
        t      = np.arange(spe) / fs

        # Col 0: raw
        if raw_ecg_ds is not None:
            s = ep_idx * spe
            axes[ri, 0].plot(t, raw_ecg_ds[s:s+spe][:len(t)],
                              color="#7f8c8d", linewidth=0.8)
        else:
            axes[ri, 0].text(0.5, 0.5, "Not saved",
                              ha="center", transform=axes[ri, 0].transAxes)
        axes[ri, 0].set_ylabel(f"Ep {ep_idx}", fontsize=9)
        axes[ri, 0].grid(True, alpha=0.2)

        # Col 1: clean
        if clean_ecg_full is not None:
            s = ep_idx * spe
            axes[ri, 1].plot(t, clean_ecg_full[s:s+spe][:len(t)],
                              color="#27ae60", linewidth=0.8)
        else:
            axes[ri, 1].plot(t, ep, color="#2980b9", linewidth=0.8)
        axes[ri, 1].grid(True, alpha=0.2)

        # Col 2: R-peaks
        det   = detect_rpeaks_all_methods(ep, fs)
        peaks = det["best_peaks"]
        axes[ri, 2].plot(t, ep, color="#2c3e50", linewidth=0.8)
        if len(peaks) > 0:
            axes[ri, 2].scatter(peaks / fs, ep[peaks],
                                 color=RPEAK_COLOR, s=55, zorder=5,
                                 edgecolors=RPEAK_EDGE, linewidths=0.8)
        axes[ri, 2].grid(True, alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    _save(fig, config["output_dir"], "phase2_raw_vs_preprocessed_rpeaks.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
#  4. Full-night tachogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_tachogram(tachogram_t, tachogram_rr, config, logger) -> None:
    t_hr = tachogram_t / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle(f"Full-Night Tachogram — {config['record_name']}",
                 fontsize=13, fontweight="bold")

    axes[0].scatter(t_hr, tachogram_rr, s=1.5, color="#2980b9", alpha=0.5)
    axes[0].set_ylabel("RR Interval (ms)")
    axes[0].set_title("RR Interval Tachogram")
    axes[0].set_ylim([0, 2200])
    axes[0].grid(True, alpha=0.25)

    hr = 60000.0 / np.clip(tachogram_rr, 100, 3000)
    axes[1].scatter(t_hr, hr, s=1.5, color="#e74c3c", alpha=0.5)
    axes[1].set_ylabel("Heart Rate (bpm)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_title("Instantaneous Heart Rate")
    axes[1].set_ylim([20, 180])
    axes[1].grid(True, alpha=0.25)

    _save(fig, config["output_dir"], "phase2_full_night_tachogram.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
#  5. HR distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_hr_histogram(results: list, config: dict,
                      logger: logging.Logger) -> None:
    hr_vals = [r["mean_hr"] for r in results if np.isfinite(r["mean_hr"])]
    if not hr_vals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Heart Rate Distribution — {config['record_name']}",
                 fontsize=13, fontweight="bold")

    axes[0].hist(hr_vals, bins=40, color="#3498db", edgecolor="black", alpha=0.8)
    axes[0].axvline(np.mean(hr_vals), color="red", linestyle="--",
                    label=f"Mean {np.mean(hr_vals):.1f}")
    axes[0].axvline(np.median(hr_vals), color="green", linestyle="--",
                    label=f"Median {np.median(hr_vals):.1f}")
    axes[0].set_xlabel("Heart Rate (bpm)")
    axes[0].set_ylabel("Epoch Count")
    axes[0].set_title("HR Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    ep_idx = [r["epoch_idx"] for r in results if np.isfinite(r["mean_hr"])]
    axes[1].plot(np.array(ep_idx) * 30 / 3600, hr_vals,
                  color="#e74c3c", linewidth=0.7)
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Mean HR (bpm)")
    axes[1].set_title("Mean HR per Epoch over Night")
    axes[1].grid(True, alpha=0.3)

    _save(fig, config["output_dir"], "phase2_hr_distribution.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
#  6. Detection quality per epoch
# ─────────────────────────────────────────────────────────────────────────────

def plot_detection_quality(results: list, config: dict,
                            logger: logging.Logger) -> None:
    ep_idx  = [r["epoch_idx"] for r in results]
    quality = [r["quality"]    for r in results]

    fig, ax = plt.subplots(figsize=(16, 4))
    c = ["#e74c3c" if q < 0.7 else "#2ecc71" for q in quality]
    ax.bar(ep_idx, quality, color=c, width=1.0, alpha=0.8)
    ax.axhline(0.7, color="black", linestyle="--", linewidth=1,
               label="Quality threshold (0.7)")
    ax.set_xlabel("Epoch Index")
    ax.set_ylabel("Quality Score")
    ax.set_title(f"R-Peak Detection Quality per Epoch — {config['record_name']}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    _save(fig, config["output_dir"], "phase2_detection_quality.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
#  7. Method breakdown (which method won most epochs)
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_breakdown(method_counts: dict, config: dict,
                           logger: logging.Logger) -> None:
    if not method_counts:
        return

    methods = list(METHOD_PRIORITY)
    counts  = [method_counts.get(m, 0) for m in methods]
    colors  = [_METHOD_COLORS.get(m, "#7f8c8d") for m in methods]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(methods, counts, color=colors, edgecolor="black", alpha=0.85)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(count),
                ha="center", fontsize=10)
    ax.set_xlabel("Detection Method")
    ax.set_ylabel("Epochs Won")
    ax.set_title(f"R-Peak Method Wins — {config['record_name']}")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)

    _save(fig, config["output_dir"], "phase2_method_breakdown.png", logger)