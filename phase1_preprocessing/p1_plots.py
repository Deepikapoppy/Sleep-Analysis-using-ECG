"""
=============================================================================
phase1_preprocessing/p1_plots.py
All Phase-1 visualisation functions — exact logic from original file.

Plots produced
──────────────
1. phase1_preprocessing_stages.png   — pipeline stages (first 10 s)
2. phase1_dwt_coefficients.png       — sub-band coefficients before/after
3. phase1_raw_vs_dwt_cleaned.png     — raw vs cleaned 30-sec comparison
4. phase1_sqi_overview.png           — SQI across all epochs (3 panels)
5. phase1_epoch_quality.png          — good/bad epoch assessment
6. phase1_psd_raw_vs_clean.png       — PSD comparison with DWT bands marked
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase1_preprocessing.p1_dwt_denoise import _dwt_subbands


# ─────────────────────────────────────────────────────────────────────────────
#  Shared save helper
# ─────────────────────────────────────────────────────────────────────────────

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
#  1. Preprocessing stages
# ─────────────────────────────────────────────────────────────────────────────

def plot_preprocessing_stages(ecg_stages, fs, config, logger):
    """
    Multi-panel plot showing each processing stage (first 10 s).
    ecg_stages : [(label, signal), ...]
    """
    n_show   = int(10 * fs)
    t        = np.arange(n_show) / fs
    n_stages = len(ecg_stages)
    colors   = ["#7f8c8d", "#2980b9", "#27ae60", "#e67e22", "#8e44ad"]

    fig, axes = plt.subplots(n_stages, 1, figsize=(16, 3 * n_stages))
    if n_stages == 1:
        axes = [axes]
    fig.suptitle(
        f"ECG DWT Pre-Processing Stages — {config['record_name']} (first 10 s)",
        fontsize=13, fontweight="bold",
    )
    for ax, (label, sig), col in zip(axes, ecg_stages, colors):
        ax.plot(t, sig[:n_show], color=col, linewidth=0.8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        ax.set_xlim([0, 10])
    axes[-1].set_xlabel("Time (s)")

    _save(fig, config["output_dir"], "phase1_preprocessing_stages.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
#  2. DWT sub-band coefficients
# ─────────────────────────────────────────────────────────────────────────────

def plot_dwt_coefficients(ecg_ds, coeffs_raw, coeffs_proc,
                          fs, wavelet, level, config, logger,
                          n_show_sec=10):
    n_rows      = level + 1
    coeff_order = list(range(level + 1))[::-1]

    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 2.5 * n_rows))
    fig.suptitle(
        f"DWT Sub-band Coefficients — {config['record_name']} "
        f"[wavelet: {wavelet}, level: {level}]",
        fontsize=13, fontweight="bold",
    )
    axes[0, 0].set_title("Before DWT Processing (Raw)",
                          fontsize=11, fontweight="bold", color="#c0392b")
    axes[0, 1].set_title("After DWT Processing (Denoised)",
                          fontsize=11, fontweight="bold", color="#27ae60")

    for row_idx, ci in enumerate(coeff_order):
        if ci == 0:
            bname = f"A{level}"
            lo, hi = 0.0, round(fs / (2 ** (level + 1)), 2)
            tag_r  = "Zeroed → baseline wander removed"
            tag_p  = "Zero (baseline removed)"
        else:
            d_num = level - ci + 1
            bname = f"D{d_num}"
            hi    = round(fs / (2 **  d_num),     2)
            lo    = round(fs / (2 ** (d_num + 1)), 2)
            if d_num == 1:
                tag_r = "Zeroed → 50 Hz PLI removed"
                tag_p = "Zero (PLI removed)"
            else:
                tag_r = "Soft-threshold applied"
                tag_p = "Soft-thresholded"

        n_disp = min(len(coeffs_raw[ci]),
                     int(n_show_sec * fs / (2 ** (ci if ci > 0 else level))))
        n_disp = max(n_disp, 10)
        t_c    = np.arange(n_disp)

        axes[row_idx, 0].plot(t_c, coeffs_raw[ci][:n_disp],
                               color="#e74c3c", linewidth=0.7, alpha=0.85)
        axes[row_idx, 0].set_ylabel(f"{bname}\n{lo}–{hi}Hz", fontsize=8)
        axes[row_idx, 0].tick_params(labelsize=7)
        axes[row_idx, 0].grid(True, alpha=0.2)
        axes[row_idx, 0].set_title(f"{bname} — {tag_r}", fontsize=8,
                                    color="#7f8c8d")

        axes[row_idx, 1].plot(t_c, coeffs_proc[ci][:n_disp],
                               color="#27ae60", linewidth=0.7, alpha=0.85)
        axes[row_idx, 1].tick_params(labelsize=7)
        axes[row_idx, 1].grid(True, alpha=0.2)
        axes[row_idx, 1].set_title(f"{bname} — {tag_p}", fontsize=8,
                                    color="#7f8c8d")

    axes[-1, 0].set_xlabel("Coefficient index")
    axes[-1, 1].set_xlabel("Coefficient index")

    try:
        plt.subplots_adjust(hspace=0.6, wspace=0.25)
    except Exception:
        pass

    _save(fig, config["output_dir"], "phase1_dwt_coefficients.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Raw vs DWT-cleaned
# ─────────────────────────────────────────────────────────────────────────────

def plot_raw_vs_dwt_cleaned(raw_ecg_ds, clean_ecg, fs, config, logger,
                             seg_start_sec=30, seg_dur_sec=30):
    wavelet = config.get("dwt_wavelet", "db4")
    level   = config.get("dwt_level",   5)

    start     = int(seg_start_sec * fs)
    end       = min(start + int(seg_dur_sec * fs), len(raw_ecg_ds), len(clean_ecg))
    t         = np.arange(end - start) / fs

    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
    fig.suptitle(
        f"Raw vs DWT-Cleaned ECG — {config['record_name']} "
        f"(t = {seg_start_sec}–{seg_start_sec+seg_dur_sec} s)",
        fontsize=13, fontweight="bold",
    )
    axes[0].plot(t, raw_ecg_ds[start:end], color="#7f8c8d", linewidth=0.8)
    axes[0].set_title("Raw ECG (downsampled to 125 Hz)",
                       fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, clean_ecg[start:end], color="#2980b9", linewidth=0.8)
    axes[1].set_title(
        f"DWT-Cleaned ECG (wavelet={wavelet}, level={level}  |  "
        "A zeroed + PLI zeroed + soft-threshold + IDWT)",
        fontsize=10, fontweight="bold",
    )
    axes[1].set_ylabel("Norm. Amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.25)

    _save(fig, config["output_dir"], "phase1_raw_vs_dwt_cleaned.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
#  4. SQI overview
# ─────────────────────────────────────────────────────────────────────────────

def plot_sqi_overview(sqi_df: pd.DataFrame, bad_mask, config, logger):
    t_hr    = np.asarray(sqi_df["epoch_idx"].to_numpy(dtype=float)) * 30.0 / 3600.0
    ksq     = np.asarray(sqi_df["kSQI_norm"].to_numpy(dtype=float))
    psq     = np.asarray(sqi_df["pSQI"].to_numpy(dtype=float))
    overall = np.asarray(sqi_df["overall_sqi"].to_numpy(dtype=float))
    cols_ep = np.where(bad_mask, "#e74c3c", "#2ecc71")

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(f"Signal Quality Index (SQI) — {config['record_name']}",
                 fontsize=13, fontweight="bold")

    axes[0].plot(t_hr, ksq, color="#9b59b6", linewidth=0.7, alpha=0.7)
    axes[0].axhline(0.25, color="red", linestyle="--", linewidth=1.0,
                    label="Threshold (norm kSQI ≥ 0.25)")
    axes[0].set_ylabel("kSQI (norm.)", fontsize=9)
    axes[0].set_title("Kurtosis SQI — QRS sharpness")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.2)

    axes[1].plot(t_hr, psq, color="#e67e22", linewidth=0.7, alpha=0.7)
    axes[1].axhline(0.30, color="red", linestyle="--", linewidth=1.0,
                    label="Threshold (pSQI ≥ 0.30)")
    axes[1].set_ylabel("pSQI", fontsize=9)
    axes[1].set_title("Power SQI — QRS band (5–15 Hz) energy fraction")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.2)

    finite = np.isfinite(t_hr) & np.isfinite(overall)
    if finite.any():
        try:
            axes[2].bar(t_hr[finite], overall[finite],
                        width=float(30.0 / 3600.0),
                        color=cols_ep[finite].tolist(), alpha=0.7)
        except Exception as exc:
            logger.warning(f"Bar plot fallback to scatter: {exc}")
            axes[2].scatter(t_hr[finite], overall[finite],
                            c=cols_ep[finite].tolist(), s=6)
    axes[2].axhline(0.50, color="black", linestyle="--", linewidth=1.2,
                    label="Threshold (overall SQI ≥ 0.50)")
    axes[2].set_ylabel("Overall SQI", fontsize=9)
    axes[2].set_title("Overall Composite SQI (green=good, red=bad)")
    axes[2].set_xlabel("Time (hours)")
    axes[2].set_ylim([0, 1.1])
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.2)

    _save(fig, config["output_dir"], "phase1_sqi_overview.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
#  5. Epoch quality assessment
# ─────────────────────────────────────────────────────────────────────────────

def plot_epoch_quality(epochs, bad_mask, config, logger):
    n          = len(epochs)
    good_count = int((~bad_mask).sum())
    bad_count  = int(bad_mask.sum())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Epoch Quality Assessment — {config['record_name']}",
                 fontsize=13, fontweight="bold")

    axes[0].bar(["Good Epochs", "Bad Epochs"], [good_count, bad_count],
                color=["#2ecc71", "#e74c3c"], edgecolor="black")
    axes[0].set_title("Epoch Quality Count")
    axes[0].set_ylabel("Count")
    for i, v in enumerate([good_count, bad_count]):
        axes[0].text(i, v + 0.5, str(v), ha="center", fontsize=11)
    axes[0].grid(True, axis="y", alpha=0.3)

    stds      = [np.std(ep) for ep in epochs]
    colors_ep = ["#e74c3c" if b else "#3498db" for b in bad_mask]
    axes[1].scatter(np.arange(n), stds, c=colors_ep, s=4, alpha=0.6)
    axes[1].set_title("Epoch Amplitude Std (red=bad)")
    axes[1].set_xlabel("Epoch index")
    axes[1].set_ylabel("Std (norm. units)")
    axes[1].grid(True, alpha=0.25)

    good_idx = np.where(~bad_mask)[0]
    bad_idx  = np.where(bad_mask)[0]
    g_ep = epochs[good_idx[0]] if len(good_idx) else epochs[0]
    b_ep = epochs[bad_idx[0]]  if len(bad_idx)  else epochs[0]
    gi   = good_idx[0] if len(good_idx) else 0
    bi   = bad_idx[0]  if len(bad_idx)  else 0
    axes[2].plot(g_ep, color="#2ecc71", label=f"Good (ep {gi})", alpha=0.8)
    axes[2].plot(b_ep, color="#e74c3c", label=f"Bad  (ep {bi})", alpha=0.8)
    axes[2].set_title("Good vs Bad Epoch Example")
    axes[2].set_xlabel("Samples")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    _save(fig, config["output_dir"], "phase1_epoch_quality.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
#  6. PSD comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectral_comparison(raw_ecg, clean_ecg, fs, config, logger,
                              wavelet="db4", level=5):
    from scipy.signal import welch as _welch

    nperseg    = min(int(fs * 4), len(raw_ecg))
    f_r, psd_r = _welch(raw_ecg,   fs=fs, nperseg=nperseg)
    f_c, psd_c = _welch(clean_ecg, fs=fs, nperseg=nperseg)

    approx_cutoff = fs / (2 ** (level + 1))
    d1_lo         = fs / 4.0
    d1_hi         = fs / 2.0

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.semilogy(f_r, psd_r, label="Raw ECG",
                color="#e74c3c", alpha=0.8, linewidth=1.2)
    ax.semilogy(f_c, psd_c, label="DWT-Cleaned ECG",
                color="#2ecc71", alpha=0.8, linewidth=1.2)

    ax.axvspan(0,     approx_cutoff, alpha=0.12, color="gray",
               label=f"< {approx_cutoff:.1f} Hz (A{level} zeroed — baseline)")
    ax.axvspan(d1_lo, d1_hi,         alpha=0.12, color="orange",
               label=f"{d1_lo:.1f}–{d1_hi:.1f} Hz (D1 zeroed — PLI band)")
    ax.axvline(50.0, color="purple", linestyle=":", linewidth=1.2,
               label="50 Hz PLI")

    ax.set_xlim([0, fs / 2])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (log scale)")
    ax.set_title(
        f"PSD — Raw vs DWT-Cleaned ECG | {config['record_name']} "
        f"[{wavelet}, L={level}]"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    _save(fig, config["output_dir"], "phase1_psd_raw_vs_clean.png", logger)
