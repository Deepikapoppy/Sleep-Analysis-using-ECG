"""
=============================================================================
phase5_final_report_testing/p5_plots.py
Diagnostic / final-report plots for a device session's PREDICTED hypnogram.

There is no PSG ground truth for device sessions, so every plot here is
built on the RF model's prediction (phase4b's smoothed output) instead of
a PSG label. Anything from training's Phase 6 that requires comparing
against ground truth (accuracy, kappa, confusion matrix, per-stage F1,
detection-method accuracy) has been intentionally dropped — there is
nothing to compare against.

Plots generated (all saved to config["output_dir"]):
  phase5_predicted_hypnogram.png   predicted hypnogram, single panel
  phase5_stage_distribution.png    predicted stage duration bar + pie
  phase5_sqi_overlay.png           predicted hypnogram + Phase 1 SQI overlay
                                    (skipped if phase1_sqi.csv absent)
  phase5_hrv_summary.png           HRV feature boxplots by predicted stage
                                    (skipped if phase3_hrv_features.csv absent)
  phase5_final_summary.png         final dashboard: predicted hypnogram +
                                    model/DWT provenance + stage durations
                                    (no accuracy/kappa — none exists)
=============================================================================
"""

import os
import json
import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


STAGE_NAMES  = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {
    0: "#e74c3c", 1: "#f39c12", 2: "#f1c40f", 3: "#2ecc71", 4: "#3498db",
}


def _hypnogram_axis(ax, hyp: np.ndarray, epoch_sec: int, title: str,
                    recording_start_time: Optional[str] = None) -> None:
    """
    Step-plot of stage vs time, colour-coded background bands.
    Self-contained (training's version lives in an unrelated module we
    don't want this testing package to depend on).
    """
    t_hr = np.arange(len(hyp)) * epoch_sec / 3600.0
    for i, stage in enumerate(hyp):
        ax.axvspan(t_hr[i], t_hr[i] + epoch_sec / 3600.0, alpha=0.85,
                   color=STAGE_COLORS.get(int(stage), "#95a5a6"), linewidth=0)
    ax.step(t_hr, hyp, where="post", color="black", linewidth=0.6, alpha=0.6)
    ax.set_yticks(sorted(STAGE_NAMES.keys()))
    ax.set_yticklabels([STAGE_NAMES[k] for k in sorted(STAGE_NAMES.keys())])
    ax.set_ylim([-0.5, 4.5])
    xlabel = "Time (hours)"
    if recording_start_time:
        xlabel += f"  (recording start ≈ {recording_start_time})"
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1: Predicted hypnogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_predicted_hypnogram(pred_smooth: np.ndarray, config: dict,
                              logger: logging.Logger,
                              recording_start_time: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(18, 4))
    fig.suptitle(
        f"ECG-Derived Predicted Hypnogram (RF model) — {config['record_name']}",
        fontsize=13, fontweight="bold",
    )
    _hypnogram_axis(ax, pred_smooth, epoch_sec=30,
                    title="Predicted (RF, smoothed)",
                    recording_start_time=recording_start_time)
    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase5_predicted_hypnogram.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 2: Predicted stage distribution (bar + pie)
# ─────────────────────────────────────────────────────────────────────────────

def plot_stage_distribution(pred_smooth: np.ndarray, config: dict,
                            logger: logging.Logger) -> None:
    counts = Counter(pred_smooth)
    stages = sorted(s for s in counts if s in STAGE_NAMES)
    labels = [STAGE_NAMES[s] for s in stages]
    values = [counts[s] * 30 / 60 for s in stages]   # → minutes
    colors = [STAGE_COLORS[s] for s in stages]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Predicted Stage Distribution — {config['record_name']}",
        fontsize=13, fontweight="bold",
    )
    bars = axes[0].bar(labels, values, color=colors, edgecolor="black")
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.0f}", ha="center", fontsize=9)
    axes[0].set_ylabel("Duration (minutes)")
    axes[0].set_title("Predicted Stage Durations")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "black", "linewidth": 0.8})
    axes[1].set_title("Predicted Stage Proportions")

    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase5_stage_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 3: Predicted hypnogram + Phase 1 SQI overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_sqi_overlay(pred_smooth: np.ndarray, sqi_df: Optional[pd.DataFrame],
                     config: dict, logger: logging.Logger) -> None:
    if sqi_df is None or "overall_sqi" not in sqi_df.columns:
        logger.info("SQI overlay skipped (phase1_sqi.csv not available).")
        return

    n      = min(len(pred_smooth), len(sqi_df))
    t_ep   = np.arange(n) * 30 / 3600.0
    stages = pred_smooth[:n]
    sqi    = sqi_df["overall_sqi"].values[:n]

    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
    fig.suptitle(
        f"Predicted Hypnogram with Phase 1 SQI Overlay — {config['record_name']}",
        fontsize=13, fontweight="bold",
    )

    _hypnogram_axis(axes[0], stages, epoch_sec=30, title="Predicted (RF, smoothed)")

    for ep_i, (st, sq) in enumerate(zip(stages, sqi)):
        col = STAGE_COLORS.get(int(st), "#95a5a6")
        axes[1].axvspan(t_ep[ep_i], t_ep[ep_i] + 30 / 3600,
                        alpha=0.18, color=col, linewidth=0)

    axes[1].plot(t_ep, sqi, color="#2c3e50", linewidth=0.7, alpha=0.55,
                label="SQI (raw)")
    sqi_smooth = pd.Series(sqi).rolling(10, center=True, min_periods=1).mean().values
    axes[1].plot(t_ep, sqi_smooth, color="#2980b9", linewidth=1.8,
                label="SQI (10-ep smoothed)")
    axes[1].axhline(0.50, color="red", linestyle="--", linewidth=1.0,
                    alpha=0.8, label="SQI threshold 0.50")
    axes[1].fill_between(t_ep, 0, 1, where=(sqi < 0.50),
                         color="#e74c3c", alpha=0.15, label="Low-SQI epoch")
    axes[1].set_ylim([0, 1.1])
    axes[1].set_ylabel("Overall SQI", fontsize=9)
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_title(
        "Phase 1 Signal Quality Index (background = predicted stage colour)",
        fontsize=10,
    )
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.2)

    try:
        plt.tight_layout()
    except Exception:
        pass
    out = os.path.join(config["output_dir"], "phase5_sqi_overlay.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 4: HRV feature distributions by predicted stage
# ─────────────────────────────────────────────────────────────────────────────

def plot_hrv_summary(pred_smooth: np.ndarray, hrv_df: Optional[pd.DataFrame],
                     config: dict, logger: logging.Logger) -> None:
    if hrv_df is None:
        logger.warning("phase3_hrv_features.csv not available — HRV summary skipped.")
        return

    n   = min(len(hrv_df), len(pred_smooth))
    hrv = hrv_df.iloc[:n].copy()
    hrv["pred_stage"] = pred_smooth[:n]
    hrv["stage_name"] = hrv["pred_stage"].map(STAGE_NAMES)

    valid = hrv[hrv["pred_stage"].isin(STAGE_NAMES.keys())].copy()
    order = [s for s in ["Wake", "N1", "N2", "N3", "REM"]
             if s in valid["stage_name"].unique()]

    features = [
        ("mean_hr",             "Mean HR (bpm)",            "#e74c3c"),
        ("rmssd",               "RMSSD (ms)",               "#2ecc71"),
        ("lf_hf_ratio",         "LF/HF Ratio",              "#e67e22"),
        ("hf_power",            "HF Power",                 "#3498db"),
        ("sdnn",                "SDNN (ms)",                "#f39c12"),
        ("sampen",              "SampEn",                   "#9b59b6"),
        ("pnn20",               "pNN20 (%) [v2]",           "#e67e22"),
        ("perm_en",             "PermEn (0–1) [v2]",        "#f97316"),
        ("resp_rate_est",       "Resp Rate (br/min) [v2]",  "#4fc3f7"),
        ("rmssd_mean_rr_ratio", "RMSSD/meanRR×100 [v2]",   "#27ae60"),
        ("hr_range",            "HR Range (bpm) [v2]",      "#1abc9c"),
        ("rr_iqr",              "RR IQR (ms) [v2]",         "#8e44ad"),
    ]
    present = [(f, t, c) for f, t, c in features if f in valid.columns]
    if not present:
        logger.warning("No known HRV features found — skipping HRV summary plot.")
        return

    n_feat = len(present)
    cols   = 4
    rows   = (n_feat + cols - 1) // cols
    stage_palette = {STAGE_NAMES[k]: v for k, v in STAGE_COLORS.items()}

    fig, axes = plt.subplots(rows, cols, figsize=(22, rows * 4))
    axes_flat = np.array(axes).flatten()
    fig.suptitle(
        f"HRV Features by Predicted Stage — {config['record_name']}\n"
        "(No ground truth — grouped by RF model prediction, not PSG)",
        fontsize=13, fontweight="bold",
    )

    for ax, (feat, title, col) in zip(axes_flat, present):
        sub  = valid[["stage_name", feat]].dropna()
        grps = [sub[sub["stage_name"] == s][feat].values
                for s in order if s in sub["stage_name"].values]
        lbls = [f"{s}\n(n={len(sub[sub['stage_name']==s])})"
                for s in order if s in sub["stage_name"].values]
        if not grps:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(grps, labels=lbls, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.8))
        for patch, s in zip(bp["boxes"],
                            [s for s in order if s in sub["stage_name"].values]):
            patch.set_facecolor(stage_palette.get(s, col))
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_xlabel("Predicted Stage", fontsize=8)

    for ax in axes_flat[n_feat:]:
        ax.set_visible(False)

    try:
        plt.tight_layout()
    except Exception:
        pass
    out = os.path.join(config["output_dir"], "phase5_hrv_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 5: Final summary dashboard (no accuracy/kappa — no ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def plot_final_summary(pred_smooth: np.ndarray, phase1_meta: dict,
                       predict_meta: dict, config: dict,
                       logger: logging.Logger) -> None:
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"SLEEP ECG ANALYSIS — DEVICE DEPLOYMENT FINAL REPORT\n"
        f"{config['record_name']}  (RF prediction — no PSG ground truth available)",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.3, height_ratios=[1, 1.3])

    ax0 = fig.add_subplot(gs[0, :])
    _hypnogram_axis(ax0, pred_smooth, 30, "Predicted Hypnogram (RF, smoothed)")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis("off")
    wavelet = phase1_meta.get("dwt_wavelet", "db4")
    level   = phase1_meta.get("dwt_level", 5)
    sqi_m   = phase1_meta.get("mean_sqi", "N/A")
    summary_text = (
        f"━━━━ SESSION SUMMARY ━━━━\n\n"
        f"Total Epochs     : {len(pred_smooth)}\n"
        f"Model            : {predict_meta.get('model_type', 'N/A')}\n"
        f"Feature columns  : {predict_meta.get('n_feature_cols', 'N/A')}\n\n"
        f"── DWT Provenance (Phase 1) ──\n"
        f"  Wavelet : {wavelet}   Level : {level}\n"
        f"  Mean SQI: {sqi_m}\n\n"
        f"── Predicted Stage Durations ──\n"
    )
    for s in sorted(STAGE_NAMES.keys()):
        mins = (pred_smooth == s).sum() * 30 / 60
        summary_text += f"  {STAGE_NAMES[s]:8s}: {mins:6.1f} min\n"
    summary_text += (
        "\nNOTE: No accuracy / kappa / confusion matrix — device sessions "
        "carry no PSG ground truth to evaluate against."
    )
    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes,
             fontsize=10, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.9))

    ax2 = fig.add_subplot(gs[1, 1])
    order  = sorted(STAGE_NAMES.keys())
    names  = [STAGE_NAMES[s] for s in order]
    mins   = [(pred_smooth == s).sum() * 30 / 60 for s in order]
    colors = [STAGE_COLORS[s] for s in order]
    bars = ax2.bar(names, mins, color=colors, edgecolor="black", alpha=0.9)
    for bar, v in zip(bars, mins):
        if v > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.0f}m", ha="center", fontsize=9)
    ax2.set_ylabel("Minutes")
    ax2.set_title("Predicted Stage Duration")
    ax2.grid(True, axis="y", alpha=0.3)

    out = os.path.join(config["output_dir"], "phase5_final_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")
