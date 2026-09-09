"""
=============================================================================
phase5_psg_labels/p5_plots.py
Step 3 — Diagnostic plots for Phase 5.

Plots generated (all saved to config["output_dir"]):
  phase5_psg_hypnogram.png        PSG gold-standard hypnogram
  phase5_psg_distribution.png     Stage duration bar + pie
  phase5_annotation_timeline.png  Raw annotation timeline (pre-alignment)
  phase5_psg_sqi_overlay.png      PSG hypnogram + Phase 1 SQI overlay
  phase5_psg_hrv_summary.png      HRV feature boxplots by PSG stage
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
import matplotlib.patches as mpatches

from phase4_sleep_classification.p4_plots import (
    STAGE_NAMES, STAGE_COLORS, _hypnogram_axis,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1: PSG hypnogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_psg_hypnogram(psg_hyp: np.ndarray,
                        config: dict,
                        logger: logging.Logger) -> None:
    """Single-panel PSG gold-standard hypnogram with optional clock-time axis."""
    recording_start_time = None
    try:
        meta1_path = os.path.join(config["output_dir"], "phase1_meta.json")
        if os.path.exists(meta1_path):
            with open(meta1_path) as f:
                recording_start_time = json.load(f).get("recording_start_time")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(18, 4))
    fig.suptitle(
        f"PSG Gold Standard Hypnogram — {config['record_name']}",
        fontsize=13, fontweight="bold",
    )
    _hypnogram_axis(ax, psg_hyp, epoch_sec=30,
                    title="PSG Gold Standard",
                    recording_start_time=recording_start_time)
    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase5_psg_hypnogram.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 2: Stage distribution (bar + pie)
# ─────────────────────────────────────────────────────────────────────────────

def plot_psg_stage_distribution(psg_hyp: np.ndarray,
                                  config: dict,
                                  logger: logging.Logger) -> None:
    counts = Counter(psg_hyp)
    stages = sorted(s for s in counts if s >= 0)
    labels = [STAGE_NAMES.get(s, str(s)) for s in stages]
    values = [counts[s] * 30 / 60 for s in stages]          # → minutes
    colors = [STAGE_COLORS.get(s, "#95a5a6") for s in stages]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"PSG Stage Distribution — {config['record_name']}",
        fontsize=13, fontweight="bold",
    )
    bars = axes[0].bar(labels, values, color=colors, edgecolor="black")
    for bar, v in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{v:.0f}", ha="center", fontsize=9,
        )
    axes[0].set_ylabel("Duration (minutes)")
    axes[0].set_title("PSG Stage Durations")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "black", "linewidth": 0.8})
    axes[1].set_title("PSG Stage Proportions")

    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase5_psg_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 3: Raw annotation timeline (pre-alignment)
# ─────────────────────────────────────────────────────────────────────────────

def plot_annotation_timeline(ann_times: np.ndarray,
                               ann_stages: np.ndarray,
                               config: dict,
                               logger: logging.Logger) -> None:
    """Colour-coded bars showing the raw annotation stream before epoch binning."""
    t_hr       = ann_times / 3600.0
    colors_map = {**STAGE_COLORS, -1: "#95a5a6"}

    fig, ax = plt.subplots(figsize=(18, 4))
    for t, s in zip(t_hr, ann_stages):
        ax.axvspan(t, t + 30 / 3600, alpha=0.7,
                   color=colors_map.get(int(s), "gray"))
    ax.set_xlabel("Time (hours)")
    ax.set_yticks([])
    ax.set_title(
        f"Raw PSG Annotation Timeline — {config['record_name']}",
        fontweight="bold",
    )
    patches = [mpatches.Patch(color=STAGE_COLORS[i],
                               label=STAGE_NAMES[i]) for i in range(5)]
    ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=5)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase5_annotation_timeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 4: PSG hypnogram + Phase 1 SQI overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_psg_sqi_overlay(psg_hyp: np.ndarray,
                          sqi_df: Optional[pd.DataFrame],
                          config: dict,
                          logger: logging.Logger) -> None:
    """
    Two-panel plot:
      Top    — PSG gold-standard hypnogram
      Bottom — Phase 1 overall_sqi trace, background coloured by PSG stage.

    Confirms that low-quality ECG epochs do not cluster in specific stages,
    which would bias the evaluation in Phase 6 (metrics).
    """
    if sqi_df is None or "overall_sqi" not in sqi_df.columns:
        logger.info("SQI overlay skipped (phase1_sqi.csv not available).")
        return

    n      = min(len(psg_hyp), len(sqi_df))
    t_ep   = np.arange(n) * 30 / 3600.0
    stages = psg_hyp[:n]
    sqi    = sqi_df["overall_sqi"].values[:n]

    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
    fig.suptitle(
        f"PSG Hypnogram with Phase 1 SQI Overlay — {config['record_name']}",
        fontsize=13, fontweight="bold",
    )

    # Panel 1 — PSG hypnogram
    _hypnogram_axis(axes[0], stages, epoch_sec=30, title="PSG Gold Standard")

    # Panel 2 — SQI trace with per-epoch stage-colour background
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
        "Phase 1 Signal Quality Index (background = PSG stage colour)",
        fontsize=10,
    )
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.2)

    try:
        plt.tight_layout()
    except Exception:
        pass
    out = os.path.join(config["output_dir"], "phase5_psg_sqi_overlay.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 5: HRV feature distributions by PSG gold-standard stage
# ─────────────────────────────────────────────────────────────────────────────

def plot_psg_hrv_summary(psg_hyp: np.ndarray,
                          config: dict,
                          logger: logging.Logger) -> None:
    """
    Boxplots of core + Phase 3 v2 HRV features grouped by PSG stage.

    Validates that feature distributions have the expected discriminability
    BEFORE Phase 4 classification:
      mean_hr      Wake > REM > N1/N2 > N3
      rmssd        N3 > REM > N2 > N1 > Wake
      lf_hf_ratio  Wake/N1 > N2 > N3/REM
      hf_power     N3 > REM > N2
      pnn20        N3 >> N2 > REM  [v2]
      perm_en      Wake/REM > N2 > N3  [v2]
      resp_rate_est REM > N2 > N3  [v2]
      rmssd_mean_rr_ratio  N3 > REM > Wake  [v2]
    """
    hrv_path = os.path.join(config["output_dir"], "phase3_hrv_features.csv")
    if not os.path.exists(hrv_path):
        logger.warning(
            "phase3_hrv_features.csv not found — PSG HRV summary skipped."
        )
        return

    hrv = pd.read_csv(hrv_path)
    n   = min(len(hrv), len(psg_hyp))
    hrv = hrv.iloc[:n].copy()
    hrv["psg_stage"]  = psg_hyp[:n]
    hrv["stage_name"] = hrv["psg_stage"].map(STAGE_NAMES)

    valid = hrv[hrv["psg_stage"] >= 0].copy()
    order = [s for s in ["Wake", "N1", "N2", "N3", "REM"]
             if s in valid["stage_name"].unique()]

    features = [
        ("mean_hr",             "Mean HR (bpm)",            "#e74c3c"),
        ("rmssd",               "RMSSD (ms)",               "#2ecc71"),
        ("lf_hf_ratio",         "LF/HF Ratio",              "#e67e22"),
        ("hf_power",            "HF Power",                 "#3498db"),
        ("sdnn",                "SDNN (ms)",                "#f39c12"),
        ("sampen",              "SampEn",                   "#9b59b6"),
        # Phase 3 v2 features
        ("pnn20",               "pNN20 (%) [v2]",           "#e67e22"),
        ("perm_en",             "PermEn (0–1) [v2]",        "#f97316"),
        ("resp_rate_est",       "Resp Rate (br/min) [v2]",  "#4fc3f7"),
        ("rmssd_mean_rr_ratio", "RMSSD/meanRR×100 [v2]",   "#27ae60"),
        ("hr_range",            "HR Range (bpm) [v2]",      "#1abc9c"),
        ("rr_iqr",              "RR IQR (ms) [v2]",         "#8e44ad"),
    ]
    present = [(f, t, c) for f, t, c in features if f in valid.columns]
    if not present:
        logger.warning("No HRV features found — skipping PSG HRV summary plot.")
        return

    n_feat = len(present)
    cols   = 4
    rows   = (n_feat + cols - 1) // cols
    stage_palette = {STAGE_NAMES[k]: v for k, v in STAGE_COLORS.items()}

    fig, axes = plt.subplots(rows, cols, figsize=(22, rows * 4))
    axes_flat = np.array(axes).flatten()
    fig.suptitle(
        f"HRV Features by PSG Gold-Standard Stage — {config['record_name']}\n"
        "(Validates Phase 3 feature discriminability before Phase 4 classification)",
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
        ax.set_xlabel("PSG Stage", fontsize=8)

    for ax in axes_flat[n_feat:]:
        ax.set_visible(False)

    try:
        plt.tight_layout()
    except Exception:
        pass
    out = os.path.join(config["output_dir"], "phase5_psg_hrv_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")
