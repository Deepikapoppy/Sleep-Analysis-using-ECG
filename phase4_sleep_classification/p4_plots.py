"""
=============================================================================
phase4_sleep_classification/p4_plots.py
Diagnostic plots for Phase 4 — Sleep Stage Classification.

Plots generated (all saved to config["output_dir"]):
  phase4_stage_scores.png               Per-stage score traces (0–1)
  phase4_ecg_hypnogram.png              Raw + smoothed hypnogram
  phase4_stage_duration_raw.png         Stage duration bar + pie (raw)
  phase4_stage_duration_smoothed.png    Stage duration bar + pie (smoothed)
  phase4_v2_stage_contributions.png     Phase 3 v2 feature distributions
                                        by classified stage (boxplots)
=============================================================================
"""

import os
import json
import logging
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Stage constants ───────────────────────────────────────────────────────────
STAGE_NAMES  = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {
    0: "#e74c3c",   # Wake  — red
    1: "#f39c12",   # N1    — orange
    2: "#3498db",   # N2    — blue
    3: "#2ecc71",   # N3    — green
    4: "#9b59b6",   # REM   — purple
}


# ─────────────────────────────────────────────────────────────────────────────
#  Score-matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_matrix(scores: np.ndarray,
                      config: dict,
                      logger: logging.Logger) -> None:
    """Five stacked subplots — one per stage — showing score traces over night."""
    t_ep        = np.arange(len(scores)) * 30 / 3600.0
    stage_names = ["Wake", "N1", "N2", "N3", "REM"]
    colors      = [STAGE_COLORS[i] for i in range(5)]

    fig, axes = plt.subplots(5, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(
        f"Rule-Based Stage Scores (0–1) — {config['record_name']}",
        fontsize=13, fontweight="bold",
    )
    for ax, name, col, i in zip(axes, stage_names, colors, range(5)):
        ax.fill_between(t_ep, 0, scores[:, i], alpha=0.5, color=col)
        ax.plot(t_ep, scores[:, i], color=col, linewidth=0.8)
        ax.set_ylabel(name, fontsize=9, rotation=0, labelpad=40, va="center")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Time (hours)")
    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase4_stage_scores.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Hypnogram axis helper
# ─────────────────────────────────────────────────────────────────────────────

def _hypnogram_axis(ax, stages, epoch_sec=30, title="",
                    recording_start_time=None) -> None:
    """
    Draw a hypnogram on `ax`.
    If recording_start_time ('HH:MM:SS') is provided, x-axis labels show real
    clock times; otherwise, relative time from recording start is used.
    """
    import datetime

    n_epochs = len(stages)
    t_sec    = np.arange(n_epochs) * epoch_sec
    t_hr     = t_sec / 3600.0

    start_sec = None
    if recording_start_time:
        try:
            parts     = [int(x) for x in str(recording_start_time).split(":")]
            h         = parts[0]
            m         = parts[1] if len(parts) > 1 else 0
            s         = parts[2] if len(parts) > 2 else 0
            start_sec = h * 3600 + m * 60 + s
        except Exception:
            start_sec = None

    for i, st in enumerate(stages):
        col = STAGE_COLORS.get(int(st), "#95a5a6")
        ax.fill_between(
            [t_hr[i], t_hr[i] + epoch_sec / 3600],
            [st, st], [st + 0.95, st + 0.95],
            color=col, alpha=0.8,
        )

    ax.step(t_hr, stages, where="post", color="black", linewidth=1.2)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["Wake", "N1", "N2", "N3", "REM"], fontsize=9)
    ax.set_ylim([-0.3, 5.2])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)

    n_ticks  = min(9, n_epochs)
    tick_idx = np.linspace(0, n_epochs - 1, n_ticks, dtype=int)
    tick_pos = t_hr[tick_idx]

    if start_sec is not None:
        def _fmt_clock(offset_sec):
            abs_s = int(start_sec + offset_sec) % 86400
            return f"{abs_s // 3600:02d}:{(abs_s % 3600) // 60:02d}"
        tick_labels = [_fmt_clock(t_sec[i]) for i in tick_idx]
        ax.set_xlabel("Clock time (HH:MM)", fontsize=9)
    else:
        tick_labels = [
            f"{int(t_sec[i]) // 3600}h {(int(t_sec[i]) % 3600) // 60:02d}m"
            for i in tick_idx
        ]
        ax.set_xlabel("Time from recording start", fontsize=9)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")
    legend_patches = [
        mpatches.Patch(color=STAGE_COLORS[i], label=["Wake","N1","N2","N3","REM"][i])
        for i in range(5)
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=5)


# ─────────────────────────────────────────────────────────────────────────────
#  Hypnogram plot (raw + smoothed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ecg_hypnogram(raw_stages, smooth_stages,
                       config: dict, logger: logging.Logger) -> None:
    """Two-panel hypnogram: raw classification on top, smoothed below."""
    recording_start_time = None
    try:
        meta1_path = os.path.join(config["output_dir"], "phase1_meta.json")
        if os.path.exists(meta1_path):
            with open(meta1_path) as f:
                m = json.load(f)
            recording_start_time = m.get("recording_start_time")
            if recording_start_time:
                logger.info(
                    f"Hypnogram: real clock timestamps "
                    f"(start={recording_start_time})"
                )
            else:
                logger.info(
                    "Hypnogram: recording_start_time not found — "
                    "using relative timestamps"
                )
    except Exception:
        pass

    fig, axes = plt.subplots(2, 1, figsize=(18, 7))
    fig.suptitle(
        f"ECG-Derived Hypnogram — {config['record_name']}",
        fontsize=14, fontweight="bold",
    )
    _hypnogram_axis(axes[0], raw_stages,    epoch_sec=30,
                    title="Raw Classification",
                    recording_start_time=recording_start_time)
    _hypnogram_axis(axes[1], smooth_stages, epoch_sec=30,
                    title="Smoothed (Median Filter, K=5)",
                    recording_start_time=recording_start_time)
    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase4_ecg_hypnogram.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Stage duration bar + pie
# ─────────────────────────────────────────────────────────────────────────────

def plot_stage_duration(stages, label: str,
                        config: dict, logger: logging.Logger) -> None:
    counts     = Counter(stages)
    labels_txt = [STAGE_NAMES[s] for s in sorted(counts)]
    values_min = [counts[s] * 30 / 60 for s in sorted(counts)]
    colors_lst = [STAGE_COLORS[s] for s in sorted(counts)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"ECG Sleep Stage Duration ({label}) — {config['record_name']}",
        fontsize=13, fontweight="bold",
    )
    bars = axes[0].bar(labels_txt, values_min,
                       color=colors_lst, edgecolor="black")
    for bar, v in zip(bars, values_min):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{v:.0f} min", ha="center", fontsize=9,
        )
    axes[0].set_ylabel("Duration (minutes)")
    axes[0].set_title("Time in Each Stage")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].pie(values_min, labels=labels_txt, colors=colors_lst,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "black", "linewidth": 0.8})
    axes[1].set_title("Stage Proportion (%)")

    plt.tight_layout()
    tag = label.lower().replace(" ", "_")
    out = os.path.join(config["output_dir"],
                       f"phase4_stage_duration_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3 v2 feature distributions by classified stage
# ─────────────────────────────────────────────────────────────────────────────

def plot_v2_stage_contributions(df: pd.DataFrame,
                                 smooth_stages,
                                 config: dict,
                                 logger: logging.Logger) -> None:
    """
    Boxplot of Phase 3 v2 features split by classified sleep stage.
    Confirms that the features are discriminative and correctly used in scoring.
    """
    v2_features = [
        ("pnn20",               "pNN20 (%)  — N3 marker",         "#e67e22"),
        ("hr_range",            "HR Range (bpm)  — REM marker",   "#1abc9c"),
        ("resp_rate_est",       "Resp Rate (br/min)  — HF peak",  "#4fc3f7"),
        ("perm_en",             "PermEn (0–1)  — complexity",     "#f97316"),
        ("rmssd_mean_rr_ratio", "RMSSD/meanRR×100  — vagal",      "#27ae60"),
        ("rr_iqr",              "RR IQR (ms)  — spread",          "#8e44ad"),
    ]
    present = [(f, t, c) for f, t, c in v2_features if f in df.columns]
    if not present:
        logger.warning("No Phase 3 v2 features in CSV — skipping v2 stage plot.")
        return

    stage_labels = ["Wake", "N1", "N2", "N3", "REM"]
    n_feat  = len(present)
    n_cols  = 3
    n_rows  = (n_feat + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes_flat = axes.flatten() if n_feat > 1 else [axes]
    fig.suptitle(
        f"Phase 3 v2 Feature Distributions by Sleep Stage — "
        f"{config['record_name']}",
        fontsize=13, fontweight="bold",
    )

    stage_col = pd.Series(smooth_stages[: len(df)], name="stage")
    df_plot   = df.copy().reset_index(drop=True)
    df_plot["stage"] = stage_col

    stage_hex = {0: "#e74c3c", 1: "#f39c12", 2: "#3498db",
                 3: "#2ecc71", 4: "#9b59b6"}

    for ax, (feat, title, col) in zip(axes_flat, present):
        groups = []
        x_pos  = []
        x_lbl  = []
        for sid, slabel in enumerate(stage_labels):
            vals = df_plot.loc[df_plot["stage"] == sid, feat].dropna().values
            if len(vals) >= 3:
                groups.append(vals)
                x_pos.append(sid)
                x_lbl.append(f"{slabel}\n(n={len(vals)})")

        if not groups:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(groups, positions=x_pos, widths=0.55,
                        patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.8))
        for patch, sid in zip(bp["boxes"], x_pos):
            patch.set_facecolor(stage_hex.get(sid, col))
            patch.set_alpha(0.6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_lbl, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)

    for ax in axes_flat[n_feat:]:
        ax.set_visible(False)

    try:
        plt.tight_layout()
    except Exception:
        pass
    out = os.path.join(config["output_dir"],
                       "phase4_v2_stage_contributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {out}")
