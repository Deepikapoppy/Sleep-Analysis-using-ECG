"""
=============================================================================
phase6_comparison_evaluation/p6_plots_core.py
Step 3a — Core comparison plots.

Plots generated (all saved to config["output_dir"]):
  phase6_hypnogram_comparison.png  PSG vs ECG hypnogram + agreement timeline
  phase6_confusion_matrix.png      Confusion matrix (raw + normalised)
  phase6_per_stage_metrics.png     Precision / Recall / F1 bar chart
  phase6_duration_comparison.png   Stage duration: PSG vs ECG
  phase6_final_summary.png         Final results summary dashboard
=============================================================================
"""

import os
import logging
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from phase4_sleep_classification.p4_plots import STAGE_NAMES, STAGE_COLORS, _hypnogram_axis


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1: Hypnogram comparison + epoch agreement timeline
# ─────────────────────────────────────────────────────────────────────────────

def plot_hypnogram_comparison(ecg_hyp, psg_hyp, config, logger: logging.Logger) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(20, 10))
    fig.suptitle(f"Hypnogram Comparison – {config['record_name']}",
                 fontsize=14, fontweight='bold')

    _hypnogram_axis(axes[0], psg_hyp, 30, "PSG Gold Standard")
    _hypnogram_axis(axes[1], ecg_hyp, 30, "ECG-Derived (Rule-Based HRV)")

    t_hr  = np.arange(len(ecg_hyp)) * 30 / 3600
    agree = (ecg_hyp == psg_hyp).astype(int)
    axes[2].fill_between(t_hr, 0, agree, where=agree == 1, color='#2ecc71', alpha=0.7, label='Agree')
    axes[2].fill_between(t_hr, 0, 1,    where=agree == 0, color='#e74c3c', alpha=0.5, label='Disagree')
    pct = 100 * agree.mean()
    axes[2].set_title(f"Epoch-by-Epoch Agreement — {pct:.1f}% match",
                      fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Time (hours)")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Disagree", "Agree"])
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase6_hypnogram_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 2: Confusion matrix (raw + normalised)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, present_stages, config, logger: logging.Logger) -> None:
    stage_labels = [STAGE_NAMES.get(s, str(s)) for s in present_stages]
    row_sums     = cm.sum(axis=1, keepdims=True)
    cm_norm      = np.divide(cm.astype(float), row_sums, where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Confusion Matrix – {config['record_name']}",
                 fontsize=13, fontweight='bold')

    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0],
                xticklabels=stage_labels, yticklabels=stage_labels,
                cmap='Blues', linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Count'})
    axes[0].set_xlabel("Predicted (ECG)", fontsize=10)
    axes[0].set_ylabel("True (PSG)", fontsize=10)
    axes[0].set_title("Raw Counts")

    sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=axes[1],
                xticklabels=stage_labels, yticklabels=stage_labels,
                cmap='YlOrRd', linewidths=0.5, linecolor='gray', vmin=0, vmax=1,
                cbar_kws={'label': 'Recall (fraction)'})
    axes[1].set_xlabel("Predicted (ECG)", fontsize=10)
    axes[1].set_ylabel("True (PSG)", fontsize=10)
    axes[1].set_title("Normalised (Row = Recall)")

    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase6_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 3: Precision / Recall / F1 bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_bar(metrics, config, logger: logging.Logger) -> None:
    stages = list(metrics["per_stage"].keys())
    prec   = [metrics["per_stage"][s]["precision"] for s in stages]
    rec    = [metrics["per_stage"][s]["recall"]    for s in stages]
    f1     = [metrics["per_stage"][s]["f1"]        for s in stages]

    x, w = np.arange(len(stages)), 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, prec, w, label='Precision', color='#3498db', edgecolor='black', alpha=0.85)
    ax.bar(x,     rec,  w, label='Recall',    color='#2ecc71', edgecolor='black', alpha=0.85)
    ax.bar(x + w, f1,   w, label='F1-Score',  color='#e74c3c', edgecolor='black', alpha=0.85)

    ax.axhline(metrics["overall_acc"], color='navy', linestyle='--', linewidth=1.5,
               label=f'Accuracy={metrics["overall_acc"]*100:.1f}%')
    ax.axhline(metrics["cohen_kappa"], color='purple', linestyle=':', linewidth=1.5,
               label=f"κ={metrics['cohen_kappa']:.3f} ({metrics.get('kappa_interpretation','')})")

    ax.set_xticks(x);  ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylim([0, 1.1]);  ax.set_ylabel("Score")
    ax.set_title(f"Per-Stage Metrics — {config['record_name']}",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9);  ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase6_per_stage_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot: Stage duration comparison (PSG vs ECG)
# ─────────────────────────────────────────────────────────────────────────────

def plot_stage_duration_comparison(ecg_hyp, psg_hyp, config, logger: logging.Logger) -> None:
    order  = [0, 1, 2, 3, 4]
    names  = [STAGE_NAMES[s] for s in order]
    colors_list = [STAGE_COLORS[s] for s in order]

    ecg_min = [Counter(ecg_hyp)[s] * 30 / 60 for s in order]
    psg_min = [Counter(psg_hyp)[s] * 30 / 60 for s in order]

    x, w = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - w/2, psg_min, w, label='PSG (Gold Standard)',
                   color=colors_list, edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + w/2, ecg_min, w, label='ECG-Derived',
                   color=colors_list, edgecolor='black', alpha=0.5, hatch='//')

    for bar, v in zip(bars1, psg_min):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.0f}m', ha='center', fontsize=8)
    for bar, v in zip(bars2, ecg_min):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.0f}m', ha='center', fontsize=8)

    ax.set_xticks(x);  ax.set_xticklabels(names)
    ax.set_ylabel("Duration (minutes)")
    ax.set_title(f"Stage Duration: PSG vs ECG — {config['record_name']}",
                 fontsize=12, fontweight='bold')
    ax.legend();  ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase6_duration_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot: Final results summary dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_final_summary(metrics, ecg_hyp, psg_hyp, config, logger: logging.Logger,
                        pipeline_meta=None) -> None:
    """
    Final results summary dashboard: PSG hypnogram, ECG hypnogram, a
    performance/DWT-provenance text box, stage duration comparison and
    per-stage F1 bar chart.
    """
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"SLEEP ECG ANALYSIS — FINAL RESULTS SUMMARY\n{config['record_name']}",
        fontsize=15, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, :])
    _hypnogram_axis(ax0, psg_hyp, 30, "PSG Gold Standard")

    ax1 = fig.add_subplot(gs[1, :])
    _hypnogram_axis(ax1, ecg_hyp, 30, "ECG-Derived (Rule-Based HRV)")

    # ── Summary text box ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.axis('off')

    wavelet = (pipeline_meta or {}).get("phase1_dwt_wavelet", "db4")
    level   = (pipeline_meta or {}).get("phase1_dwt_level",   5)
    sqi_m   = (pipeline_meta or {}).get("phase1_mean_sqi",    "N/A")

    summary_text = (
        f"━━━━ PERFORMANCE SUMMARY ━━━━\n\n"
        f"Overall Accuracy : {metrics['overall_acc']*100:.1f}%\n"
        f"Cohen's Kappa    : {metrics['cohen_kappa']:.3f} "
        f"({metrics.get('kappa_interpretation', '')})\n"
        f"Valid Epochs     : {metrics['n_epochs']}\n\n"
        f"── Per-Stage F1 ──\n"
    )
    for name, v in metrics["per_stage"].items():
        summary_text += f"  {name:8s}: {v['f1']:.3f}\n"
    summary_text += (
        f"\n── DWT Provenance ──\n"
        f"  Wavelet : {wavelet}  Level : {level}\n"
        f"  SQI mean: {sqi_m}\n"
    )
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

    # ── Stage duration ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 1])
    order  = [0, 1, 2, 3, 4]
    names  = [STAGE_NAMES[s] for s in order]
    ecg_m  = [Counter(ecg_hyp).get(s, 0) * 30 / 60 for s in order]
    psg_m  = [Counter(psg_hyp).get(s, 0) * 30 / 60 for s in order]
    x, w   = np.arange(len(names)), 0.35
    ax3.bar(x - w/2, psg_m, w, label='PSG', color=[STAGE_COLORS[s] for s in order],
            edgecolor='black', alpha=0.9)
    ax3.bar(x + w/2, ecg_m, w, label='ECG', color=[STAGE_COLORS[s] for s in order],
            edgecolor='black', alpha=0.5, hatch='//')
    ax3.set_xticks(x);  ax3.set_xticklabels(names, fontsize=9)
    ax3.set_ylabel("Minutes");  ax3.set_title("Stage Duration")
    ax3.legend(fontsize=8);  ax3.grid(True, axis='y', alpha=0.3)

    # ── F1 per stage ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 2])
    f1_stages = list(metrics["per_stage"].keys())
    f1_vals   = [metrics["per_stage"][s]["f1"] for s in f1_stages]
    f1_colors = [STAGE_COLORS[i] for i in range(len(f1_stages))]
    bars = ax4.barh(f1_stages, f1_vals, color=f1_colors, edgecolor='black', alpha=0.85)
    ax4.axvline(0.5, color='red', linestyle='--', linewidth=1.0, label='F1=0.5')
    for bar, v in zip(bars, f1_vals):
        ax4.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{v:.3f}', va='center', fontsize=9)
    ax4.set_xlim([0, 1.15])
    ax4.set_title("F1-Score per Stage")
    ax4.legend(fontsize=8);  ax4.grid(True, axis='x', alpha=0.3)

    out = os.path.join(config["output_dir"], "phase6_final_summary.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved -> {out}")
