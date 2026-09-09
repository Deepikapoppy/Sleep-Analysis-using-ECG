"""
=============================================================================
phase6_comparison_evaluation/p6_plots_hrv.py
Step 3b — HRV / signal-quality / detection-method diagnostic plots.

Plots generated (all saved to config["output_dir"]):
  phase6_hrv_by_psg_stage.png       HRV feature boxplots by PSG stage
                                     (core 6 + Phase 3 v2 features)
  phase6_sqi_agreement.png          SQI vs epoch agreement analysis
  phase6_v2_feature_comparison.png  v2 features: PSG stage vs ECG stage
  phase6_detection_method_accuracy.png  Per R-peak method accuracy

All four are best-effort: they skip gracefully and log a message if the
required upstream file/columns are not available.
=============================================================================
"""

import os
import logging

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase4_sleep_classification.p4_plots import STAGE_NAMES, STAGE_COLORS


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 5: HRV boxplots by PSG stage (core + Phase 3 v2 features)
# ─────────────────────────────────────────────────────────────────────────────

def plot_hrv_by_psg_stage(hrv, psg_hyp, config, logger: logging.Logger) -> None:
    """
    12-panel boxplot of HRV features grouped by PSG stage.

    Core features:
      mean_hr, rmssd, lf_hf_ratio, hf_power, sdnn, sampen

    Phase 3 v2 additions:
      pnn20              — more sensitive N3 marker than pNN50 in 30-s epochs
      perm_en            — complexity; high in Wake/REM, low in N3
      resp_rate_est      — N3 ≈ 12–16, REM ≈ 15–22 br/min
      rmssd_mean_rr_ratio— HR-independent vagal index; high in N3
      hr_range           — intra-epoch HR swing; elevated in REM
      rr_iqr             — robust beat-to-beat spread; moderate in N2

    Purpose: validates that the Phase 5 (PSG) labels and the underlying
    HRV features genuinely differ across stages.
    """
    features = [
        # ── Core ────────────────────────────────────────────────────────
        ("mean_hr",             "Mean HR (bpm)"),
        ("rmssd",               "RMSSD (ms)"),
        ("lf_hf_ratio",         "LF/HF Ratio"),
        ("hf_power",            "HF Power"),
        ("sdnn",                "SDNN (ms)"),
        ("sampen",              "SampEn"),
        # ── Phase 3 v2 ──────────────────────────────────────────────────
        ("pnn20",               "pNN20 (%) [v2 — N3 marker]"),
        ("perm_en",             "PermEn (0–1) [v2 — complexity]"),
        ("resp_rate_est",       "Resp Rate (br/min) [v2]"),
        ("rmssd_mean_rr_ratio", "RMSSD/meanRR×100 (%) [v2]"),
        ("hr_range",            "HR Range (bpm) [v2 — REM marker]"),
        ("rr_iqr",              "RR IQR (ms) [v2 — N2 spread]"),
    ]

    df_merged = hrv.copy()
    df_merged["psg_stage"]  = psg_hyp[:len(hrv)]
    df_merged["stage_name"] = df_merged["psg_stage"].map(STAGE_NAMES)

    valid         = df_merged[df_merged["psg_stage"] >= 0].copy()
    order         = ["Wake", "N1", "N2", "N3", "REM"]
    present_order = [s for s in order if s in valid["stage_name"].unique()]
    stage_palette = {STAGE_NAMES[k]: v for k, v in STAGE_COLORS.items()}

    present = [(f, t) for f, t in features if f in valid.columns]
    n_feat  = len(present)
    if n_feat == 0:
        logger.warning("plot_hrv_by_psg_stage: no matching HRV columns found — skipped.")
        return
    cols    = 4
    rows    = (n_feat + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(22, rows * 4))
    fig.suptitle(
        f"HRV Features by PSG Sleep Stage — {config['record_name']}\n"
        "(Core + Phase 3 v2 features — validates Phase 5 PSG labelling)",
        fontsize=13, fontweight='bold'
    )
    axes_flat = np.array(axes).flatten()

    for ax, (feat, title) in zip(axes_flat, present):
        sub  = valid[["stage_name", feat]].dropna()
        grps = [sub[sub["stage_name"] == s][feat].values for s in present_order
                if len(sub[sub["stage_name"] == s]) > 0]
        lbls = [f"{s}\n(n={len(sub[sub['stage_name']==s])})"
                for s in present_order if len(sub[sub["stage_name"]==s]) > 0]
        if not grps:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(grps, labels=lbls, patch_artist=True, notch=False,
                        medianprops={'color': 'black', 'linewidth': 2})
        for patch, s in zip(bp['boxes'], [s for s in present_order
                                           if len(sub[sub["stage_name"]==s]) > 0]):
            patch.set_facecolor(stage_palette.get(s, '#95a5a6'))
            patch.set_alpha(0.8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.25)
        ax.set_xlabel("PSG Stage", fontsize=8)

    for ax in axes_flat[n_feat:]:
        ax.set_visible(False)

    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase6_hrv_by_psg_stage.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 6: SQI vs Agreement analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_sqi_agreement_analysis(ecg_hyp, psg_hyp, config, logger: logging.Logger) -> None:
    """
    Quantifies the relationship between Phase 1 signal quality (SQI) and
    per-epoch agreement between ECG-derived and PSG labels.

    Hypothesis: low-SQI epochs (poor ECG signal) should produce more
    misclassifications because R-peak detection and HRV feature extraction
    are less reliable on noisy segments.

    Shows:
      Panel 1 — Agreement rate vs SQI bin (bar chart)
      Panel 2 — SQI distribution split by agree / disagree (histogram)
    """
    sqi_path = os.path.join(config["output_dir"], "phase1_sqi.csv")
    if not os.path.exists(sqi_path):
        logger.info("SQI-agreement plot skipped (phase1_sqi.csv absent).")
        return

    sqi_df = pd.read_csv(sqi_path)
    n      = min(len(ecg_hyp), len(psg_hyp), len(sqi_df))

    sqi    = sqi_df["overall_sqi"].values[:n]
    agree  = (ecg_hyp[:n] == psg_hyp[:n]).astype(int)
    valid  = psg_hyp[:n] >= 0

    sqi_v   = sqi[valid]
    agree_v = agree[valid]

    # ── Bin by SQI ────────────────────────────────────────────────────────
    bins   = np.linspace(0, 1, 11)   # 10 bins of 0.10 width
    labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(bins[:-1], bins[1:])]
    bin_idx = np.digitize(sqi_v, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(labels) - 1)

    bin_acc  = []
    bin_cnt  = []
    for b in range(len(labels)):
        mask     = bin_idx == b
        bin_cnt.append(mask.sum())
        bin_acc.append(float(agree_v[mask].mean()) if mask.sum() > 0 else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Signal Quality (SQI) vs Epoch Agreement — {config['record_name']}",
        fontsize=13, fontweight='bold'
    )

    # Panel 1: Agreement rate per SQI bin
    bar_colors = ['#e74c3c' if (v is not np.nan and v < 0.5) else '#2ecc71'
                  for v in bin_acc]
    x          = np.arange(len(labels))
    axes[0].bar(x, [v if not np.isnan(v) else 0 for v in bin_acc],
                color=bar_colors, edgecolor='black', alpha=0.8)
    axes[0].axhline(np.nanmean(bin_acc), color='navy', linestyle='--',
                    linewidth=1.5, label=f'Mean={np.nanmean(bin_acc):.2f}')

    # Annotate with counts
    for xi, (acc_v, cnt) in enumerate(zip(bin_acc, bin_cnt)):
        if not np.isnan(acc_v):
            axes[0].text(xi, acc_v + 0.01, f'n={cnt}', ha='center',
                         fontsize=7, rotation=45)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, fontsize=8)
    axes[0].set_ylim([0, 1.15])
    axes[0].set_xlabel("SQI Bin")
    axes[0].set_ylabel("Agreement Rate")
    axes[0].set_title("Epoch Agreement Rate per SQI Bin")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Panel 2: SQI histogram — agree vs disagree
    sqi_agree    = sqi_v[agree_v == 1]
    sqi_disagree = sqi_v[agree_v == 0]
    axes[1].hist(sqi_agree,    bins=20, color='#2ecc71', alpha=0.65,
                 label=f'Agree (n={len(sqi_agree)})',    edgecolor='black')
    axes[1].hist(sqi_disagree, bins=20, color='#e74c3c', alpha=0.65,
                 label=f'Disagree (n={len(sqi_disagree)})', edgecolor='black')
    axes[1].axvline(0.50, color='black', linestyle='--', linewidth=1.2,
                    alpha=0.7, label='SQI threshold 0.50')
    axes[1].set_xlabel("Overall SQI")
    axes[1].set_ylabel("Epoch Count")
    axes[1].set_title("SQI Distribution: Agree vs Disagree Epochs")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase6_sqi_agreement.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 7: v2 feature comparison — PSG stage vs ECG-classified stage
# ─────────────────────────────────────────────────────────────────────────────

def plot_v2_feature_comparison(hrv, ecg_hyp, psg_hyp, config, logger: logging.Logger) -> None:
    """
    Side-by-side boxplots of Phase 3 v2 features grouped first by PSG stage
    (ground truth, left column) and then by ECG-classified stage
    (right column).

    Purpose: identify which stage confusions are caused by overlapping
    feature distributions (e.g., if N2 and N3 look similar in pnn20, that
    explains N2<->N3 misclassifications shown in the confusion matrix).

    Features shown: pnn20, perm_en, rmssd_mean_rr_ratio, hr_range
    """
    v2_features = [
        ("pnn20",               "pNN20 (%) — N3 marker"),
        ("perm_en",             "PermEn (0–1) — complexity"),
        ("rmssd_mean_rr_ratio", "RMSSD/meanRR×100 — vagal index"),
        ("hr_range",            "HR Range (bpm) — REM marker"),
    ]
    present = [(f, t) for f, t in v2_features if f in hrv.columns]
    if not present:
        logger.info("v2 feature comparison skipped (no v2 features in CSV).")
        return

    n    = min(len(hrv), len(ecg_hyp), len(psg_hyp))
    df   = hrv.iloc[:n].copy()
    df["psg_stage"]  = psg_hyp[:n]
    df["ecg_stage"]  = ecg_hyp[:n]
    df["psg_name"]   = df["psg_stage"].map(STAGE_NAMES)
    df["ecg_name"]   = df["ecg_stage"].map(STAGE_NAMES)

    order         = ["Wake", "N1", "N2", "N3", "REM"]
    stage_palette = {STAGE_NAMES[k]: v for k, v in STAGE_COLORS.items()}

    n_feat = len(present)
    fig, axes = plt.subplots(n_feat, 2, figsize=(18, n_feat * 4))
    if n_feat == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f"Phase 3 v2 Features: PSG (ground truth) vs ECG Classified — "
        f"{config['record_name']}\n"
        "(Overlapping distributions explain specific stage confusions)",
        fontsize=12, fontweight='bold'
    )
    axes[0, 0].set_title("PSG Gold-Standard Stage", fontsize=11, fontweight='bold',
                          color='#27ae60')
    axes[0, 1].set_title("ECG-Classified Stage (Phase 4)", fontsize=11, fontweight='bold',
                          color='#2980b9')

    for ri, (feat, title) in enumerate(present):
        for ci, (stage_col, stage_name_col) in enumerate([("psg_stage", "psg_name"),
                                                           ("ecg_stage", "ecg_name")]):
            ax = axes[ri, ci]
            sub = df[[stage_col, stage_name_col, feat]].dropna(subset=[stage_col, feat])
            name_col = stage_name_col

            grps  = [sub[sub[name_col] == s][feat].values
                     for s in order if s in sub[name_col].values]
            lbls  = [f"{s}\n(n={len(sub[sub[name_col]==s])})"
                     for s in order if s in sub[name_col].values]
            if not grps:
                ax.set_visible(False)
                continue
            bp = ax.boxplot(grps, labels=lbls, patch_artist=True, notch=False,
                            medianprops=dict(color='black', linewidth=1.8))
            for patch, s in zip(bp['boxes'],
                                 [s for s in order if s in sub[name_col].values]):
                patch.set_facecolor(stage_palette.get(s, '#95a5a6'))
                patch.set_alpha(0.75)
            if ci == 0:
                ax.set_ylabel(title, fontsize=9)
            ax.grid(True, axis='y', alpha=0.2)
            ax.set_xlabel("Stage", fontsize=8)

    try:
        plt.tight_layout()
    except Exception:
        pass
    out = os.path.join(config["output_dir"], "phase6_v2_feature_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 8: R-peak detection method vs classification accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_detection_method_accuracy(ecg_hyp, psg_hyp, config, logger: logging.Logger) -> None:
    """
    Per-method accuracy bar chart using phase2_rr_epoch_summary.csv.
    Shows whether epochs processed by fallback detectors (scipy, biosppy)
    are harder to classify correctly — informing pipeline quality decisions.
    """
    p2_path = os.path.join(config["output_dir"], "phase2_rr_epoch_summary.csv")
    if not os.path.exists(p2_path):
        logger.info("Detection-method accuracy plot skipped (phase2_rr_epoch_summary.csv absent).")
        return

    df2 = pd.read_csv(p2_path)
    if "method" not in df2.columns:
        logger.info("Detection-method accuracy plot skipped (no 'method' column).")
        return

    n = min(len(df2), len(ecg_hyp), len(psg_hyp))
    df2 = df2.iloc[:n].copy()
    df2["ecg_stage"] = ecg_hyp[:n]
    df2["psg_stage"] = psg_hyp[:n]
    df2["agree"]     = (df2["ecg_stage"] == df2["psg_stage"]) & (df2["psg_stage"] >= 0)

    # Bucket granular neurokit names -> "neurokit2"
    df2["method_bucket"] = df2["method"].apply(
        lambda m: "neurokit2" if str(m).startswith("neurokit2") else str(m)
    )

    summary = (df2.groupby("method_bucket")["agree"]
                  .agg(accuracy="mean", n_epochs="count")
                  .reset_index()
                  .sort_values("accuracy", ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"R-Peak Detection Method vs Classification Accuracy — {config['record_name']}",
        fontsize=13, fontweight='bold'
    )

    colors = ['#2ecc71' if 'neurokit' in m
              else '#3498db' if m == 'wfdb_xqrs'
              else '#e67e22' if m == 'biosppy_hamilton'
              else '#e74c3c'
              for m in summary["method_bucket"]]

    # Panel 1: Accuracy per method
    bars = axes[0].bar(summary["method_bucket"], summary["accuracy"],
                       color=colors, edgecolor='black', alpha=0.85)
    for bar, v in zip(bars, summary["accuracy"]):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005, f'{v:.2f}',
                     ha='center', fontsize=9)
    axes[0].axhline(df2["agree"].mean(), color='navy', linestyle='--',
                    linewidth=1.5, label=f'Overall={df2["agree"].mean():.3f}')
    axes[0].set_ylim([0, 1.1])
    axes[0].set_xlabel("Detection Method")
    axes[0].set_ylabel("Epoch Accuracy")
    axes[0].set_title("Accuracy by R-Peak Detection Method")
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Panel 2: Epoch count per method (pie)
    axes[1].pie(summary["n_epochs"], labels=summary["method_bucket"],
                colors=colors, autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'black', 'linewidth': 0.8})
    axes[1].set_title("Epoch Count per Detection Method")

    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase6_detection_method_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved -> {out}")
