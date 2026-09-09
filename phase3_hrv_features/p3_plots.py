"""
=============================================================================
phase3_hrv_features/p3_plots.py
Diagnostic plots for Phase 3 — one plot per tier + combined overview.

Plots generated:
  phase3_sqi_gate.png        SQI scores over night
  phase3_tier1_primary.png   18 T1 features
  phase3_tier2_confirm.png   19 T2 features
  phase3_tier3_resolver.png  Selected T3 features
  phase3_all_features.png    Full 71-feature overview
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _smooth(series, w: int = 10):
    return pd.Series(series).rolling(w, center=True, min_periods=1).mean().values


def _t_hr(df: pd.DataFrame) -> np.ndarray:
    return df["epoch_idx"].values * 30.0 / 3600.0


def _plot_panel(df, feature_list, title, out_path, logger, ncols: int = 3):
    """Generic multi-panel plot."""
    avail = [(f, lbl, col) for f, lbl, col in feature_list if f in df.columns]
    if not avail:
        logger.info(f"  Plot skipped (no features available): {out_path}")
        return

    t = _t_hr(df)
    rows = (len(avail) + ncols - 1) // ncols
    fig, axes = plt.subplots(rows, ncols,
                              figsize=(6 * ncols, 3 * rows), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    axes_flat = np.array(axes).flatten()

    for ax, (feat, lbl, col) in zip(axes_flat, avail):
        vals = df[feat].values.astype(float)
        ax.plot(t, vals,          color=col, alpha=0.28, linewidth=0.5)
        ax.plot(t, _smooth(vals), color=col, linewidth=1.4)
        ax.set_title(lbl, fontsize=8, fontweight="bold")
        ax.set_xlabel("Time (hr)", fontsize=7)
        ax.grid(True, alpha=0.15)

    for ax in axes_flat[len(avail):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

def plot_sqi_gate(df: pd.DataFrame, config: dict, logger: logging.Logger):
    features = [
        ("overall_sqi", "Overall SQI",   "#2c3e50"),
        ("kSQI_norm",   "kSQI (norm)",   "#e74c3c"),
        ("pSQI",        "pSQI",          "#3498db"),
        ("basSQI",      "basSQI",        "#2ecc71"),
    ]
    # Add SQI threshold line overlay
    t = _t_hr(df)
    avail = [(f, lbl, col) for f, lbl, col in features if f in df.columns]
    if not avail:
        logger.info("SQI gate plot skipped — no SQI columns.")
        return

    fig, axes = plt.subplots(len(avail), 1,
                              figsize=(18, 3 * len(avail)), sharex=True)
    if len(avail) == 1:
        axes = [axes]
    fig.suptitle(f"SQI Gate — {config['record_name']}",
                 fontsize=13, fontweight="bold")

    for ax, (feat, lbl, col) in zip(axes, avail):
        vals = df[feat].values.astype(float)
        ax.plot(t, vals,          color=col, alpha=0.4,  linewidth=0.6)
        ax.plot(t, _smooth(vals), color=col, linewidth=1.4)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.5, label="Threshold 0.50")
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (hours)")
    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase3_sqi_gate.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


def plot_tier1(df: pd.DataFrame, config: dict, logger: logging.Logger):
    features = [
        ("mean_hr",             "Mean HR (bpm)",          "#e74c3c"),
        ("mean_rr",             "Mean RR (ms)",           "#3498db"),
        ("rmssd",               "RMSSD (ms)",             "#2ecc71"),
        ("pnn20",               "pNN20 (%)",              "#e67e22"),
        ("sdnn",                "SDNN (ms)",              "#9b59b6"),
        ("hr_range",            "HR Range (bpm)",         "#1abc9c"),
        ("rr_autocorr_lag1",    "RR Autocorr Lag-1",      "#f39c12"),
        ("hf_power",            "HF Power",               "#1abc9c"),
        ("lf_hf_ratio",         "LF/HF Ratio",            "#e74c3c"),
        ("lf_power",            "LF Power",               "#e67e22"),
        ("resp_rate_est",       "Resp Rate (br/min)",     "#4fc3f7"),
        ("perm_en",             "PermEn",                 "#f97316"),
        ("dfa_alpha1",          "DFA α1",                 "#e67e22"),
        ("sd1",                 "SD1 (ms)",               "#1abc9c"),
        ("sd1_sd2_ratio",       "SD1/SD2",                "#2ecc71"),
        ("sampen",              "SampEn",                 "#34495e"),
        ("rmssd_mean_rr_ratio", "RMSSD/meanRR×100",       "#27ae60"),
        ("sdnn_rmssd_ratio",    "SDNN/RMSSD",             "#2c3e50"),
    ]
    _plot_panel(df, features,
                f"T1 Primary Screamer (18 features) — {config['record_name']}",
                os.path.join(config["output_dir"], "phase3_tier1_primary.png"),
                logger)


def plot_tier2(df: pd.DataFrame, config: dict, logger: logging.Logger):
    features = [
        ("median_rr",          "Median RR (ms)",         "#f39c12"),
        ("pnn50",              "pNN50 (%)",              "#9b59b6"),
        ("cv",                 "CV (%)",                 "#95a5a6"),
        ("rr_iqr",             "RR IQR (ms)",            "#8e44ad"),
        ("min_hr",             "Min HR (bpm)",           "#2980b9"),
        ("max_hr",             "Max HR (bpm)",           "#c0392b"),
        ("hf_nu",              "HF n.u. (%)",            "#3498db"),
        ("lf_nu",              "LF n.u. (%)",            "#9b59b6"),
        ("peak_hf_freq",       "Peak HF Freq (Hz)",      "#16a085"),
        ("total_power",        "Total Power",            "#7f8c8d"),
        ("sd2",                "SD2 (ms)",               "#e74c3c"),
        ("dfa_alpha2",         "DFA α2",                 "#e74c3c"),
        ("prsa_dc",            "PRSA DC",                "#2ecc71"),
        ("porta_asymmetry",    "Porta Asymmetry (%)",    "#e67e22"),
        ("edr_breath_rate",    "EDR Breath Rate (br/min)","#e74c3c"),
        ("edr_regularity",     "EDR Regularity (CV%)",   "#9b59b6"),
        ("edr_baseline_wander","EDR Baseline Wander",    "#2980b9"),
        ("sleep_cycle_pos",    "Sleep Cycle Position",   "#1abc9c"),
        ("hr_slope_epoch",     "HR Slope (bpm/min)",     "#c0392b"),
    ]
    _plot_panel(df, features,
                f"T2 Confirmatory (19 features) — {config['record_name']}",
                os.path.join(config["output_dir"], "phase3_tier2_confirm.png"),
                logger)


def plot_tier3(df: pd.DataFrame, config: dict, logger: logging.Logger):
    features = [
        ("guzik_asymmetry",    "Guzik Asymmetry",        "#7f77dd"),
        ("mse_scale2",         "MSE Scale 2",            "#534AB7"),
        ("mse_scale4",         "MSE Scale 4",            "#378ADD"),
        ("higuchi_fd",         "Higuchi FD",             "#e74c3c"),
        ("fuzzy_entropy",      "Fuzzy Entropy",          "#e67e22"),
        ("recurrence_rate",    "Recurrence Rate",        "#2ecc71"),
        ("rr_entropy_rate",    "RR Entropy Rate",        "#9b59b6"),
        ("crc_phase_sync",     "CRC Phase Sync (PLV)",   "#1D9E75"),
        ("crc_coherence_hf",   "CRC HF Coherence",       "#085041"),
        ("r_amplitude_cv",     "R-Amp CV",               "#e74c3c"),
        ("qrs_duration",       "QRS Duration (ms)",      "#3498db"),
        ("pr_interval",        "PR Interval (ms)",       "#2ecc71"),
        ("rr_spectral_entropy","RR Spectral Entropy",    "#9b59b6"),
        ("rr_triangular_index","RR Triangular Index",    "#f39c12"),
        ("edr_breath_depth",   "EDR Breath Depth",       "#16a085"),
    ]
    _plot_panel(df, features,
                f"T3 Resolver (selected features) — {config['record_name']}",
                os.path.join(config["output_dir"], "phase3_tier3_resolver.png"),
                logger, ncols=3)


def plot_all_features(df: pd.DataFrame, config: dict, logger: logging.Logger):
    """Compact overview of ALL 71 features in one figure."""
    all_feat_cols = [c for c in df.columns
                     if c not in ("epoch_idx", "is_bad", "n_rr",
                                  "detection_method", "sqi_flag")
                     and not c.endswith("_zscore")]
    t    = _t_hr(df)
    ncols = 4
    nrows = (len(all_feat_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 2.5 * nrows), sharex=True)
    fig.suptitle(f"All Features Overview ({len(all_feat_cols)}) — "
                 f"{config['record_name']}", fontsize=12, fontweight="bold")
    axes_flat = np.array(axes).flatten()

    for ax, feat in zip(axes_flat, all_feat_cols):
        vals = pd.to_numeric(df[feat], errors="coerce").values.astype(float)
        ax.plot(t, vals, color="#555", alpha=0.3, linewidth=0.4)
        ax.plot(t, _smooth(vals, w=7), color="#222", linewidth=0.8)
        ax.set_title(feat, fontsize=6, fontweight="bold")
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.1)

    for ax in axes_flat[len(all_feat_cols):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = os.path.join(config["output_dir"], "phase3_all_features.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")
