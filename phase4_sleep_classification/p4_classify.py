"""
=============================================================================
phase4_sleep_classification/p4_classify.py
Step 3 — Epoch classification and temporal smoothing.

  classify_epochs   : applies all stage scorers, time-of-night priors,
                      FIX-5b adaptive Wake boost, softmax → argmax.
  smooth_hypnogram  : two-pass smoothing (median filter + min-run merger).

RC8 fix history (this file)
────────────────────────────
  RC8a — Time-of-night priors were computed every loop iteration but
         discarded (never applied to `scores`). REM's late-night 1.30x
         boost and N3's late-night 0.35x suppression were dead code.
         Fixed: priors now built into an (n_epochs, 5) array and
         multiplied into `scores` before the softmax.
  RC8b — PROTECTED set in smooth_hypnogram was empty despite the comment
         claiming N3/REM were protected from the min-run merger. Short
         REM/N3 bursts were being silently merged into neighboring N2.
         Fixed: PROTECTED = {3, 4} (N3, REM).
=============================================================================
"""

import logging
from collections import Counter

import numpy as np
import pandas as pd

from phase4_sleep_classification.p4_scoring import (
    score_wake, score_n1, score_n2, score_n3_gated, score_rem_storm,
    compute_rmssd_storm_series,
)

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


# ─────────────────────────────────────────────────────────────────────────────
#  Safe-default registry (all z-score keys that scorers may request)
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_DEFAULTS = {
    # Original features
    "mean_hr": 0.5, "rmssd": 0.5, "lf_power": 0.5,
    "hf_power": 0.5, "lf_hf_ratio": 0.5, "sampen": 0.5,
    "sd1": 0.5, "pnn50": 0.5,
    "sdnn_rmssd_ratio": 0.5, "hr_epoch_delta": 0.0,
    # Original z-scores
    "mean_hr_zscore": 0.0, "lf_hf_ratio_zscore": 0.0,
    "rmssd_zscore": 0.0, "sampen_zscore": 0.0,
    "pnn50_zscore": 0.0,
    "sd1_zscore": 0.0,
    "hr_epoch_delta_zscore": 0.0,
    # Phase 3 v2 features
    "pnn20": 0.5, "pnn20_zscore": 0.0,
    "hr_range": 0.5, "hr_range_zscore": 0.0,
    "resp_rate_est": 0.5, "resp_rate_est_zscore": 0.0,
    "perm_en": 0.5, "perm_en_zscore": 0.0,
    "rmssd_mean_rr_ratio": 50.0, "rmssd_mean_rr_ratio_zscore": 0.0,
    "median_rr": 0.5, "median_rr_zscore": 0.0,
    "rr_iqr": 0.5, "rr_iqr_zscore": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Classify
# ─────────────────────────────────────────────────────────────────────────────

def classify_epochs(df: pd.DataFrame,
                    config: dict,
                    logger: logging.Logger):
    """
    RC6 + FIX-5b + RC8a: Apply rule-based scoring, time-of-night priors, and
    adaptive Wake boost; return (raw_stages, scores).

    Parameters
    ----------
    df     : normalised feature DataFrame (output of normalise_features)
    config : pipeline config dict
    logger : Logger

    Returns
    -------
    raw_stages : np.ndarray of int, shape (n_epochs,)
    scores     : np.ndarray of float, shape (n_epochs, 5)  (post-prior, pre-softmax)
    """
    n_epochs     = len(df)
    scores       = np.zeros((n_epochs, 5))
    storm_series = compute_rmssd_storm_series(df)
    logger.info(
        f"Autonomic storm series: mean={storm_series.mean():.3f}  "
        f"max={storm_series.max():.3f}  "
        f"epochs with storm>0.20: {(storm_series > 0.20).sum()}"
    )

    # FIX-5b: subject-adaptive Wake HR boost
    _wake_hr_p75 = (float(df["__wake_hr_p75"].iloc[0])
                    if "__wake_hr_p75" in df.columns else float("nan"))
    if np.isfinite(_wake_hr_p75):
        wake_adaptive_boost = float(
            np.clip(1.0 + (75.0 - _wake_hr_p75) / 50.0, 1.0, 1.60)
        )
    else:
        wake_adaptive_boost = 1.0
    logger.info(
        f"FIX-5b adaptive Wake boost: wake_hr_p75={_wake_hr_p75:.1f} bpm  "
        f"→ wake_score_multiplier={wake_adaptive_boost:.3f}"
    )

    # Per-epoch scoring
    for i in range(n_epochs):
        row = df.iloc[i].copy()
        # Fill missing values with safe defaults
        for k, v in _SAFE_DEFAULTS.items():
            val = row.get(k, np.nan)
            if pd.isna(val):
                row[k] = v
            else:
                try:
                    row[k] = float(val)
                except (TypeError, ValueError):
                    row[k] = v

        scores[i, 0] = score_wake(row) * wake_adaptive_boost
        scores[i, 1] = score_n1(row)
        scores[i, 2] = score_n2(row)
        scores[i, 3] = score_n3_gated(row)
        scores[i, 4] = score_rem_storm(row, storm_series[i])

    # FIX-5b: detect high-Wake-onset subjects
    first_30_wake_frac = 0.0
    if n_epochs >= 30:
        first_30_raw       = np.argmax(scores[:30], axis=1)
        first_30_wake_frac = float(np.mean(first_30_raw == 0))
    high_wake_onset = first_30_wake_frac > 0.40
    if high_wake_onset:
        logger.info(
            f"FIX-5b HIGH WAKE ONSET detected: "
            f"{first_30_wake_frac * 100:.0f}% of first 30 epochs score Wake "
            f"— boosting first-third Wake prior 1.00 → 1.30"
        )

    # ── RC8a FIX ────────────────────────────────────────────────────────────
    # RC6: PSG-prevalence-informed time-of-night priors.
    # Previously `prior` was recomputed every loop iteration and discarded —
    # it never touched `scores`. Now built as a full (n_epochs, 5) array and
    # actually multiplied into the scores before the softmax below.
    priors = np.ones((n_epochs, 5))
    for i in range(n_epochs):
        frac = i / max(n_epochs - 1, 1)
        if frac < 0.33:
            wake_p = 1.30 if high_wake_onset else 1.00
            priors[i] = [wake_p, 0.85, 1.35, 0.80, 0.25]  # N2 dominant early; N3/REM rare
        elif frac < 0.66:
            priors[i] = [0.80, 0.90, 1.30, 0.85, 1.00]    # N2 still dominant mid-night
        else:
            priors[i] = [1.00, 0.85, 1.15, 0.35, 1.30]    # REM rises late but N2 not penalised

    scores = scores * priors   # <-- RC8a: priors now actually applied

    logger.info(
        f"RC8a priors applied — early-third REM prior=0.25, "
        f"late-third REM prior=1.30, late-third N3 prior=0.35"
    )

    # RC6: softmax temperature = 2.5 (was 1.5) — sharpens stage separation
    exp_s      = np.exp(scores * 2.5)
    raw_stages = np.argmax(exp_s / exp_s.sum(axis=1, keepdims=True), axis=1)

    counts = Counter(raw_stages)
    logger.info(
        f"Raw stage distribution: "
        f"{ {STAGE_NAMES.get(k, str(k)): v for k, v in sorted(counts.items())} }"
    )
    return raw_stages, scores


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal smoothing
# ─────────────────────────────────────────────────────────────────────────────

def smooth_hypnogram(raw_stages,
                     kernel_size: int = 5,
                     min_run_epochs: int = 2,
                     logger=None):
    """
    Two-pass smoothing that removes single-epoch noise WITHOUT erasing
    legitimate minority stages (REM, N3) that appear in short bursts.

    Pass 1 — Median filter (kernel_size must be odd; auto-corrected if even).
    Pass 2 — Min-run merger: runs shorter than min_run_epochs are replaced
              by the adjacent longer run, unless the stage is N3 or REM
              (PROTECTED stages are never forcibly merged away).

    RC8b FIX: PROTECTED was previously an empty set — the docstring/comment
    claimed N3 and REM were protected from the min-run merger, but nothing
    actually was. Short REM/N3 runs (very common with a noisy per-epoch
    rule-based classifier) were being merged into whichever neighboring
    stage was longer, almost always N2. PROTECTED now actually contains
    the N3 (3) and REM (4) stage codes.
    """
    from scipy.signal import medfilt

    stages_series = pd.Series(raw_stages.copy()).astype(float)
    stages_series[stages_series < 0] = np.nan
    filled = stages_series.ffill().bfill().values.astype(int)

    kernel   = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    smoothed = medfilt(filled.astype(float), kernel_size=kernel).astype(int)

    PROTECTED = {3, 4}  # RC8b FIX: N3, REM — never forcibly merged

    if min_run_epochs > 1:
        result = smoothed.copy()
        n      = len(result)
        i      = 0
        while i < n:
            current_stage = result[i]
            j = i
            while j < n and result[j] == current_stage:
                j += 1
            run_len = j - i
            if run_len < min_run_epochs and current_stage not in PROTECTED:
                left_stage  = result[i - 1] if i > 0 else -1
                right_stage = result[j]     if j < n else -1
                left_len    = i
                right_len   = n - j
                replace_with = (left_stage
                                if (left_len >= right_len or right_stage < 0)
                                else right_stage)
                if replace_with >= 0:
                    result[i:j] = replace_with
            i = j
        smoothed = result

    if logger:
        counts = Counter(smoothed)
        logger.info(
            f"Smoothed stage distribution (K={kernel}, "
            f"min_run={min_run_epochs}, PROTECTED={sorted(PROTECTED)}): "
            f"{ {STAGE_NAMES.get(k, str(k)): v for k, v in sorted(counts.items())} }"
        )
    return smoothed