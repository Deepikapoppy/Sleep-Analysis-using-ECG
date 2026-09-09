"""
=============================================================================
phase4_sleep_classification/p4_scoring.py
Rule-based stage scoring functions (all RCA + RC7 fixes applied).

Each score_*() function receives one normalised epoch row (pd.Series)
and returns a float in [0, 1] indicating how strongly that epoch
matches the corresponding sleep stage.

RCA fix history
───────────────
  RC1 — N3 gate: AND → OR logic; thresholds relaxed; suppression 0.08 → 0.20
  RC2 — REM core: sd1×hf → sd1_zscore-based (HF is LOW during genuine REM)
  RC3 — Wake z-score gate added (hr_z < -1.0 AND rmssd_z > 0.5)
  RC4 — N1 gate tightened; bell-curve peak shifted 0.55 → 0.60
  RC5 — Spectral smoothing in Phase 3; N2 z-score gate (hr_z > 1.0)
  RC6 — Priors redesigned (PSG prevalence); softmax temp 1.5 → 2.5
  RC7 — Wake gate relaxed; N1 gate relaxed; Wake core boosted
=============================================================================
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Wake
# ─────────────────────────────────────────────────────────────────────────────

def score_wake(row) -> float:
    """
    RC3 + RC7 FIX.
    Gate: suppress only when hr_z < -1.0 AND rmssd_z > 0.5
    (both bradycardia AND elevated vagal tone must be present).
    Core boosted with positive hr_z contribution; perm_en added as complexity
    supplement (Wake has high PermEn).
    """
    hr      = float(row["mean_hr"])
    rmssd   = float(row["rmssd"])
    lf_hf   = float(row["lf_hf_ratio"])
    se      = float(row["sampen"])
    hf      = float(row["hf_power"])
    hr_z    = float(row.get("mean_hr_zscore",  0.0))
    rmssd_z = float(row.get("rmssd_zscore",    0.0))
    perm_en = float(row.get("perm_en",         0.5))

    # RC7: only suppress when BOTH definite bradycardia AND vagal dominance
    if hr_z < -1.0 and rmssd_z > 0.5:
        return 0.10

    hr_z_pos = max(0.0, hr_z)                            # positive z: above-mean HR
    core     = hr * (1.0 - rmssd) + hr_z_pos * 0.15     # z-score bonus for Wake
    supp     = (lf_hf * 0.50 + se * 0.40 + (1.0 - hf) * 0.30
                + perm_en * 0.30) / 1.50
    return core * 0.70 + supp * 0.30


# ─────────────────────────────────────────────────────────────────────────────
#  N1
# ─────────────────────────────────────────────────────────────────────────────

def score_n1(row) -> float:
    """
    RC4 + RC7 FIX.
    Gate relaxed: only suppress when hr_z < -1.2 (definite deep-NREM bradycardia).
    Bell-curve peak at hr = 0.60 (wake-adjacent), narrower (3.0) to stop N1
    eating into N2 territory.
    """
    hr    = float(row["mean_hr"])
    rmssd = float(row["rmssd"])
    lf    = float(row["lf_power"])
    se    = float(row["sampen"])
    hr_z  = float(row.get("mean_hr_zscore", 0.0))

    if hr_z < -1.2:
        return 0.05

    hr_score    = max(0.0, 1.0 - abs(hr - 0.60) * 3.0)
    rmssd_score = max(0.0, 1.0 - rmssd * 2.5)
    supp        = (lf * 0.60 + (1.0 - se) * 0.40) / 1.00
    return (hr_score * 1.5 + rmssd_score * 1.2 + supp * 0.8) / 3.5


# ─────────────────────────────────────────────────────────────────────────────
#  N2
# ─────────────────────────────────────────────────────────────────────────────

def score_n2(row) -> float:
    """
    RC5 + RC7 FIX.
    Gate raised to hr_z > 1.0 (only clearly elevated HR suppresses N2).
    resp_rate_est and rr_iqr added as N2 physiological shape constraints.
    """
    hr    = float(row["mean_hr"])
    lf    = float(row["lf_power"])
    lf_hf = float(row["lf_hf_ratio"])
    rmssd = float(row["rmssd"])
    se    = float(row["sampen"])
    hr_z  = float(row.get("mean_hr_zscore", 0.0))

    if hr_z > 1.5:
        return 0.08

    resp_rate   = float(row.get("resp_rate_est", 0.5))
    resp_n2_fit = max(0.0, 1.0 - abs(resp_rate - 0.40) * 1.5)

    rr_iqr  = float(row.get("rr_iqr", 0.5))
    iqr_fit = max(0.0, 1.0 - abs(rr_iqr - 0.35) * 2.0)

    hr_low    = max(0.0, 1.0 - abs(hr - 0.35) * 2.5)
    regular   = 1.0 - se
    rmssd_mod = max(0.0, 1.0 - abs(rmssd - 0.40) * 2.5)

    core = (lf_hf * 0.55 + hr_low * 0.45)   # additive — N2 survives when one term is low
    supp = (lf * 1.0 + rmssd_mod * 0.8 + regular * 0.7
            + resp_n2_fit * 0.5 + iqr_fit * 0.4) / 3.4
    return core * 0.60 + supp * 0.40


# ─────────────────────────────────────────────────────────────────────────────
#  N3 (base + gated)
# ─────────────────────────────────────────────────────────────────────────────

def score_n3(row) -> float:
    """
    Base N3 score.
    pnn20 added alongside pnn50 (more sensitive in 30-s epochs).
    rmssd_mean_rr_ratio used as HR-normalised vagal supplement.
    """
    hr    = float(row["mean_hr"])
    rmssd = float(row["rmssd"])
    hf    = float(row["hf_power"])
    lf_hf = float(row["lf_hf_ratio"])
    se    = float(row["sampen"])
    pnn50 = float(row["pnn50"])

    pnn20         = float(row.get("pnn20", pnn50))
    pnn_combined  = (pnn50 + pnn20) / 2.0

    rmssd_rr_ratio = float(row.get("rmssd_mean_rr_ratio", rmssd * 100.0))
    rr_vagal       = min(1.0, rmssd_rr_ratio / 100.0)

    hf_effective = max(hf, pnn_combined * 0.5)
    vagal_core   = rmssd * hf_effective

    supp = ((1.0 - hr) * 1.5 + (1.0 - lf_hf) * 1.2
            + (1.0 - se) * 0.8 + pnn_combined * 1.0
            + rr_vagal * 0.8) / 5.3

    return vagal_core * 0.55 + supp * 0.45


def score_n3_gated(row) -> float:
    """
    RC1 FIX: OR logic gate — passes when EITHER bradycardia OR vagal dominance.
    Both conditions together trigger a ×1.20 bonus.
    Neither condition suppresses to ×0.20 (was ×0.08).
    pnn20_zscore added to vagal gate (Phase 3 v2, more sensitive for 30-s epochs).
    """
    base    = score_n3(row)
    hr_z    = float(row.get("mean_hr_zscore",     0.0))
    lfhf_z  = float(row.get("lf_hf_ratio_zscore", 0.0))
    pnn50_z = float(row.get("pnn50_zscore",        0.0))
    pnn20_z = float(row.get("pnn20_zscore",        0.0))

    gate_hr    = hr_z    < -0.8                          # tighter: truly bradycardic
    gate_vagal = (pnn50_z > 0.8) or (pnn20_z > 0.7)    # removed lfhf_z — too broad

    if gate_hr and gate_vagal:
        return base * 1.15       # both → strong N3
    elif gate_hr:
        return base * 0.60       # bradycardia alone → partial credit
    elif gate_vagal:
        return base * 0.35       # vagal alone is NOT enough for N3
    else:
        return base * 0.08       # neither → strong suppression


# ─────────────────────────────────────────────────────────────────────────────
#  REM
# ─────────────────────────────────────────────────────────────────────────────

def score_rem(row) -> float:
    """
    RC2 FIX: rem_core = f(sd1_zscore) — replaces the broken sd1×hf core.
    HF is LOW during genuine REM; the old multiplicative core was near-zero
    for true REM epochs.  SD1 above the night mean (sd1_z > 0) signals a
    variability burst — the hallmark of phasic REM.

    perm_en: high complexity in REM (high PermEn ≈ 0.82–0.98).
    hr_range: elevated during phasic REM autonomic bursts.
    rmssd_mean_rr_ratio: soft suppressor when ratio is very high (N3 territory).
    """
    sd1_z      = float(row.get("sd1_zscore",            0.0))
    se         = float(row["sampen"])
    hr         = float(row["mean_hr"])
    rmssd      = float(row["rmssd"])
    lf_hf      = float(row["lf_hf_ratio"])
    sdnn_rmssd = float(row.get("sdnn_rmssd_ratio",       0.5))
    hr_delta_z = float(row.get("hr_epoch_delta_zscore",  0.0))
    perm_en    = float(row.get("perm_en",                0.5))
    hr_range   = float(row.get("hr_range",               0.5))

    # RC2 FIX: z-score-based core — clips sd1_z ∈ [-1, 2] → [0, 1]
    rem_core = min(1.0, max(0.0, (sd1_z - 0.3) / 2.0))   # zero floor shifted right — sd1_z must be above-mean to score

    hr_mod           = max(0.0, 1.0 - abs(hr - 0.38) * 2.5)
    hr_delta_contrib = min(1.0, max(0.0, (hr_delta_z + 1.0) / 3.0))
    perm_rem         = max(0.0, 1.0 - abs(perm_en - 0.90) * 4.0)
    hr_range_rem     = min(1.0, hr_range * 1.2)

    rmssd_rr_ratio = float(row.get("rmssd_mean_rr_ratio", 50.0))
    rem_vagal_supp = min(1.0, max(0.0,
                        1.0 - max(0.0, rmssd_rr_ratio - 80.0) / 40.0))

    supp = (se * 0.50 + (1.0 - lf_hf) * 0.40 + rmssd * 0.40
            + hr_mod * 0.50 + sdnn_rmssd * 0.55
            + hr_delta_contrib * 0.50
            + perm_rem * 0.45 + hr_range_rem * 0.40) / 3.70

    return (rem_core * 0.55 + supp * 0.45) * rem_vagal_supp


# ─────────────────────────────────────────────────────────────────────────────
#  Autonomic-storm helper for REM gate
# ─────────────────────────────────────────────────────────────────────────────

def compute_rmssd_storm_series(df_normed: pd.DataFrame,
                                window: int = 20) -> np.ndarray:
    """
    Detect per-epoch Autonomic Storm events — hallmark of phasic REM.
    Returns a normalised [0, 1] storm score per epoch.
    """
    if "rmssd" not in df_normed.columns:
        return np.zeros(len(df_normed))

    rmssd = pd.to_numeric(df_normed["rmssd"], errors="coerce").fillna(0.5).values
    n     = len(rmssd)
    storm = np.zeros(n, dtype=float)
    half  = window // 2

    for i in range(n):
        lo    = max(0, i - half)
        hi    = min(n, i + half + 1)
        local = rmssd[lo:hi]
        lmu   = float(np.mean(local))
        lstd  = float(np.std(local, ddof=1)) if len(local) > 2 else 0.0
        if lstd < 1e-4:
            lstd = 1e-4
        storm[i] = max(0.0, (rmssd[i] - lmu) / lstd)

    s_max = storm.max()
    if s_max > 1e-8:
        storm /= s_max
    return storm


def score_rem_storm(row, storm_val: float) -> float:
    """
    RC2 FIX: storm threshold lowered 0.28 → 0.20; suppression raised 0.40 → 0.50;
    hr_z gate relaxed 0.5 → 0.8 so moderate-HR REM epochs are not blocked.
    """
    base  = score_rem(row)
    hr_z  = float(row.get("mean_hr_zscore", 0.0))
    gate_ok = (storm_val > 0.55) and (hr_z < 0.6)   # top 45% of storm, not bottom 80%
    return base * 1.35 if gate_ok else base * 0.45   # reduced boost, stronger non-storm suppression
