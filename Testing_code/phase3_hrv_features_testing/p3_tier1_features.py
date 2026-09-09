"""
=============================================================================
phase3_hrv_features/p3_tier1_features.py
T1 Primary Screamer — 18 features
All RR-derived; high-confidence stage discriminators (~60% epoch coverage).

Features
────────
Time-domain (5)
  mean_hr           Wake ≥65 bpm; N3 <55 bpm. Single strongest Wake gate.
  mean_rr           Inverse of HR. >1000 ms → N3, <750 ms → Wake.
  rmssd             Best single vagal tone marker. Very high in N3.
  pnn20             % |ΔRR|>20 ms. More sensitive than pNN50 for 30-s epochs.
  sdnn              Overall HRV magnitude. High Wake/REM, low N3.

Temporal shape (2)
  hr_range          max_hr − min_hr within epoch. High REM/Wake, very low N3.
  rr_autocorr_lag1  Lag-1 autocorrelation. High positive in N3.

Frequency (4)
  hf_power          RSA — pure parasympathetic. Highest in N3.
  lf_hf_ratio       Sympathovagal balance. High Wake/N2, very low N3.
  lf_power          Mixed sympathetic + baroreflex. Dominant in N2.
  resp_rate_est     ECG-derived breath rate (br/min). N3: 12–16, REM: 15–22.

Non-linear (5)
  perm_en           Permutation entropy. Wake: 0.85–1.00; N3: 0.55–0.78.
  dfa_alpha1        α1>1.0 in N3 (strong long-range correlations). Best NL N3.
  sd1               Beat-to-beat variability (RMSSD proxy). High in N3.
  sd1_sd2_ratio     High → parasympathetic (N3). Low → long-range (Wake).
  sampen            Unpredictability. High Wake/REM, low N3. Needs n_rr ≥ 10.

Derived (2)
  rmssd_mean_rr_ratio  HR-normalised vagal index. High N3, low REM.
  sdnn_rmssd_ratio     High → sympathetic (Wake/REM). Low → vagal (N3).
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from itertools import permutations
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Time-domain
# ─────────────────────────────────────────────────────────────────────────────

def compute_t1_time(rr_ms: list) -> dict:
    """
    T1 time-domain features: mean_hr, mean_rr, rmssd, pnn20, sdnn, hr_range.
    """
    keys = ["mean_rr", "rmssd", "pnn20", "sdnn", "mean_hr",
            "min_hr", "max_hr", "hr_range"]
    if len(rr_ms) < 4:
        return {k: np.nan for k in keys}

    rr   = np.array(rr_ms, dtype=float)
    diff = np.abs(np.diff(rr))
    n    = max(len(diff), 1)

    mean_rr = float(np.mean(rr))
    sdnn    = float(np.std(rr, ddof=1))
    rmssd   = float(np.sqrt(np.mean(diff ** 2)))
    pnn20   = float(100.0 * np.sum(diff > 20.0) / n)
    mean_hr = float(60000.0 / mean_rr)
    min_hr  = float(60000.0 / np.max(rr))
    max_hr  = float(60000.0 / np.min(rr))
    hr_range = float(max_hr - min_hr)

    return {
        "mean_rr"  : mean_rr,
        "rmssd"    : rmssd,
        "pnn20"    : pnn20,
        "sdnn"     : sdnn,
        "mean_hr"  : mean_hr,
        "min_hr"   : min_hr,
        "max_hr"   : max_hr,
        "hr_range" : hr_range,
    }


def compute_rr_autocorr_lag1(rr_ms: list) -> float:
    """
    Lag-1 autocorrelation of RR series.
    High positive → N3 (each RR predicts next — regular RSA).
    Near-zero → Wake/REM. Only needs 4 beats.
    """
    if len(rr_ms) < 4:
        return np.nan
    rr = np.array(rr_ms, dtype=float)
    rr_c = rr - np.mean(rr)
    denom = np.sum(rr_c ** 2)
    if denom < 1e-12:
        return np.nan
    return float(np.sum(rr_c[:-1] * rr_c[1:]) / denom)


# ─────────────────────────────────────────────────────────────────────────────
#  Frequency-domain (Lomb-Scargle)  — T1 subset
# ─────────────────────────────────────────────────────────────────────────────

def _lomb_scargle_bands(rr_ms: list) -> dict:
    """
    Compute LF, HF, LF/HF, resp_rate_est via Lomb-Scargle periodogram.
    Returns NaN dict if n_rr < 10 or span < 20 s.
    """
    from astropy.timeseries import LombScargle

    nan_keys = ["lf_power", "hf_power", "lf_hf_ratio", "resp_rate_est",
                "total_power", "vlf_power"]
    nan_d = {k: np.nan for k in nan_keys}

    if len(rr_ms) < 10:
        return nan_d

    rr = np.array(rr_ms, dtype=float) / 1000.0
    t  = np.cumsum(rr) - rr[0]
    if t[-1] < 20.0:
        return nan_d

    try:
        freq  = np.linspace(0.001, 0.5, 1000)
        power = LombScargle(t, rr - np.mean(rr)).power(freq)

        def band_power(flo, fhi):
            mask = (freq >= flo) & (freq < fhi)
            return float(np.trapz(power[mask], freq[mask])) if mask.sum() > 0 else 0.0

        vlf = band_power(0.003, 0.04)
        lf  = band_power(0.04,  0.15)
        hf  = band_power(0.15,  0.40)
        tot = vlf + lf + hf

        hf_mask = (freq >= 0.15) & (freq < 0.40)
        if hf_mask.sum() > 0:
            peak_hf   = float(freq[hf_mask][np.argmax(power[hf_mask])])
            resp_rate = float(peak_hf * 60.0)
        else:
            peak_hf   = np.nan
            resp_rate = np.nan

        return {
            "vlf_power"    : vlf,
            "lf_power"     : lf,
            "hf_power"     : hf,
            "lf_hf_ratio"  : float(lf / (hf + 1e-10)),
            "total_power"  : tot,
            "resp_rate_est": resp_rate,
        }
    except Exception:
        return nan_d


# ─────────────────────────────────────────────────────────────────────────────
#  Non-linear — T1 subset
# ─────────────────────────────────────────────────────────────────────────────

def compute_perm_entropy(rr_ms: list, order: int = 3, normalize: bool = True) -> float:
    """
    Permutation Entropy (PermEn). Fast O(N), no free parameter.
    Wake: 0.85–1.00 | N3: 0.55–0.78 | REM: 0.82–0.98.
    Reference: Bandt & Pompe (2002).
    """
    if len(rr_ms) < order + 2:
        return np.nan
    try:
        rr    = np.array(rr_ms, dtype=float)
        perms = list(permutations(range(order)))
        pidx  = {p: i for i, p in enumerate(perms)}
        cnts  = np.zeros(len(perms), dtype=float)
        for i in range(len(rr) - order + 1):
            pattern = tuple(np.argsort(rr[i:i + order], kind='stable'))
            cnts[pidx[pattern]] += 1
        cnts = cnts[cnts > 0]
        p    = cnts / cnts.sum()
        pe   = float(-np.sum(p * np.log2(p)))
        if normalize:
            import math
            max_pe = np.log2(math.factorial(order))
            pe = pe / max_pe if max_pe > 0 else pe
        return float(pe)
    except Exception:
        return np.nan


def compute_sample_entropy(rr_ms: list, m: int = 2, r_coef: float = 0.2) -> float:
    """
    Sample Entropy. High Wake/REM (complex), low N3 (regular). Needs n_rr ≥ 10.
    """
    if len(rr_ms) < 10:
        return np.nan
    try:
        rr = np.array(rr_ms, dtype=float)
        r  = r_coef * np.std(rr)
        N  = len(rr)

        def count_templates(m_len):
            count = 0
            for i in range(N - m_len):
                tmpl = rr[i:i + m_len]
                for j in range(i + 1, N - m_len + 1):
                    if np.max(np.abs(rr[j:j + m_len] - tmpl)) <= r:
                        count += 1
            return count

        B = count_templates(m)
        A = count_templates(m + 1)
        return float(-np.log(A / B)) if B > 0 else np.nan
    except Exception:
        return np.nan


def compute_poincare_t1(rr_ms: list) -> dict:
    """SD1, SD2, SD1/SD2 — Poincaré plot features."""
    if len(rr_ms) < 4:
        return {"sd1": np.nan, "sd2": np.nan, "sd1_sd2_ratio": np.nan}
    rr   = np.array(rr_ms, dtype=float)
    diff = np.diff(rr)
    sd1  = float(np.sqrt(0.5) * np.std(diff, ddof=1))
    sd2  = float(np.sqrt(max(2 * np.var(rr, ddof=1) - 0.5 * np.var(diff, ddof=1), 0)))
    ratio = float(sd1 / (sd2 + 1e-10))
    return {"sd1": sd1, "sd2": sd2, "sd1_sd2_ratio": ratio}


def compute_dfa_alpha1(rr_ms: list) -> float:
    """
    DFA α1 (short-range scaling, windows 4–16).
    α1 > 1.0 in N3 (strong long-range correlations from slow waves).
    Best single nonlinear N3 marker. Needs n_rr ≥ 20.
    """
    if len(rr_ms) < 20:
        return np.nan
    try:
        rr = np.array(rr_ms, dtype=float)
        N  = len(rr)
        y  = np.cumsum(rr - np.mean(rr))

        def fluctuation(n):
            F_vals = []
            for start in range(0, N - n + 1, n):
                seg   = y[start:start + n]
                t_seg = np.arange(n)
                trend = np.polyval(np.polyfit(t_seg, seg, 1), t_seg)
                F_vals.append(np.mean((seg - trend) ** 2))
            return np.sqrt(np.mean(F_vals)) if F_vals else np.nan

        scales = np.unique(np.round(
            np.logspace(np.log10(4), np.log10(16), 8)).astype(int))
        valid  = [(np.log10(s), np.log10(fluctuation(s)))
                  for s in scales if s < N and np.isfinite(fluctuation(s))]
        if len(valid) < 3:
            return np.nan
        xs, ys = zip(*valid)
        return float(np.polyfit(xs, ys, 1)[0])
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  Main T1 compute entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_tier1(rr_ms: list) -> dict:
    """
    Compute all 18 T1 Primary Screamer features for one epoch.

    Parameters
    ----------
    rr_ms : list of RR intervals in milliseconds (from Phase 2, already filtered)

    Returns
    -------
    dict with keys:
        mean_hr, mean_rr, rmssd, pnn20, sdnn,
        min_hr, max_hr, hr_range,
        rr_autocorr_lag1,
        hf_power, lf_hf_ratio, lf_power, resp_rate_est,
        (vlf_power, total_power — also computed for postprocess use)
        perm_en, dfa_alpha1, sd1, sd2, sd1_sd2_ratio, sampen,
        rmssd_mean_rr_ratio, sdnn_rmssd_ratio
    """
    feats = {}

    # Time-domain
    feats.update(compute_t1_time(rr_ms))
    feats["rr_autocorr_lag1"] = compute_rr_autocorr_lag1(rr_ms)

    # Frequency — full LS for all bands (used by T2 as well)
    ls = _lomb_scargle_bands(rr_ms)
    feats.update(ls)

    # Non-linear
    feats["perm_en"]   = compute_perm_entropy(rr_ms)
    feats["dfa_alpha1"]= compute_dfa_alpha1(rr_ms)
    feats.update(compute_poincare_t1(rr_ms))
    feats["sampen"]    = compute_sample_entropy(rr_ms)

    # Derived (need mean_rr, rmssd, sdnn already in feats)
    mean_rr = feats.get("mean_rr", np.nan)
    rmssd   = feats.get("rmssd",   np.nan)
    sdnn    = feats.get("sdnn",    np.nan)
    feats["rmssd_mean_rr_ratio"] = (
        float(rmssd / (mean_rr + 1e-6) * 100.0)
        if np.isfinite(rmssd) and np.isfinite(mean_rr) else np.nan
    )
    feats["sdnn_rmssd_ratio"] = (
        float(sdnn / (rmssd + 1e-6))
        if np.isfinite(sdnn) and np.isfinite(rmssd) else np.nan
    )

    return feats
