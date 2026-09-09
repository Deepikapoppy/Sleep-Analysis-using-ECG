"""
=============================================================================
phase3_hrv_features/p3_tier2_features.py
T2 Confirmatory — 19 features
Refines stage decisions; introduces first EDR features. (~25% more coverage)

Features
────────
Time-domain (6)
  median_rr         Robust to ectopic beats. High N3, low Wake.
  pnn50             Parasympathetic index. High N3, very low Wake.
  cv                SDNN normalised for HR. Elevated Wake/REM.
  rr_iqr            Robust spread (Q75-Q25). Bell-curve reward N2 scoring.
  min_hr            Deepest bradycardia moment → N3 flag.
  max_hr            Peak HR within epoch. Elevated Wake/REM.

Frequency (5)
  hf_nu             HF normalised units. High N3. Cross-subject comparison.
  lf_nu             LF normalised units. High N2.
  peak_hf_freq      Dominant respiratory freq (Hz). Low ~0.20–0.25 → N3.
  total_power       Overall spectral energy. High Wake.
  sd2               Long-term HRV = SDNN proxy. High Wake.

Nonlinear (3)
  prsa_dc           Deceleration capacity. High N3/N2. Bauer 2006.
  porta_asymmetry   % deceleration beats in Poincaré. REM signature.
  sd2               (computed above; also in nonlinear group)

EDR (3)
  edr_breath_rate   ECG-derived breathing rate. N3: 10–14. REM: 15–22 br/min.
  edr_regularity    Breath-to-breath CV%. REM: >15%, N3: <5%.
  edr_baseline_wander  Baseline wander amplitude → respiration proxy.

Context (2)
  sleep_cycle_pos   Fractional position in 90-min ultradian cycle.
  hr_slope_epoch    Within-epoch HR trend slope.

Derived (2) — computed in postprocess using T1+T2 results
  hr_epoch_delta    Inter-epoch HR transition.
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Time-domain T2 additions
# ─────────────────────────────────────────────────────────────────────────────

def compute_t2_time(rr_ms: list) -> dict:
    """
    T2 time-domain: median_rr, pnn50, cv, rr_iqr.
    min_hr / max_hr already computed in T1; reuse from there.
    """
    keys = ["median_rr", "pnn50", "cv", "rr_iqr"]
    if len(rr_ms) < 4:
        return {k: np.nan for k in keys}

    rr   = np.array(rr_ms, dtype=float)
    diff = np.abs(np.diff(rr))
    n    = max(len(diff), 1)
    mean_rr = float(np.mean(rr))
    sdnn    = float(np.std(rr, ddof=1))

    return {
        "median_rr": float(np.median(rr)),
        "pnn50"    : float(100.0 * np.sum(diff > 50.0) / n),
        "cv"       : float(sdnn / (mean_rr + 1e-10) * 100.0),
        "rr_iqr"   : float(np.percentile(rr, 75) - np.percentile(rr, 25)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Frequency T2 additions (uses T1 LS power already available)
# ─────────────────────────────────────────────────────────────────────────────

def compute_t2_freq(lf: float, hf: float, tot: float, rr_ms: list) -> dict:
    """
    T2 freq features derived from T1 LS bands.
    Also computes peak_hf_freq here (was shared in T1 resp_rate_est call).

    Parameters
    ----------
    lf, hf, tot : from T1 _lomb_scargle_bands result
    rr_ms       : needed to re-derive peak_hf_freq if not passed from T1
    """
    keys = ["hf_nu", "lf_nu", "peak_hf_freq"]
    nan_d = {k: np.nan for k in keys}

    # peak_hf_freq via Lomb-Scargle
    peak_hf_freq = np.nan
    if len(rr_ms) >= 10:
        try:
            from astropy.timeseries import LombScargle
            rr  = np.array(rr_ms, dtype=float) / 1000.0
            t   = np.cumsum(rr) - rr[0]
            if t[-1] >= 20.0:
                freq  = np.linspace(0.001, 0.5, 1000)
                power = LombScargle(t, rr - np.mean(rr)).power(freq)
                hf_mask = (freq >= 0.15) & (freq < 0.40)
                if hf_mask.sum() > 0:
                    peak_hf_freq = float(freq[hf_mask][np.argmax(power[hf_mask])])
        except Exception:
            pass

    lf_hf_sum = lf + hf + 1e-10
    return {
        "hf_nu"       : float(hf / lf_hf_sum * 100.0) if np.isfinite(hf) else np.nan,
        "lf_nu"       : float(lf / lf_hf_sum * 100.0) if np.isfinite(lf) else np.nan,
        "peak_hf_freq": peak_hf_freq,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  DFA alpha2 — T2 (long-range scaling 16–64)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dfa_alpha2(rr_ms: list) -> float:
    """
    DFA α2 (long-range, windows 16–64). Needs n_rr ≥ 20.
    Less stage-specific than α1. Differentiates pathological HRV.
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
            np.logspace(np.log10(16), np.log10(64), 8)).astype(int))
        valid  = [(np.log10(s), np.log10(fluctuation(s)))
                  for s in scales if s < N and np.isfinite(fluctuation(s))]
        if len(valid) < 3:
            return np.nan
        xs, ys = zip(*valid)
        return float(np.polyfit(xs, ys, 1)[0])
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  PRSA — Deceleration and Acceleration Capacity
# ─────────────────────────────────────────────────────────────────────────────

def compute_prsa_dc(rr_ms: list, T: int = 2) -> float:
    """
    Phase-Rectified Signal Averaging — Deceleration Capacity.
    High DC in N3/N2 (vagal dominance). Robust to artefacts.
    Reference: Bauer et al. Lancet 2006.
    Needs n_rr ≥ 10.
    """
    if len(rr_ms) < 10:
        return np.nan
    try:
        rr = np.array(rr_ms, dtype=float)
        # Identify deceleration anchors: RR[i] > RR[i-1]
        anchors = [i for i in range(1, len(rr))
                   if rr[i] > rr[i - 1] and i >= T and i + T < len(rr)]
        if not anchors:
            return np.nan
        segments = np.array([rr[a - T:a + T + 1] for a in anchors])
        avg = np.mean(segments, axis=0)
        # DC = (X(0) + X(1) - X(-1) - X(-2)) / 4
        mid = T
        dc  = (avg[mid] + avg[mid + 1] - avg[mid - 1] - avg[mid - 2]) / 4.0
        return float(dc)
    except Exception:
        return np.nan


def compute_prsa_ac(rr_ms: list, T: int = 2) -> float:
    """
    PRSA — Acceleration Capacity. Quantifies sympathetic HR increases.
    High in Wake/REM. Robust to artefacts.
    """
    if len(rr_ms) < 10:
        return np.nan
    try:
        rr = np.array(rr_ms, dtype=float)
        anchors = [i for i in range(1, len(rr))
                   if rr[i] < rr[i - 1] and i >= T and i + T < len(rr)]
        if not anchors:
            return np.nan
        segments = np.array([rr[a - T:a + T + 1] for a in anchors])
        avg = np.mean(segments, axis=0)
        mid = T
        ac  = (avg[mid] + avg[mid + 1] - avg[mid - 1] - avg[mid - 2]) / 4.0
        return float(ac)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  Poincaré asymmetry — T2 (Porta)
# ─────────────────────────────────────────────────────────────────────────────

def compute_porta_asymmetry(rr_ms: list) -> float:
    """
    Porta asymmetry index — % deceleration beats in Poincaré space.
    REM shows distinctly asymmetric pattern (autonomic instability).
    Reference: Porta et al. 2009.
    """
    if len(rr_ms) < 5:
        return np.nan
    try:
        rr   = np.array(rr_ms, dtype=float)
        diff = np.diff(rr)
        n_tot = len(diff)
        n_dec = int(np.sum(diff > 0))   # RR[i+1] > RR[i] → deceleration
        return float(n_dec / n_tot * 100.0) if n_tot > 0 else np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  EDR — T2 subset
# ─────────────────────────────────────────────────────────────────────────────

def compute_edr_t2(ecg_epoch: Optional[np.ndarray],
                   r_peaks: list,
                   fs: float) -> dict:
    """
    T2 EDR features: edr_breath_rate, edr_regularity, edr_baseline_wander.

    Parameters
    ----------
    ecg_epoch : cleaned ECG epoch array (preprocessed_epochs.npy row)
    r_peaks   : list of R-peak sample indices within this epoch
    fs        : sampling frequency (Hz)

    Notes
    -----
    edr_baseline_wander must be extracted from the ORIGINAL ECG before
    Phase 1 DWT cleaning removes it. If unavailable, returns NaN.
    edr_breath_rate and edr_regularity are derived from R-wave amplitude
    modulation visible on the cleaned ECG.
    """
    keys = ["edr_breath_rate", "edr_regularity", "edr_baseline_wander"]
    nan_d = {k: np.nan for k in keys}

    if ecg_epoch is None or len(r_peaks) < 6:
        return nan_d

    try:
        rp = np.array(r_peaks, dtype=int)
        # R-wave amplitudes at peak positions
        valid = rp[(rp >= 0) & (rp < len(ecg_epoch))]
        if len(valid) < 6:
            return nan_d
        amps = ecg_epoch[valid].astype(float)

        # ── EDR breath rate via FFT of R-amplitude envelope ─────────────────
        # Resample amplitudes to uniform 1 Hz grid
        beat_times = valid.astype(float) / fs
        t_uniform  = np.arange(beat_times[0], beat_times[-1], 1.0 / fs)
        edr_signal = np.interp(t_uniform, beat_times, amps)
        fft_vals   = np.abs(np.fft.rfft(edr_signal - np.mean(edr_signal)))
        freqs      = np.fft.rfftfreq(len(edr_signal), d=1.0 / fs)
        breath_band = (freqs >= 0.10) & (freqs <= 0.50)   # 6–30 br/min
        if breath_band.sum() > 0:
            peak_f     = float(freqs[breath_band][np.argmax(fft_vals[breath_band])])
            breath_rate = float(peak_f * 60.0)
        else:
            breath_rate = np.nan

        # ── EDR regularity — breath-to-breath CV of R-amplitude ─────────────
        amp_cv = float(np.std(amps, ddof=1) / (np.mean(np.abs(amps)) + 1e-10) * 100.0)

        # ── Baseline wander (proxy from low-pass of ECG) ─────────────────────
        # NOTE: most informative BEFORE DWT cleaning; here we extract from
        # the already-cleaned signal as a degraded proxy.
        from scipy.signal import butter, filtfilt
        nyq  = fs / 2.0
        cutoff = min(0.5 / nyq, 0.99)
        b, a = butter(2, cutoff, btype='low')
        bw   = float(np.std(filtfilt(b, a, ecg_epoch.astype(float))))

        return {
            "edr_breath_rate"     : breath_rate,
            "edr_regularity"      : amp_cv,
            "edr_baseline_wander" : bw,
        }
    except Exception:
        return nan_d


# ─────────────────────────────────────────────────────────────────────────────
#  Context features — T2
# ─────────────────────────────────────────────────────────────────────────────

def compute_sleep_cycle_pos(epoch_idx: int,
                             n_epochs: int,
                             cycle_sec: float = 5400.0,
                             epoch_sec: float = 30.0) -> float:
    """
    Fractional position in 90-minute ultradian sleep cycle (0.0–1.0).
    N3 dominates 0–30% of cycle. REM dominates 70–100%.
    """
    t_sec     = epoch_idx * epoch_sec
    pos       = (t_sec % cycle_sec) / cycle_sec
    return float(pos)


def compute_hr_slope_epoch(rr_ms: list, fs: float = 125.0) -> float:
    """
    Within-epoch HR trend slope (bpm per minute).
    REM phasic: rapid HR increases. Sleep onset: gradual slowing.
    """
    if len(rr_ms) < 6:
        return np.nan
    try:
        rr = np.array(rr_ms, dtype=float)
        t  = np.cumsum(rr / 1000.0)   # seconds
        hr = 60000.0 / rr             # instantaneous HR
        # linear fit, convert slope from bpm/s → bpm/min
        slope = float(np.polyfit(t, hr, 1)[0]) * 60.0
        return slope
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  Main T2 compute entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_tier2(rr_ms: list,
                  t1_feats: dict,
                  ecg_epoch: Optional[np.ndarray],
                  r_peaks: list,
                  fs: float,
                  epoch_idx: int,
                  n_epochs: int,
                  epoch_sec: float = 30.0) -> dict:
    """
    Compute all T2 Confirmatory features for one epoch.

    Parameters
    ----------
    rr_ms     : RR intervals (ms) from Phase 2
    t1_feats  : dict from compute_tier1() — reuses lf/hf/tot
    ecg_epoch : ECG epoch array for EDR
    r_peaks   : R-peak indices within epoch
    fs        : sampling rate
    epoch_idx : epoch index (0-based)
    n_epochs  : total number of epochs
    epoch_sec : epoch duration in seconds

    Returns
    -------
    dict of T2 feature values
    """
    feats = {}

    # Time-domain
    feats.update(compute_t2_time(rr_ms))

    # Frequency (uses already-computed LS powers from T1)
    lf  = t1_feats.get("lf_power",    np.nan)
    hf  = t1_feats.get("hf_power",    np.nan)
    tot = t1_feats.get("total_power", np.nan)
    feats.update(compute_t2_freq(lf, hf, tot, rr_ms))

    # sd2 — long-range Poincaré (also T2 nonlinear)
    sd2 = t1_feats.get("sd2", np.nan)   # already in T1 poincare
    feats["sd2"] = sd2   # re-expose in T2 for clarity (no recompute)

    # DFA alpha2
    feats["dfa_alpha2"] = compute_dfa_alpha2(rr_ms)

    # PRSA
    feats["prsa_dc"]           = compute_prsa_dc(rr_ms)
    feats["prsa_ac"]           = compute_prsa_ac(rr_ms)   # T3 group but computed here
    feats["porta_asymmetry"]   = compute_porta_asymmetry(rr_ms)

    # EDR
    feats.update(compute_edr_t2(ecg_epoch, r_peaks, fs))

    # Context
    feats["sleep_cycle_pos"] = compute_sleep_cycle_pos(epoch_idx, n_epochs,
                                                        epoch_sec=epoch_sec)
    feats["hr_slope_epoch"]  = compute_hr_slope_epoch(rr_ms, fs)

    # hr_epoch_delta is computed across epochs in postprocess (needs prev epoch HR)
    # Placeholder NaN here; filled in by p3_postprocess.py
    feats["hr_epoch_delta"] = np.nan

    return feats
