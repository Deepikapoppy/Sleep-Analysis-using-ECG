"""
=============================================================================
phase3_hrv_features/p3_tier3_features.py
T3 Resolver — 29 features
Hard confusion: N3/REM, N1/N2, N1/REM. Full ECG morphology + CRC. (~12%)

Features
────────
Nonlinear RR (10)
  prsa_ac            Acceleration capacity. High Wake/REM.
  guzik_asymmetry    Squared-distance Poincaré asymmetry. REM: >0.6.
  mse_scale2         Multiscale Entropy scale 2. N1/REM vs N2.
  mse_scale4         Multiscale Entropy scale 4. Paired with scale 2.
  higuchi_fd         Fractal dimension. FD→1=N3, FD→2=Wake. Needs ~15 RR.
  fuzzy_entropy      More stable than SampEn for short series (5–9 beats).
  recurrence_rate    RQA determinism. High in N3.
  rr_entropy_rate    Conditional entropy of symbolised RR. N3: low.
  apen               Approximate Entropy. Supplements SampEn for n<10.
  dfa_alpha2         Long-range fractal scaling (duplicated from T2 for use here)

EDR morphology (4)
  edr_r_amplitude    R-wave amplitude modulation → respiration (N3 vs REM).
  edr_qrs_area       More robust than R-amplitude under noise.
  edr_breath_depth   Tidal volume proxy. N3: deep. REM/N1: shallow.
  edr_apnea_index    Apnea proxy. Gates HRV during apneic epochs.

CRC (3)
  crc_cross_spectrum HRV–EDR cross-power. N3: coherence 0.8–0.95.
  crc_coherence_hf   MSC between RR and EDR in HF band. N3: near-perfect.
  crc_phase_sync     Phase locking value. N3: 0.7–0.9. REM: 0.2–0.4.

ECG morphology (5)
  r_amplitude_cv     R-wave amplitude CV. High Wake/REM, very low N3.
  r_amplitude_mean   N3: large uniform amplitudes. Wake: variable.
  qrs_duration       QRS width. Narrows in parasympathetic N3.
  pr_interval        Prolonged PR in N3 (vagal AV node slowing).
  hf_peaks_count     Count of distinct HF PSD peaks. REM/N1: multi-peaked.

Frequency (3)
  rr_spectral_entropy  Spread of spectral power. Wake/REM: high. N3: low.
  coherence_lf_hf      LF↔HF coupling. N3: decoupled. N2: partial.
  vlf_power            VLF (already computed in T1); re-exposed for T3 use.

Time (2)
  rr_triangular_index  RR histogram area / peak bin. High variability = Wake.
  nn50                 Absolute count |ΔRR|>50 ms. More stable than pNN50.
  nn20                 Absolute count |ΔRR|>20 ms. Complements nn50.

Frequency (1 more)
  ulf_ratio            Ultra-low freq ratio. Rises N2/N3 (10-epoch rolling).
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Nonlinear RR — T3
# ─────────────────────────────────────────────────────────────────────────────

def compute_guzik_asymmetry(rr_ms: list) -> float:
    """
    Guzik asymmetry index — squared-distance Poincaré method.
    REM: ratio > 0.6 (deceleration-dominant). N3: near-symmetric (~0.5).
    Reference: Guzik et al. 2006.
    """
    if len(rr_ms) < 5:
        return np.nan
    try:
        rr   = np.array(rr_ms, dtype=float)
        x, y = rr[:-1], rr[1:]
        # Deceleration points: above the line y=x
        mask_dec = y > x
        d_dec = np.sum((y[mask_dec] - x[mask_dec]) ** 2)
        d_tot = np.sum((y - x) ** 2)
        return float(d_dec / (d_tot + 1e-10))
    except Exception:
        return np.nan


def compute_mse(rr_ms: list, scale: int = 2, m: int = 2, r_coef: float = 0.15) -> float:
    """
    Multiscale Entropy at a given scale.
    Primary method to separate N1/REM from N2.
    N1/REM show elevated mid-scale complexity.
    """
    if len(rr_ms) < scale * 10:
        return np.nan
    try:
        rr = np.array(rr_ms, dtype=float)
        # Coarse-grain
        N    = len(rr) // scale
        cg   = np.array([np.mean(rr[i * scale:(i + 1) * scale]) for i in range(N)])
        if len(cg) < 10:
            return np.nan
        r = r_coef * np.std(cg)

        def count_templates(m_len):
            count = 0
            for i in range(N - m_len):
                tmpl = cg[i:i + m_len]
                for j in range(i + 1, N - m_len + 1):
                    if np.max(np.abs(cg[j:j + m_len] - tmpl)) <= r:
                        count += 1
            return count

        B = count_templates(m)
        A = count_templates(m + 1)
        return float(-np.log(A / B)) if B > 0 else np.nan
    except Exception:
        return np.nan


def compute_higuchi_fd(rr_ms: list, kmax: int = 6) -> float:
    """
    Higuchi Fractal Dimension. FD→1 = regular (N3), FD→2 = irregular (Wake).
    Only needs ~15 RR points. Top-5 ECG feature (BMC 2024).
    """
    if len(rr_ms) < 15:
        return np.nan
    try:
        x  = np.array(rr_ms, dtype=float)
        N  = len(x)
        L  = []
        ks = range(1, kmax + 1)
        for k in ks:
            Lk = []
            for m in range(1, k + 1):
                Lmk = 0.0
                n_max = int(np.floor((N - m) / k))
                for i in range(1, n_max):
                    Lmk += abs(x[m + i * k - 1] - x[m + (i - 1) * k - 1])
                Lmk = Lmk * (N - 1) / (n_max * k)
                Lk.append(Lmk)
            L.append(np.mean(Lk))
        valid = [(np.log(1.0 / k), np.log(l)) for k, l in zip(ks, L)
                 if l > 0 and np.isfinite(np.log(l))]
        if len(valid) < 3:
            return np.nan
        xs, ys = zip(*valid)
        return float(np.polyfit(xs, ys, 1)[0])
    except Exception:
        return np.nan


def compute_fuzzy_entropy(rr_ms: list, m: int = 2, r: float = 0.2, n: float = 2.0) -> float:
    """
    Fuzzy Entropy — more stable than SampEn for short series (~30 beats).
    Handles n_rr = 5–9 epochs. High Wake, low N3.
    """
    if len(rr_ms) < 5:
        return np.nan
    try:
        rr = np.array(rr_ms, dtype=float)
        N  = len(rr)
        sd = np.std(rr, ddof=1)
        if sd < 1e-10:
            return 0.0
        tol = r * sd

        def phi(m_len):
            total = 0.0
            count = 0
            for i in range(N - m_len):
                xi = rr[i:i + m_len]
                mu_i = np.mean(xi)
                for j in range(N - m_len):
                    if i == j:
                        continue
                    xj  = rr[j:j + m_len]
                    mu_j = np.mean(xj)
                    d   = np.max(np.abs((xi - mu_i) - (xj - mu_j)))
                    total += np.exp(-(d ** n) / tol)
                    count += 1
            return total / max(count, 1)

        phi_m  = phi(m)
        phi_m1 = phi(m + 1)
        return float(-np.log(phi_m1 / (phi_m + 1e-10)))
    except Exception:
        return np.nan


def compute_recurrence_rate(rr_ms: list, eps: float = 0.15) -> float:
    """
    Recurrence Quantification Analysis — recurrence rate.
    High determinism in N3 (periodic rhythm). N2 vs N3 discrimination.
    """
    if len(rr_ms) < 5:
        return np.nan
    try:
        rr   = np.array(rr_ms, dtype=float)
        sd   = np.std(rr, ddof=1)
        if sd < 1e-10:
            return 1.0
        tol   = eps * sd
        N     = len(rr)
        count = 0
        for i in range(N):
            for j in range(N):
                if i != j and abs(rr[i] - rr[j]) < tol:
                    count += 1
        return float(count / (N * (N - 1)))
    except Exception:
        return np.nan


def compute_rr_entropy_rate(rr_ms: list, n_sym: int = 5) -> float:
    """
    Conditional entropy of symbolised RR sequence.
    N3: low (predictable transitions). Complements perm_en.
    """
    if len(rr_ms) < 10:
        return np.nan
    try:
        rr     = np.array(rr_ms, dtype=float)
        # Symbolise into n_sym bins
        pcts   = np.linspace(0, 100, n_sym + 1)
        bins   = np.percentile(rr, pcts)
        sym    = np.digitize(rr, bins[1:-1])
        # Conditional entropy H(X_n | X_n-1)
        joint  = {}
        margin = {}
        for i in range(1, len(sym)):
            pair = (sym[i - 1], sym[i])
            joint[pair]        = joint.get(pair, 0) + 1
            margin[sym[i - 1]] = margin.get(sym[i - 1], 0) + 1
        total   = sum(joint.values())
        h_cond  = 0.0
        for (a, b), cnt in joint.items():
            p_joint = cnt / total
            p_a     = margin[a] / total
            h_cond -= p_joint * np.log2(p_joint / p_a + 1e-12)
        return float(h_cond)
    except Exception:
        return np.nan


def compute_apen(rr_ms: list, m: int = 2, r_coef: float = 0.2) -> float:
    """
    Approximate Entropy — supplements SampEn for n_rr < 10 epochs.
    Defined for n_rr ≥ 5.
    """
    if len(rr_ms) < 5:
        return np.nan
    try:
        rr = np.array(rr_ms, dtype=float)
        N  = len(rr)
        r  = r_coef * np.std(rr, ddof=1)

        def phi(m_len):
            count = np.zeros(N - m_len + 1)
            for i in range(N - m_len + 1):
                tmpl = rr[i:i + m_len]
                for j in range(N - m_len + 1):
                    if np.max(np.abs(rr[j:j + m_len] - tmpl)) <= r:
                        count[i] += 1
            c = count / (N - m_len + 1)
            return float(np.sum(np.log(c + 1e-10)) / (N - m_len + 1))

        return float(phi(m) - phi(m + 1))
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  EDR morphology — T3
# ─────────────────────────────────────────────────────────────────────────────

def compute_edr_t3(ecg_epoch: Optional[np.ndarray],
                   r_peaks: list,
                   fs: float) -> dict:
    """
    T3 EDR morphology: edr_r_amplitude, edr_qrs_area, edr_breath_depth,
    edr_apnea_index.
    """
    keys = ["edr_r_amplitude", "edr_qrs_area",
            "edr_breath_depth", "edr_apnea_index"]
    nan_d = {k: np.nan for k in keys}

    if ecg_epoch is None or len(r_peaks) < 6:
        return nan_d

    try:
        rp    = np.array(r_peaks, dtype=int)
        valid = rp[(rp >= 0) & (rp < len(ecg_epoch))]
        if len(valid) < 6:
            return nan_d
        amps  = ecg_epoch[valid].astype(float)

        # QRS area (window ±10 samples around each R-peak)
        half_w = int(fs * 0.08)   # 80 ms each side
        areas  = []
        for rpi in valid:
            lo = max(0, rpi - half_w)
            hi = min(len(ecg_epoch), rpi + half_w)
            areas.append(float(np.trapz(np.abs(ecg_epoch[lo:hi]))))

        # Amplitude modulation → breath depth proxy (std of amplitudes)
        breath_depth = float(np.std(amps, ddof=1))

        # Apnea index: count windows where mean amplitude < 30% of max for >5 s
        apnea_thresh = 0.30 * float(np.max(np.abs(amps))) if len(amps) > 0 else 0
        window_s     = int(5.0 * fs)
        n_apnea      = 0
        for start in range(0, len(ecg_epoch) - window_s, window_s):
            seg = ecg_epoch[start:start + window_s]
            if np.mean(np.abs(seg)) < apnea_thresh:
                n_apnea += 1

        return {
            "edr_r_amplitude" : float(np.std(amps, ddof=1) / (np.mean(np.abs(amps)) + 1e-10)),
            "edr_qrs_area"    : float(np.mean(areas)) if areas else np.nan,
            "edr_breath_depth": breath_depth,
            "edr_apnea_index" : float(n_apnea),
        }
    except Exception:
        return nan_d


# ─────────────────────────────────────────────────────────────────────────────
#  CRC — Cardiorespiratory Coupling
# ─────────────────────────────────────────────────────────────────────────────

def compute_crc(rr_ms: list,
                ecg_epoch: Optional[np.ndarray],
                r_peaks: list,
                fs: float) -> dict:
    """
    CRC features: crc_cross_spectrum, crc_coherence_hf, crc_phase_sync.
    N3: near-perfect coherence (RSA dominance). REM: low, fragmented.
    Reference: Thomas et al. Sleep 2014.
    """
    keys = ["crc_cross_spectrum", "crc_coherence_hf", "crc_phase_sync"]
    nan_d = {k: np.nan for k in keys}

    if ecg_epoch is None or len(rr_ms) < 10 or len(r_peaks) < 6:
        return nan_d

    try:
        from scipy import signal as sg
        rp   = np.array(r_peaks, dtype=int)
        valid = rp[(rp >= 0) & (rp < len(ecg_epoch))]
        if len(valid) < 6:
            return nan_d
        amps  = ecg_epoch[valid].astype(float)
        rr_arr = np.array(rr_ms, dtype=float)

        # Interpolate both series to uniform 4 Hz grid
        beat_t  = np.cumsum(rr_arr / 1000.0)
        beat_t -= beat_t[0]
        t_uni   = np.arange(0, beat_t[-1], 0.25)   # 4 Hz
        rr_uni  = np.interp(t_uni, beat_t, rr_arr)

        amp_t   = valid.astype(float) / fs
        amp_uni = np.interp(t_uni, amp_t, amps)

        n_min   = min(len(rr_uni), len(amp_uni))
        rr_uni  = rr_uni[:n_min]
        amp_uni = amp_uni[:n_min]
        if n_min < 16:
            return nan_d

        # Cross-spectrum
        freqs_cs, pxy = sg.csd(rr_uni, amp_uni, fs=4.0, nperseg=min(64, n_min))
        freqs_rr, pxx = sg.welch(rr_uni, fs=4.0, nperseg=min(64, n_min))
        freqs_ed, pyy = sg.welch(amp_uni, fs=4.0, nperseg=min(64, n_min))

        hf_mask = (freqs_cs >= 0.15) & (freqs_cs < 0.40)
        if hf_mask.sum() < 2:
            return nan_d

        cross_pwr  = float(np.mean(np.abs(pxy[hf_mask])))
        coherence  = float(np.mean(
            np.abs(pxy[hf_mask]) ** 2 /
            (pxx[hf_mask[:len(pxx)]] * pyy[hf_mask[:len(pyy)]] + 1e-20)
        ))

        # Phase synchrony (PLV) in HF band
        analytic_rr  = np.angle(sg.hilbert(rr_uni))
        analytic_amp = np.angle(sg.hilbert(amp_uni))
        phase_diff   = analytic_rr - analytic_amp
        plv          = float(np.abs(np.mean(np.exp(1j * phase_diff))))

        return {
            "crc_cross_spectrum": cross_pwr,
            "crc_coherence_hf"  : coherence,
            "crc_phase_sync"    : plv,
        }
    except Exception:
        return nan_d


# ─────────────────────────────────────────────────────────────────────────────
#  ECG Morphology — T3
# ─────────────────────────────────────────────────────────────────────────────

def compute_ecg_morphology(ecg_epoch: Optional[np.ndarray],
                            r_peaks: list,
                            fs: float) -> dict:
    """
    T3 ECG morphology: r_amplitude_cv, r_amplitude_mean,
    qrs_duration, pr_interval.
    """
    keys = ["r_amplitude_cv", "r_amplitude_mean", "qrs_duration", "pr_interval"]
    nan_d = {k: np.nan for k in keys}

    if ecg_epoch is None or len(r_peaks) < 4:
        return nan_d

    try:
        rp    = np.array(r_peaks, dtype=int)
        valid = rp[(rp >= 0) & (rp < len(ecg_epoch))]
        if len(valid) < 4:
            return nan_d
        amps = ecg_epoch[valid].astype(float)
        mean_amp = float(np.mean(amps))
        cv_amp   = float(np.std(amps, ddof=1) / (abs(mean_amp) + 1e-10))

        # QRS duration: width at 50% of amplitude for each peak
        half_qrs = int(fs * 0.06)   # 60 ms search window
        qrs_durs = []
        for rpi in valid:
            lo  = max(0, rpi - half_qrs)
            hi  = min(len(ecg_epoch), rpi + half_qrs)
            seg = np.abs(ecg_epoch[lo:hi])
            thresh = 0.5 * np.max(seg) if np.max(seg) > 0 else 0
            above  = np.where(seg > thresh)[0]
            if len(above) >= 2:
                qrs_durs.append((above[-1] - above[0]) / fs * 1000.0)   # ms
        qrs_dur = float(np.mean(qrs_durs)) if qrs_durs else np.nan

        # PR interval: distance from P-wave peak to R-peak
        # Simplified: search 50–200 ms before R-peak for a local max
        pr_ints = []
        p_win_lo = int(fs * 0.05)   # 50 ms
        p_win_hi = int(fs * 0.20)   # 200 ms
        for rpi in valid:
            lo = max(0, rpi - p_win_hi)
            hi = max(0, rpi - p_win_lo)
            if hi <= lo:
                continue
            p_seg = ecg_epoch[lo:hi]
            p_peak = int(np.argmax(np.abs(p_seg)))
            pr_ints.append((rpi - (lo + p_peak)) / fs * 1000.0)   # ms
        pr_int = float(np.mean(pr_ints)) if pr_ints else np.nan

        return {
            "r_amplitude_cv"  : cv_amp,
            "r_amplitude_mean": mean_amp,
            "qrs_duration"    : qrs_dur,
            "pr_interval"     : pr_int,
        }
    except Exception:
        return nan_d


# ─────────────────────────────────────────────────────────────────────────────
#  Frequency T3
# ─────────────────────────────────────────────────────────────────────────────

def compute_t3_freq(rr_ms: list) -> dict:
    """
    T3 freq features: hf_peaks_count, rr_spectral_entropy, coherence_lf_hf.
    """
    keys = ["hf_peaks_count", "rr_spectral_entropy", "coherence_lf_hf"]
    nan_d = {k: np.nan for k in keys}

    if len(rr_ms) < 10:
        return nan_d

    try:
        from astropy.timeseries import LombScargle
        from scipy.signal import find_peaks

        rr   = np.array(rr_ms, dtype=float) / 1000.0
        t    = np.cumsum(rr) - rr[0]
        if t[-1] < 20.0:
            return nan_d

        freq  = np.linspace(0.001, 0.5, 1000)
        power = LombScargle(t, rr - np.mean(rr)).power(freq)

        # HF peaks count
        hf_mask  = (freq >= 0.15) & (freq < 0.40)
        hf_power = power[hf_mask]
        peaks, _ = find_peaks(hf_power, prominence=0.01 * np.max(hf_power + 1e-10))
        hf_peaks = len(peaks)

        # Spectral entropy
        p_norm  = power / (np.sum(power) + 1e-10)
        p_nonz  = p_norm[p_norm > 0]
        sp_ent  = float(-np.sum(p_nonz * np.log2(p_nonz)))

        # LF–HF coherence via Hilbert phase difference
        lf_mask = (freq >= 0.04) & (freq < 0.15)
        if lf_mask.sum() > 0 and hf_mask.sum() > 0:
            # Signal filtered into LF and HF bands (interpolate to uniform)
            from scipy.signal import butter, filtfilt
            nyq   = 0.5 * 4.0   # 4 Hz resample
            rr_u  = np.interp(np.arange(0, t[-1], 0.25), t, rr)
            b_lf, a_lf = butter(2, [0.04 / nyq, 0.15 / nyq], btype='band')
            b_hf, a_hf = butter(2, [0.15 / nyq, 0.40 / nyq], btype='band')
            if len(rr_u) > 20:
                lf_sig  = filtfilt(b_lf, a_lf, rr_u)
                hf_sig  = filtfilt(b_hf, a_hf, rr_u)
                from scipy.signal import hilbert
                phi_lf  = np.angle(hilbert(lf_sig))
                phi_hf  = np.angle(hilbert(hf_sig))
                plv_lf_hf = float(np.abs(np.mean(np.exp(1j * (phi_lf - phi_hf)))))
            else:
                plv_lf_hf = np.nan
        else:
            plv_lf_hf = np.nan

        return {
            "hf_peaks_count"      : float(hf_peaks),
            "rr_spectral_entropy" : sp_ent,
            "coherence_lf_hf"     : plv_lf_hf,
        }
    except Exception:
        return nan_d


# ─────────────────────────────────────────────────────────────────────────────
#  Time T3
# ─────────────────────────────────────────────────────────────────────────────

def compute_t3_time(rr_ms: list) -> dict:
    """
    T3 time features: rr_triangular_index, nn50, nn20.
    """
    if len(rr_ms) < 4:
        return {"rr_triangular_index": np.nan, "nn50": np.nan, "nn20": np.nan}
    rr   = np.array(rr_ms, dtype=float)
    diff = np.abs(np.diff(rr))

    # Triangular index: total RR count / height of histogram peak
    counts, _ = np.histogram(rr, bins=max(5, len(rr) // 3))
    tri = float(len(rr) / (np.max(counts) + 1e-10))

    return {
        "rr_triangular_index": tri,
        "nn50"               : float(np.sum(diff > 50.0)),
        "nn20"               : float(np.sum(diff > 20.0)),
    }


def compute_ulf_ratio(rr_ms: list) -> float:
    """
    Ultra-low frequency ratio.
    Only meaningful as a 10-epoch rolling window feature (filled in postprocess).
    Per-epoch value = NaN; rolling fill in p3_postprocess.py.
    """
    return np.nan   # rolling in postprocess


# ─────────────────────────────────────────────────────────────────────────────
#  Main T3 compute entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_tier3(rr_ms: list,
                  ecg_epoch: Optional[np.ndarray],
                  r_peaks: list,
                  fs: float) -> dict:
    """
    Compute all 29 T3 Resolver features for one epoch.

    Parameters
    ----------
    rr_ms     : RR intervals (ms)
    ecg_epoch : ECG epoch array (preprocessed_epochs.npy row); None if unavailable
    r_peaks   : R-peak sample indices within epoch
    fs        : sampling frequency (Hz)

    Returns
    -------
    dict of T3 feature values
    """
    feats = {}

    # Nonlinear RR
    feats["guzik_asymmetry"]  = compute_guzik_asymmetry(rr_ms)
    feats["mse_scale2"]       = compute_mse(rr_ms, scale=2)
    feats["mse_scale4"]       = compute_mse(rr_ms, scale=4)
    feats["higuchi_fd"]       = compute_higuchi_fd(rr_ms)
    feats["fuzzy_entropy"]    = compute_fuzzy_entropy(rr_ms)
    feats["recurrence_rate"]  = compute_recurrence_rate(rr_ms)
    feats["rr_entropy_rate"]  = compute_rr_entropy_rate(rr_ms)
    feats["apen"]             = compute_apen(rr_ms)
    # prsa_ac already computed in T2 — exposed via T2 feats dict
    # dfa_alpha2 already in T2 — re-exposed via T2 feats dict
    # ulf_ratio: rolling window only, placeholder NaN
    feats["ulf_ratio"]        = np.nan

    # EDR morphology
    feats.update(compute_edr_t3(ecg_epoch, r_peaks, fs))

    # CRC
    feats.update(compute_crc(rr_ms, ecg_epoch, r_peaks, fs))

    # ECG morphology
    feats.update(compute_ecg_morphology(ecg_epoch, r_peaks, fs))

    # Frequency T3
    feats.update(compute_t3_freq(rr_ms))

    # Time T3
    feats.update(compute_t3_time(rr_ms))

    return feats
