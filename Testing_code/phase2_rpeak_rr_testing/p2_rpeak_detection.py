"""
=============================================================================
phase2_rpeak_rr/p2_rpeak_detection.py
Step 2 — R-peak detection for DWT-preprocessed sleep ECG.

Architecture: Calibrated Cascade (Primary → Fallback → Safety Net)
─────────────────────────────────────────────────────────────────────
  STAGE 1 — calibrate_subject() — called ONCE per subject:
      Run neurokit + elgendi on first 4 clean epochs.
      Learn which method wins for THIS subject's ECG morphology.
      Compute subject-specific quality threshold (not fixed 0.80).
      Establish baseline HR for recall gate context.

  STAGE 2 — detect_rpeaks_all_methods() — called per epoch:
      PRIMARY   : neurokit or elgendi (winner from calibration)
                  → precision gate (ratio ≥ subj_threshold)
                  → recall gate    (beat count vs neighbour HR)
                  → BOTH pass? DONE.
      FALLBACK  : elgendi or neurokit (the other one)
                  → same dual gate
                  → pass? DONE.
      SAFETY NET: scipy_prominence
                  → ONLY if primary + fallback both return < 2 peaks
                  → result used regardless of quality score

Methods REMOVED vs original:
─────────────────────────────
  modified_pantompkins — REMOVED: redundant after DWT preprocessing.
      Pan-Tompkins was designed for raw unfiltered ECG (1985).
      DWT (db4 level-5) already handles baseline wander, PLI, and
      T-wave suppression that modified PT was compensating for.

  kalidas2017 — REMOVED: elgendi2010 covers its use case post-DWT.
      kalidas SWT-based method targets raw noisy/arrhythmic ECG.
      After DWT cleaning, elgendi two-MA handles residual morphology
      variation more reliably and faster (~38ms vs ~52ms).

Dual quality gate explained:
─────────────────────────────
  Gate 1 — PRECISION: quality_ratio ≥ subj_threshold
      "Of peaks I found, are their RR intervals plausible?"
      Catches false positives (artifact peaks).

  Gate 2 — RECALL: n_detected ≥ expected × tolerance
      "Did I find ENOUGH peaks vs recent heart rate?"
      Catches false negatives (missed beats from morphology shift).
      Tolerance scales with hr_confidence — wrong HR → loose gate.

Research basis:
─────────────────
  Makowski et al. (2021) Neuroinformatics
      — neurokit best on preprocessed ECG; quality-gated selection
  Elgendi et al. (2010) PLOS ONE
      — two-MA validated specifically on sleep ECG
  Behar et al. (2013) Physiological Measurement
      — two-stage approach for overnight recordings
  Llamedo & Martinez (2012) IEEE TBME
      — adaptive single method per subject > fixed multi-method
  Lipponen & Tarvainen (2019) Computer Methods Programs Biomed
      — quality_ratio ≥ 0.80 acceptance threshold (clinical Holter)
=============================================================================
"""

import warnings
import numpy as np
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# ── Default quality gate (overridden by subject calibration) ─────────────────
QUALITY_GATE = 0.80

# ── RR plausibility bounds ───────────────────────────────────────────────────
_RR_MIN_MS = 300.0    # HR < 200 bpm
_RR_MAX_MS = 2400.0   # HR >  25 bpm

# ── Plot colours (used by p2_plots) ─────────────────────────────────────────
RPEAK_COLOR = "#f39c12"
RPEAK_EDGE  = "#e67e22"

# Consistent method priority used across phase2 modules (plots, init)
METHOD_PRIORITY = ["neurokit", "elgendi2010", "scipy_prominence"]


# =============================================================================
#  SHARED HELPERS
# =============================================================================

def _remove_close_peaks(ep: np.ndarray,
                         peaks: np.ndarray,
                         min_rp: int) -> np.ndarray:
    """
    When two detected peaks are within min_rp samples of each other,
    keep only the taller one. Prevents double-counting one QRS complex.
    """
    if len(peaks) < 2:
        return peaks
    keep = [int(peaks[0])]
    for p in peaks[1:]:
        if p - keep[-1] < min_rp:
            if ep[p] > ep[keep[-1]]:
                keep[-1] = int(p)
        else:
            keep.append(int(p))
    return np.array(keep, dtype=int)


def _validate_peaks(ep: np.ndarray,
                    peaks: np.ndarray,
                    min_rp: int) -> np.ndarray:
    """
    Post-detection validation applied to ALL detectors uniformly:

    1. Remove out-of-bounds indices.
    2. Relative amplitude threshold: ≥ 40% of tallest peak in epoch.
       Rejects P-waves and T-waves on z-scored DWT-cleaned ECG.
    3. Absolute floor: 60th-percentile of epoch amplitude.
       Prevents low-amplitude noise being accepted on flat epochs.
    4. Remove too-close duplicates via _remove_close_peaks.
    """
    if len(peaks) < 2:
        return np.array([], dtype=int)

    peaks = peaks[(peaks >= 0) & (peaks < len(ep))]
    if len(peaks) == 0:
        return np.array([], dtype=int)

    peak_amps  = ep[peaks]
    max_amp    = float(np.max(peak_amps))
    rel_thresh = 0.40 * max_amp
    abs_floor  = float(np.percentile(ep, 60))
    thresh     = max(rel_thresh, abs_floor)
    peaks      = peaks[peak_amps > thresh]

    return _remove_close_peaks(ep, peaks, min_rp)


def _quick_rr_quality(peaks: np.ndarray,
                       fs: float) -> tuple:
    """
    Fast RR quality estimate — no full artifact correction.

    Returns
    -------
    n_valid : int   — number of RR intervals within 300–2400 ms
    ratio   : float — n_valid / total RR intervals detected
    """
    if len(peaks) < 2:
        return 0, 0.0
    rr_ms   = np.diff(peaks) / fs * 1000.0
    valid   = rr_ms[(rr_ms >= _RR_MIN_MS) & (rr_ms <= _RR_MAX_MS)]
    n_valid = len(valid)
    ratio   = n_valid / max(len(rr_ms), 1)
    return n_valid, ratio


def _quality_score(peaks: np.ndarray,
                   ep: np.ndarray,
                   fs: float) -> float:
    """
    Composite quality score = n_valid_rr × quality_ratio.

    Penalises methods that detect many peaks but most are artifacts.
    Example: 43 peaks, 5 valid RR → score = 5 × (5/42) = 0.60
             40 peaks, 38 valid RR → score = 38 × (38/39) = 37.0  ← wins
    """
    if len(peaks) < 2:
        return 0.0
    n_valid, ratio = _quick_rr_quality(peaks, fs)
    return float(n_valid) * ratio


def _check_recall(peaks: np.ndarray,
                  fs: float,
                  epoch_len: int,
                  neighbor_hr_bpm: float = None,
                  hr_confidence: float = 0.0) -> bool:
    """
    Gate 2: Did the detector find ENOUGH peaks?

    Catches morphology-shift misses that Gate 1 (precision) cannot:
    e.g. neurokit finds 9 peaks, all with plausible RR → ratio = 1.0
    but neighbour HR = 68 bpm → expect ~34 beats → 9 is clearly wrong.

    Two checks:
    ───────────
    Check A — absolute physiological floor (always active):
        At max RR = 2400 ms (25 bpm), minimum beats in epoch = floor(sec/2.4)
        For 30-sec epoch: minimum = 12 beats.

    Check B — neighbour HR context (active when hr_confidence ≥ 0.3):
        Expected beats = epoch_sec × (neighbor_hr / 60)
        Minimum accepted = expected × tolerance_floor
        tolerance_floor = 0.40 + 0.30 × hr_confidence
          → conf=1.0: floor=0.70 (tight,  HR is reliable)
          → conf=0.5: floor=0.55 (medium)
          → conf=0.3: floor=0.49 (loose, barely above absolute)
        This means wrong/uncertain HR → gate relaxes automatically.
    """
    epoch_sec  = epoch_len / fs
    n_detected = len(peaks)

    # Check A — absolute floor, always applies
    if n_detected < int(epoch_sec / 2.4):
        return False

    # Check B — neighbour HR context
    if neighbor_hr_bpm is None or hr_confidence < 0.3:
        # No reliable context — pass on Check B
        return True

    tolerance    = 0.40 + (0.30 * hr_confidence)
    min_expected = (epoch_sec * neighbor_hr_bpm / 60.0) * tolerance
    return n_detected >= min_expected


# =============================================================================
#  INDIVIDUAL DETECTORS
# =============================================================================

def _detect_neurokit(ep: np.ndarray,
                     fs: float,
                     min_rp: int) -> np.ndarray:
    """
    NeuroKit2 default gradient+threshold detector.
    Best on DWT-cleaned ECG with flat baseline and sharp QRS.
    Expected primary method for majority of SLPDB subjects.
    """
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(ep, sampling_rate=fs, method="neurokit")
        return _validate_peaks(
            ep, np.asarray(info["ECG_R_Peaks"], dtype=int), min_rp
        )
    except Exception:
        return np.array([], dtype=int)


def _detect_elgendi(ep: np.ndarray,
                    fs: float,
                    min_rp: int) -> np.ndarray:
    """
    Elgendi (2010) two-moving-average detector.
    Validated specifically on sleep ECG (PLOS ONE 2010).
    Morphology-agnostic: works on wide QRS, PVC, axis-shifted beats.
    Expected primary method for subjects with low-amplitude or
    morphologically variable QRS.
    """
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(ep, sampling_rate=fs, method="elgendi2010")
        return _validate_peaks(
            ep, np.asarray(info["ECG_R_Peaks"], dtype=int), min_rp
        )
    except Exception:
        return np.array([], dtype=int)


def _detect_scipy(ep: np.ndarray,
                  fs: float,
                  min_rp: int) -> np.ndarray:
    """
    scipy find_peaks prominence-based detector.
    NO physiological model — purely signal prominence.
    prominence=0.3 recalibrated for DWT z-scored ECG
    (raw ECG used 0.5; after DWT+z-score, QRS amplitudes are more
    uniform and compressed, so 0.5 rejects valid peaks).

    SAFETY NET ONLY — called when both neurokit and elgendi
    return fewer than 2 peaks. Result used regardless of quality.
    """
    try:
        peaks, _ = find_peaks(
            ep,
            distance=int(0.30 * fs),
            prominence=0.3
        )
        return _validate_peaks(ep, peaks, min_rp)
    except Exception:
        return np.array([], dtype=int)


# =============================================================================
#  STAGE 1 — SUBJECT CALIBRATION
#  Call once per subject before the main epoch loop.
# =============================================================================

def calibrate_subject(epochs: np.ndarray,
                      bad_mask: np.ndarray,
                      fs: float,
                      n_cal: int = 4) -> dict:
    """
    Run neurokit + elgendi on the first n_cal clean epochs.
    Determine which method better suits this subject's ECG morphology.
    Compute a subject-specific quality threshold.

    Why calibration instead of fixed order?
    ────────────────────────────────────────
    Llamedo & Martinez (2012): adaptive single method per subject
    outperforms fixed multi-method cascade across overnight recordings.
    Subject ECG varies by: body position, lead placement, BMI, cardiac
    axis, sleep disorder type. No single method wins for all subjects.

    Parameters
    ----------
    epochs   : (n_epochs, samples_per_epoch) DWT-cleaned array
    bad_mask : bool array, True = epoch flagged bad by Phase 1
    fs       : sampling frequency (Hz)
    n_cal    : number of clean epochs to use for calibration (default 4)

    Returns
    -------
    dict:
        primary           : "neurokit" or "elgendi2010"
        subj_threshold    : float  — subject-specific quality gate
        baseline_hr       : float or None  — median HR from cal epochs
        cal_score_nk      : float  — mean quality score for neurokit
        cal_score_elgendi : float  — mean quality score for elgendi
    """
    nk_scores = []
    el_scores = []
    cal_hrs   = []
    min_rp    = int(0.20 * fs)
    cal_count = 0

    for idx, epoch in enumerate(epochs):
        if bad_mask[idx]:
            continue
        if cal_count >= n_cal:
            break

        ep = epoch.copy()
        # Polarity safety net
        if np.abs(np.percentile(ep, 1)) > np.abs(np.percentile(ep, 99)):
            ep = -ep

        nk_peaks = _detect_neurokit(ep, fs, min_rp)
        el_peaks = _detect_elgendi(ep, fs, min_rp)

        nk_score = _quality_score(nk_peaks, ep, fs)
        el_score = _quality_score(el_peaks, ep, fs)

        nk_scores.append(nk_score)
        el_scores.append(el_score)

        # Collect HR from whichever method scored better
        better = nk_peaks if nk_score >= el_score else el_peaks
        if len(better) >= 2:
            rr    = np.diff(better) / fs * 1000.0
            valid = rr[(rr >= 300.0) & (rr <= 2400.0)]
            if len(valid) > 0:
                cal_hrs.append(60000.0 / float(np.mean(valid)))

        cal_count += 1

    mean_nk = float(np.mean(nk_scores)) if nk_scores else 0.0
    mean_el = float(np.mean(el_scores)) if el_scores else 0.0

    # Primary = method with higher mean quality score across cal epochs
    primary = "neurokit" if mean_nk >= mean_el else "elgendi2010"

    # Subject-specific threshold:
    # mean score of winning method minus one std dev, floor at 0.65.
    # More conservative than fixed 0.80 for subjects with naturally
    # lower quality scores (e.g. noisy lead, irregular morphology).
    winning_scores = nk_scores if primary == "neurokit" else el_scores
    if winning_scores:
        subj_threshold = max(
            0.65,
            float(np.mean(winning_scores)) - float(np.std(winning_scores))
        )
    else:
        subj_threshold = QUALITY_GATE   # fallback to global default

    return {
        "primary"           : primary,
        "subj_threshold"    : round(subj_threshold, 3),
        "baseline_hr"       : float(np.median(cal_hrs)) if cal_hrs else None,
        "cal_score_nk"      : round(mean_nk, 3),
        "cal_score_elgendi" : round(mean_el, 3),
    }


# =============================================================================
#  STAGE 2 — PER-EPOCH DETECTION  (main entry point)
# =============================================================================

def detect_rpeaks_all_methods(epoch: np.ndarray,
                               fs: float,
                               epoch_idx: int = 0,
                               subject_config: dict = None,
                               neighbor_hr_bpm: float = None,
                               hr_confidence: float = 0.0) -> dict:
    """
    Calibrated cascade: Primary → Fallback → Safety Net.

    Flow
    ────
    Step 1 — PRIMARY (neurokit or elgendi per subject_config):
        Run detector.
        Apply dual gate:
          Gate 1 (precision): ratio ≥ subj_threshold
          Gate 2 (recall):    beat count vs neighbour HR
        Both pass → RETURN immediately.
        One fails → continue to fallback.

    Step 2 — FALLBACK (the other of neurokit / elgendi):
        Same dual gate.
        Pass → RETURN.
        Fail → continue to safety net.
        Note: best result seen so far is always tracked.
              If fallback fails gate but scored better than primary,
              fallback result is still carried as best_peaks.

    Step 3 — SAFETY NET (scipy):
        Called ONLY if both primary + fallback returned < 2 peaks.
        Result used regardless of quality score.
        Marks stop_reason = "scipy_safety_net" for diagnostics.

    Parameters
    ----------
    epoch          : (samples,) DWT-cleaned, z-scored ECG epoch
    fs             : sampling frequency (Hz)
    epoch_idx      : epoch index for diagnostics (not used in logic)
    subject_config : dict from calibrate_subject(); None = use defaults
    neighbor_hr_bpm: mean HR from recent 2-3 valid epochs (optional)
    hr_confidence  : 0.0–1.0, how reliable neighbor_hr_bpm is

    Returns
    -------
    dict:
        best_peaks   : np.ndarray — R-peak sample indices
        best_method  : str
        all_peaks    : {method: ndarray} — empty if method not reached
        all_counts   : {method: int}
        all_scores   : {method: float}
        methods_run  : int  — 1, 2, or 3 (cascade depth)
        stop_reason  : str  — gate_passed@<method> / scipy_safety_net
                              / exhausted
    """
    ep        = epoch.copy()
    min_rp    = int(0.20 * fs)
    epoch_len = len(ep)

    # Polarity safety net
    if np.abs(np.percentile(ep, 1)) > np.abs(np.percentile(ep, 99)):
        ep = -ep

    # Determine primary method and threshold from calibration
    if subject_config is not None:
        primary   = subject_config["primary"]
        threshold = subject_config["subj_threshold"]
    else:
        primary   = "neurokit"
        threshold = QUALITY_GATE

    fallback = "elgendi2010" if primary == "neurokit" else "neurokit"

    method_names = [primary, fallback, "scipy_prominence"]
    all_peaks    = {m: np.array([], dtype=int) for m in method_names}
    all_scores   = {m: 0.0                     for m in method_names}
    all_counts   = {m: 0                        for m in method_names}

    detectors = {
        "neurokit"         : _detect_neurokit,
        "elgendi2010"      : _detect_elgendi,
        "scipy_prominence" : _detect_scipy,
    }

    best_peaks  = np.array([], dtype=int)
    best_method = "failed"
    best_score  = 0.0
    methods_run = 0
    stop_reason = "exhausted"

    # ── Step 1 & 2: Primary then Fallback ────────────────────────────────────
    for method in [primary, fallback]:
        peaks    = detectors[method](ep, fs, min_rp)
        score    = _quality_score(peaks, ep, fs)
        _, ratio = _quick_rr_quality(peaks, fs)

        all_peaks[method]  = peaks
        all_scores[method] = score
        all_counts[method] = len(peaks)
        methods_run       += 1

        # Always track best result seen so far
        if score > best_score and len(peaks) >= 2:
            best_peaks  = peaks
            best_method = method
            best_score  = score

        # Dual gate check
        if len(peaks) >= 2:
            precision_ok = (ratio >= threshold)
            recall_ok    = _check_recall(
                peaks, fs, epoch_len,
                neighbor_hr_bpm=neighbor_hr_bpm,
                hr_confidence=hr_confidence
            )
            if precision_ok and recall_ok:
                stop_reason = f"gate_passed@{method}"
                break
            # precision OK but recall failed → continue cascade
            # (next method may detect the missed beats)

    # ── Step 3: scipy safety net ─────────────────────────────────────────────
    # Only called when BOTH primary and fallback returned < 2 peaks.
    # If either returned ≥ 2 peaks (even if gates failed), best_peaks
    # already holds the better result — scipy not needed.
    if len(best_peaks) < 2:
        peaks = _detect_scipy(ep, fs, min_rp)
        all_peaks["scipy_prominence"]  = peaks
        all_scores["scipy_prominence"] = _quality_score(peaks, ep, fs)
        all_counts["scipy_prominence"] = len(peaks)
        methods_run += 1

        if len(peaks) >= 2:
            best_peaks  = peaks
            best_method = "scipy_prominence"
            stop_reason = "scipy_safety_net"

    return {
        "best_peaks"  : best_peaks,
        "best_method" : best_method,
        "all_peaks"   : all_peaks,
        "all_counts"  : all_counts,
        "all_scores"  : all_scores,
        "methods_run" : methods_run,
        "stop_reason" : stop_reason,
    }