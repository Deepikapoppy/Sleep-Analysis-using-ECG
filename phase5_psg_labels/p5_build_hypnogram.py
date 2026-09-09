"""
=============================================================================
phase5_psg_labels/p5_build_hypnogram.py
Step 2 — Map raw annotation labels → numeric stage ids, then build the
epoch-aligned PSG hypnogram array.

  map_stage           string label → int stage id  (-1 = unknown)
  build_psg_hypnogram annotation object → np.ndarray (n_ecg_epochs,)

FIX-3: Annotation offset detection and compensation.
  SLPDB .st files often start with ann.sample[0] > 0, meaning the first
  annotation does not coincide with the start of the ECG recording.
  Uncorrected, all pre-annotation epochs are forward-filled with Wake,
  inflating the Wake count.
  Fix: subtract ann_times[0] when it exceeds 0.5 × epoch_sec.

pandas FIX: .fillna(method='ffill') → .ffill() / .bfill() (matches
  Phase 3/4 style throughout).
=============================================================================
"""

import logging
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd

from phase4_sleep_classification.p4_plots import STAGE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
#  Label → numeric stage
# ─────────────────────────────────────────────────────────────────────────────

def map_stage(label: str, stage_map: dict) -> int:
    """
    Map a raw annotation string to a numeric stage id.
    Falls back to partial-key matching; returns -1 if no match found.

    stage_map example:
        {"W": 0, "1": 1, "2": 2, "3": 3, "4": 3, "R": 4}
    """
    label = label.strip().upper()
    if label in stage_map:
        return stage_map[label]
    for k in stage_map:
        if k in label:
            return stage_map[k]
    return -1


# ─────────────────────────────────────────────────────────────────────────────
#  Epoch-aligned PSG hypnogram
# ─────────────────────────────────────────────────────────────────────────────

def build_psg_hypnogram(ann,
                         fs_orig: float,
                         n_ecg_epochs: int,
                         config: dict,
                         logger: logging.Logger
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a PSG hypnogram aligned to ECG epoch indices.

    Parameters
    ----------
    ann          : WFDB annotation object (SLPDB) or list of tuples (HMC)
    fs_orig      : original recording sample-rate (Hz)
    n_ecg_epochs : number of ECG epochs from Phase 1 meta
    config       : pipeline config dict  (requires 'stage_map', 'epoch_sec')
    logger       : Logger

    Returns
    -------
    psg_hyp    : np.ndarray (n_ecg_epochs,) int  stage ids (0–4)
    ann_times  : np.ndarray  annotation onset times in seconds
                 (after offset correction)
    ann_stages : np.ndarray  mapped int stages (before epoch alignment)
    """
    stage_map = {k.upper(): v for k, v in config["stage_map"].items()}
    epoch_sec = config["epoch_sec"]   # 30

    if isinstance(ann, list):
        # HMC: list of (onset_sec, duration_sec, label) tuples — already 0-based
        ann_times  = np.array([r[0] for r in ann])
        ann_stages = np.array([map_stage(r[2], stage_map) for r in ann])
        ann_offset_sec = 0.0
    else:
        # SLPDB: WFDB annotation object
        ann_times  = np.array(ann.sample) / fs_orig
        ann_stages = np.array([map_stage(lbl, stage_map)
                                for lbl in ann.aux_note])

        # FIX-3: detect and compensate annotation lead-in offset
        ann_offset_sec = 0.0
        if len(ann_times) > 0 and ann_times[0] > 0.5 * epoch_sec:
            ann_offset_sec = float(ann_times[0])
            logger.warning(
                f"FIX-3 ANNOTATION OFFSET DETECTED: ann.sample[0] = "
                f"{ann.sample[0]} samples = {ann_offset_sec:.1f} s.  "
                f"ECG recording starts {ann_offset_sec:.1f} s BEFORE the "
                f"first PSG annotation — subtracting offset so "
                f"ann_times[0] → 0 and epoch alignment is preserved."
            )
            ann_times = ann_times - ann_offset_sec
        else:
            logger.info(
                f"FIX-3 annotation offset check: ann_times[0] = "
                f"{ann_times[0] if len(ann_times) > 0 else 'N/A':.1f}s — "
                f"within tolerance, no offset applied."
            )

    if len(ann_times) > 0:
        logger.info(
            f"Annotation time range (after offset correction): "
            f"{ann_times[0]:.1f}s – {ann_times[-1]:.1f}s  "
            f"(offset subtracted: {ann_offset_sec:.1f}s)"
        )

    # Map annotations → epoch indices
    psg_hyp = np.full(n_ecg_epochs, -1, dtype=int)
    for t, stage in zip(ann_times, ann_stages):
        if t < 0:
            continue
        ep_start = int(t // epoch_sec)
        for ep in range(ep_start, min(ep_start + 1, n_ecg_epochs)):
            psg_hyp[ep] = stage

    # Forward/back-fill gaps; remaining unknowns default to Wake (0)
    psg_hyp = (pd.Series(psg_hyp.astype(float))
                 .replace(-1, np.nan)
                 .ffill()
                 .bfill()
                 .fillna(0)
                 .values.astype(int))

    counts = Counter(psg_hyp)
    logger.info(
        f"PSG aligned: {len(psg_hyp)} epochs | "
        f"offset_corrected={ann_offset_sec:.1f}s | "
        f"distribution: "
        f"{ {STAGE_NAMES.get(k, k): v for k, v in counts.items()} }"
    )
    return psg_hyp, ann_times, ann_stages
