"""
phase1_preprocessing/
══════════════════════════════════════════════════════════════════
FILE                        STEP   RESPONSIBILITY
──────────────────────────────────────────────────────────────────
p1_logging.py               —      setup_logger, log_phase_header, log_step
p1_load_ecg.py              1      load_ecg, extract_recording_start_time
p1_downsample.py            2      downsample (polyphase 125 Hz)
p1_baseline_wander.py       3a     remove_baseline_wander (zero cA_level)
p1_pli_removal.py           3b     remove_pli (zero cD1)
p1_soft_threshold.py        3c     soft_threshold_d2_d5 + reconstruct (IDWT)
p1_dwt_denoise.py           3      dwt_filter (calls 3a + 3b + 3c)
p1_polarity_correction.py   4      detect_and_apply_global_polarity
p1_zscore_normalisation.py  5      normalize
p1_segmentation.py          6      segment_epochs
p1_sqi.py                   7      compute_sqi_all_epochs, detect_bad_epochs
p1_plots.py                 8      all 6 diagnostic plots
p1_save.py                  9      save_preprocessed (.npy / .csv / .json)
p1_pipeline.py              ALL    run_phase1_record  ← main entry point
══════════════════════════════════════════════════════════════════
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .p1_logging              import setup_logger, log_phase_header, log_step
from .p1_load_ecg             import load_ecg, extract_recording_start_time
from .p1_downsample           import downsample
from .p1_baseline_wander      import remove_baseline_wander, get_baseline_band_hz
from .p1_pli_removal          import remove_pli, get_pli_band_hz
from .p1_soft_threshold       import soft_threshold_d2_d5, reconstruct
from .p1_dwt_denoise          import dwt_filter
from .p1_polarity_correction  import detect_and_apply_global_polarity
from .p1_zscore_normalisation import normalize
from .p1_segmentation         import segment_epochs
from .p1_sqi                  import compute_epoch_sqi, compute_sqi_all_epochs, detect_bad_epochs
from .p1_save                 import save_preprocessed
from .p1_pipeline             import run_phase1_record

__all__ = [
    # logging
    "setup_logger", "log_phase_header", "log_step",
    # load
    "load_ecg", "extract_recording_start_time",
    # step 2
    "downsample",
    # step 3a/3b/3c/3
    "remove_baseline_wander", "get_baseline_band_hz",
    "remove_pli", "get_pli_band_hz",
    "soft_threshold_d2_d5", "reconstruct",
    "dwt_filter",
    # step 4
    "detect_and_apply_global_polarity",
    # step 5
    "normalize",
    # step 6
    "segment_epochs",
    # step 7
    "compute_epoch_sqi", "compute_sqi_all_epochs", "detect_bad_epochs",
    # step 9
    "save_preprocessed",
    # master runner
    "run_phase1_record",
]
