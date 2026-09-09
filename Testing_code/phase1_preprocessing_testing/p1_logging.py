"""
=============================================================================
phase1_preprocessing/p1_logging.py
Centralised logger for all Phase-1 modules.
=============================================================================
"""

import os
import sys
import logging
from datetime import datetime


def setup_logger(name: str,
                 log_dir: str,
                 log_file: str = "phase1_pipeline.log",
                 level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a named logger that writes to:
      •  <log_dir>/<log_file>   — persistent log file
      •  stdout                 — terminal output

    Safe to call multiple times — will not add duplicate handlers.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger          # already set up — return as-is

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def log_phase_header(logger: logging.Logger,
                     record: str,
                     dataset: str,
                     fs_orig: float,
                     fs_target: float,
                     wavelet: str,
                     level: int) -> None:
    """Print a standardised banner at the start of Phase-1 for each record."""
    logger.info("=" * 65)
    logger.info("  PHASE 1 — ECG Pre-Processing (DWT Pipeline)")
    logger.info(f"  Record   : {record}")
    logger.info(f"  Dataset  : {dataset.upper()}")
    logger.info(f"  Fs orig  : {fs_orig} Hz  →  target: {fs_target} Hz")
    logger.info(f"  Wavelet  : {wavelet}   Level: {level}")
    logger.info(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)


def log_step(logger: logging.Logger, step_num: int, description: str) -> None:
    """Emit a clearly visible step separator line."""
    logger.info(f"--- Step {step_num}: {description} ---")
