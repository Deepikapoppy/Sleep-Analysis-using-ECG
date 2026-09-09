"""
=============================================================================
phase2_rpeak_rr/p2_logging.py
Centralised logger for all Phase-2 modules.
=============================================================================
"""

import os
import sys
import logging
from datetime import datetime


def setup_logger(name: str,
                 log_dir: str,
                 log_file: str = "phase2_pipeline.log",
                 level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a named logger writing to file + stdout.
    Safe to call multiple times — no duplicate handlers added.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

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
                     n_epochs: int,
                     fs: float) -> None:
    logger.info("=" * 65)
    logger.info("  PHASE 2 — R-Peak Detection & RR Interval Extraction")
    logger.info(f"  Record   : {record}")
    logger.info(f"  Dataset  : {dataset.upper()}")
    logger.info(f"  Epochs   : {n_epochs}  |  Fs: {fs} Hz")
    # Derive current method list from the detector module when available
    try:
        from phase2_rpeak_rr.p2_rpeak_detection import METHOD_PRIORITY
    except Exception:
        try:
            from .p2_rpeak_detection import METHOD_PRIORITY
        except Exception:
            METHOD_PRIORITY = ["neurokit", "elgendi2010", "scipy_prominence"]

    methods_str = " | ".join(METHOD_PRIORITY)
    logger.info(f"  Methods  : {methods_str}")
    logger.info(f"  Mode     : PARALLEL (all {len(METHOD_PRIORITY)} run, best selected)")
    logger.info(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)


def log_step(logger: logging.Logger, step_num: int, description: str) -> None:
    logger.info(f"--- Step {step_num}: {description} ---")
