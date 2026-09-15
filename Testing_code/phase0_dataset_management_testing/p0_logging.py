"""
=============================================================================
phase0_dataset_management/p0_logging.py
Shared logging utilities used across all Phase 0 (and later phase) modules.

Provides:
  - setup_logger(name, log_dir)  -> configured logging.Logger
        Writes to both console (INFO+) and a rotating-free timestamped file
        under log_dir (DEBUG+). Safe to call multiple times with the same
        name — won't duplicate handlers.
  - log_phase_header(logger, phase_name)
        Prints a visually distinct banner in the log when a new phase
        starts, so long pipeline runs are easy to scan through.
  - setup_environment()
        One-shot call for reproducibility / cleanliness: seeds numpy's
        legacy global RNG, quiets noisy third-party warnings that would
        otherwise clutter the log (wfdb/mne/pywt deprecation spam), and
        ensures matplotlib uses a non-interactive backend for headless runs.
=============================================================================
"""

import os
import sys
import logging
import warnings
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
#  Logger setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_dir: str = "logs",
                 log_file: str | None = None,
                 level: int = logging.DEBUG,
                 console_level: int = logging.INFO) -> logging.Logger:
    """
    Create (or fetch) a named logger with:
      - a console handler at `console_level` (concise, no timestamp clutter)
      - a file handler at DEBUG writing to log_dir/<name>_<timestamp>.log by
        default, or to the custom file name if `log_file` is provided.

    Idempotent: calling this again with the same `name` returns the same
    logger without adding duplicate handlers (important since several
    p0_*.py modules each call setup_logger independently in their
    `if __name__ == "__main__":` blocks).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False   # don't double-log via the root logger

    if logger.handlers:
        # Already configured (e.g. re-imported in the same process) — reuse.
        return logger

    os.makedirs(log_dir, exist_ok=True)
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_file)

    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter(
        fmt="[%(levelname)s] %(message)s",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.debug(f"Logger '{name}' initialized -> {log_path}")
    return logger


# ─────────────────────────────────────────────────────────────────────────────
#  Phase banner
# ─────────────────────────────────────────────────────────────────────────────

def log_phase_header(logger: logging.Logger, phase_name: str,
                     width: int = 78) -> None:
    """
    Print a visually distinct banner marking the start of a pipeline phase,
    e.g. log_phase_header(logger, "PHASE 0 — Dataset Management").
    Purely cosmetic — makes long multi-phase run logs easy to scan.
    """
    bar = "=" * width
    logger.info(bar)
    logger.info(phase_name.center(width))
    logger.info(bar)


# ─────────────────────────────────────────────────────────────────────────────
#  Environment setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_environment(seed: int = 42, quiet_warnings: bool = True) -> None:
    """
    One-shot environment configuration, call once near the top of a run:
      - seeds numpy's global RNG for reproducibility (R-peak jitter checks,
        any bootstrap/resampling steps downstream, etc.)
      - suppresses common noisy third-party warnings (wfdb/pywt/mne
        DeprecationWarning / RuntimeWarning spam) so real problems in the
        log don't get buried
      - forces matplotlib to the non-interactive "Agg" backend, since this
        pipeline only ever saves plots to disk (never displays them) and
        may run headless / over SSH
    """
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    if quiet_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                module="wfdb")
        warnings.filterwarnings("ignore", category=UserWarning,
                                module="pywt")

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        pass
