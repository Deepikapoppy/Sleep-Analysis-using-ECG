"""
=============================================================================
phase4b_ml_classification_testing/p4b_logging.py
Logging setup for Phase 4b prediction — mirrors Phase 4 style.
=============================================================================
"""

import os
import logging


def setup_logger(name: str = "phase4b_test",
                 log_dir: str = "logs",
                 log_file: str = "phase4b_test.log") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    fmt = "%(asctime)s  [%(levelname)s]  %(message)s"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log_phase_header(logger: logging.Logger,
                     rec_name: str,
                     dataset: str,
                     n_epochs: int,
                     model_path: str) -> None:
    logger.info("=" * 70)
    logger.info("  PHASE 4b — RF Sleep Stage Prediction (deployment, no ground truth)")
    logger.info(f"  Session  : {rec_name}  |  Dataset: {dataset}")
    logger.info(f"  Epochs   : {n_epochs}")
    logger.info(f"  Model    : {model_path}")
    logger.info("=" * 70)


def log_step(logger: logging.Logger, step: int, label: str) -> None:
    logger.info(f"── Step {step}: {label}")
