"""
=============================================================================
phase0_dataset_management/p0_scan_datasets.py
Scans local dataset folders and returns sorted lists of valid record names.
No downloading — reads only what is already on disk.

Supported layouts
─────────────────
SLPDB   (WFDB format):
    slpdb_dataset/
        slp01a.hea   slp01a.dat   slp01a.st
        slp01b.hea   ...

HMC     (EDF format):
    hmc_dataset/
        SN146.edf    SN146_sleepscoring.txt
        SN147.edf    SN147_sleepscoring.txt
        ...
=============================================================================
"""

import os
import glob
import logging
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def scan_datasets(config: dict,
                  logger: logging.Logger = None) -> List[str]:
    """
    Discover all valid records for the dataset specified in `config["dataset"]`.
    Returns a sorted list of record-name strings (no file extensions).

    Raises FileNotFoundError if the dataset folder does not exist.
    Raises RuntimeError if no records are found.
    """
    dataset = config.get("dataset", "slpdb")
    if dataset == "hmc":
        return _scan_hmc(config, logger)
    return _scan_slpdb(config, logger)


# ─────────────────────────────────────────────────────────────────────────────
#  SLPDB scanner
# ─────────────────────────────────────────────────────────────────────────────

def _scan_slpdb(config: dict, logger) -> List[str]:
    path = config["local_slpdb_path"]
    _check_dir(path, "SLPDB", "local_slpdb_path")

    hea_files = glob.glob(os.path.join(path, "*.hea"))
    records   = []
    missing_ann = []

    for hf in hea_files:
        base = os.path.splitext(hf)[0]
        name = os.path.basename(base)
        if os.path.exists(base + ".st"):
            records.append(name)
        else:
            missing_ann.append(name)

    records = sorted(records)

    if logger:
        logger.info(f"[SLPDB] Scanned: {path}")
        logger.info(f"  Found {len(records)} valid records (signal + annotation)")
        if missing_ann:
            logger.warning(f"  {len(missing_ann)} .hea files lack .st annotation: "
                           f"{missing_ann[:5]}{'...' if len(missing_ann)>5 else ''}")

    if not records:
        raise RuntimeError(
            f"No valid SLPDB records (.hea + .st) found in: {path}"
        )
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  HMC scanner
# ─────────────────────────────────────────────────────────────────────────────

def _scan_hmc(config: dict, logger) -> List[str]:
    path = config["local_hmc_path"]
    _check_dir(path, "HMC", "local_hmc_path")

    edf_files = glob.glob(os.path.join(path, "*.edf"))
    records   = []
    missing_ann = []

    for ef in edf_files:
        stem = os.path.splitext(os.path.basename(ef))[0]
        if stem.lower().endswith("_sleepscoring"):
            continue          # skip annotation EDF files

        base      = os.path.splitext(ef)[0]
        ann_txt   = base + "_sleepscoring.txt"
        ann_notxt = base + "_sleepscoring"
        if os.path.exists(ann_txt) or os.path.exists(ann_notxt):
            records.append(stem)
        else:
            missing_ann.append(stem)

    records = sorted(records)

    if logger:
        logger.info(f"[HMC] Scanned: {path}")
        logger.info(f"  Found {len(records)} valid records (.edf + annotation)")
        if missing_ann:
            logger.warning(f"  {len(missing_ann)} .edf files lack annotation: "
                           f"{missing_ann[:5]}{'...' if len(missing_ann)>5 else ''}")

    if not records:
        raise RuntimeError(
            f"No valid HMC records (.edf + _sleepscoring.txt) found in: {path}\n"
            "Expected pairs like SN146.edf + SN146_sleepscoring.txt"
        )
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────

def _check_dir(path: str, label: str, config_key: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{label} folder not found: {path}\n"
            f"Update CONFIG['{config_key}'] in p0_config.py."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from phase0_dataset_management.p0_config import CONFIG
    from phase0_dataset_management.p0_logging import setup_logger

    logger = setup_logger("scan", log_dir="logs")
    for ds in ("slpdb", "hmc"):
        CONFIG["dataset"] = ds
        try:
            recs = scan_datasets(CONFIG, logger)
            print(f"{ds}: {len(recs)} records — first 5: {recs[:5]}")
        except Exception as exc:
            print(f"{ds}: {exc}")
