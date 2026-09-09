"""
=============================================================================
phase0_dataset_management/p0_config.py
Central configuration for the entire Sleep Analysis pipeline.
Edit the paths and selector here — nothing else needs to change.
=============================================================================
"""

import os

# ─── EDIT THESE TWO LINES ─────────────────────────────────────────────────────
# Default base (unused when absolute paths below are set)
_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "datasets")

CONFIG = {
    # ── DATASET SELECTOR ──────────────────────────────────────────────────────
    # "slpdb"  →  original SLPDB pipeline (WFDB .hea/.dat/.st)
    # "hmc"    →  Haaglanden Medisch Centrum (EDF + _sleepscoring.txt)
    "dataset"  : "hmc",

    # ── LOCAL PATHS (already downloaded — no internet needed) ─────────────────
    # Updated local dataset paths (absolute locations)
    "local_slpdb_path" : r"C:\Users\admin\Downloads\database_sp\Datasets\slpdb",
    "local_hmc_path"   : r"C:\Users\admin\Downloads\database_sp\Datasets\haaglanden-medisch-centrum-sleep-staging-database-1.0.0\haaglanden-medisch-centrum-sleep-staging-database-1.0.0\recordings",

    # ── RECORD SELECTION ──────────────────────────────────────────────────────
    "record_name" : None,    # set per subject at runtime
    "process_all" : True,    # True = iterate all discovered records

    # ── SIGNAL PARAMETERS ─────────────────────────────────────────────────────
    "target_fs"        : 125,
    "epoch_sec"        : 30,
    "ecg_channel"      : 0,          # fallback index for SLPDB
    "hmc_ecg_keywords" : ["ECG", "EKG", "ecg", "ekg"],

    # ── DWT PARAMETERS (used in Phase 1) ──────────────────────────────────────
    "dwt_wavelet" : "db4",
    "dwt_level"   : 5,

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    "results_dir" : "results",
    "log_file"    : "pipeline.log",

   

    # ── SLEEP STAGE MAPS ──────────────────────────────────────────────────────
    "stage_map" : {
        "W"  : 0,   # Wake
        "1"  : 1,   # NREM Stage 1
        "2"  : 2,   # NREM Stage 2
        "3"  : 3,   # NREM Stage 3 (SWS)
        "4"  : 3,   # NREM Stage 4 → merged into N3
        "R"  : 4,   # REM
        "MT" : 0,   # Movement time → Wake
        "?"  : -1,  # Unknown / artifact
    },
    "hmc_stage_map" : {
        "SLEEP STAGE W"  : 0,
        "SLEEP STAGE N1" : 1,
        "SLEEP STAGE N2" : 2,
        "SLEEP STAGE N3" : 3,
        "SLEEP STAGE R"  : 4,
        "SLEEP STAGE REM": 4,
        "W"              : 0,
        "N1"             : 1,
        "N2"             : 2,
        "N3"             : 3,
        "R"              : 4,
        "REM"            : 4,
        "MOVEMENT"       : 0,
        "UNSCORED"       : -1,
    },
    "stage_labels" : {0: "Wake", 1: "N1", 2: "N2", 3: "N3 (SWS)", 4: "REM"},
    "stage_colors" : {
        0: "#e74c3c",   # Wake  — red
        1: "#f39c12",   # N1    — orange
        2: "#f1c40f",   # N2    — yellow
        3: "#2ecc71",   # N3    — green
        4: "#3498db",   # REM   — blue
    },
}
 # Dynamically set output_dir based on selected dataset
CONFIG["output_dir"] = os.path.join(
    "results", CONFIG["dataset"], "plots", "per_subject"
    )

def get_record_path(config: dict) -> str:
    """Return full base path to a record (no file extension)."""
    dataset = config.get("dataset", "slpdb")
    if dataset == "hmc":
        return os.path.join(config["local_hmc_path"], config["record_name"])
    return os.path.join(config["local_slpdb_path"], config["record_name"])


def make_subject_dirs(record_name: str, results_dir: str = "results",
                      dataset: str = "slpdb") -> str:
    """
    Create and return the per-subject output directory:
        results/<dataset>/plots/per_subject/<record>/
    Also ensures metrics/, predictions/, and plots/overall/ exist.
    """
    ds_root     = os.path.join(results_dir, dataset)
    subject_dir = os.path.join(ds_root, "plots", "per_subject", record_name)
    for path in [
        subject_dir,
        os.path.join(ds_root, "metrics"),
        os.path.join(ds_root, "predictions"),
        os.path.join(ds_root, "plots", "overall"),
    ]:
        os.makedirs(path, exist_ok=True)
    return subject_dir
