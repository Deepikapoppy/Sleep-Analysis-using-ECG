"""
=============================================================================
phase0_dataset_management_testing/p0_config.py
Central configuration for the DEVICE-ONLY testing pipeline.
slpdb/hmc support has been intentionally removed from this testing package —
production code for those datasets lives in the main phase0_dataset_management
folder. This package only ever scans/loads/registers "device" JSON sessions.
=============================================================================
"""

import os

# Keep test outputs inside Testing_code even when a driver is launched from
# the project root.
TESTING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Single source of truth for the testing SQLite database path. This package
# must NEVER default to Database.create_database.DB_PATH / Database.db_manager
# .DB_PATH — those resolve to Sleepanalysis.db, the PRODUCTION/training
# database. Every p0_*.py module's db_path default pulls from here instead,
# so test runs can never silently write into production.
TEST_DB_PATH = "test_pipeline.db"

CONFIG = {
    # Kept as a literal (not a selector) so downstream code that still checks
    # config.get("dataset") keeps working without needing an if/else per file.
    "dataset" : "device",

    # ── TESTING DATABASE ────────────────────────────────────────────────────
    "db_path" : TEST_DB_PATH,

    # ── DEVICE PATHS ────────────────────────────────────────────────────────
    "local_device_path"     : r"C:\Users\admin\Downloads\database_sp\Datasets\device_json",
    "local_test_device_path": r"C:\Users\admin\Downloads\database_sp\Test_Dataset",
    "use_test_device_path"  : True,    # True → scan/load from local_test_device_path

    # ── RECORD SELECTION ────────────────────────────────────────────────────
    "record_name" : None,    # set per session at runtime
    "process_all" : True,

    # ── SIGNAL PARAMETERS ───────────────────────────────────────────────────
    "target_fs" : 125,
    "epoch_sec" : 30,

    # ── DEVICE (JSON) PARAMETERS ────────────────────────────────────────────
    "device_ecg_key"          : "ECG_CH_A",     # key holding the raw ECG samples
    "device_id_key"           : "admissionId",  # groups chunk files into one session
    "device_file_glob"        : "ECG_*.json",   # ONLY ECG chunks — other sensor
                                                  # files (SPO2_UNFILTERED_*,
                                                  # NISO101_*, *_data_*) share
                                                  # these folders but must be
                                                  # excluded, not just skipped
                                                  # on a missing-field basis
    "device_gap_tolerance_ms" : 500,
    "device_fs_tolerance_pct" : 2.0,

    # ── DWT PARAMETERS (used in Phase 1) ────────────────────────────────────
    "dwt_wavelet" : "db4",
    "dwt_level"   : 5,

    # ── OUTPUT ───────────────────────────────────────────────────────────────
    "results_dir" : os.path.join(TESTING_ROOT, "results_test"),
    "log_file"    : "pipeline.log",

    # ── PLOTTING (device format has no ground truth, so only used for
    #    predicted hypnograms in later phases — never a PSG comparison plot) ──
    "stage_labels" : {0: "Wake", 1: "N1", 2: "N2", 3: "N3 (SWS)", 4: "REM"},
    "stage_colors" : {
        0: "#e74c3c", 1: "#f39c12", 2: "#f1c40f", 3: "#2ecc71", 4: "#3498db",
    },
}

CONFIG["output_dir"] = os.path.join(
    CONFIG["results_dir"], CONFIG["dataset"], "plots", "per_subject"
)


def get_active_device_path(config: dict) -> str:
    """
    Returns local_test_device_path if config['use_test_device_path'] is True,
    otherwise local_device_path. All device code paths (scan, register, load)
    call this instead of reading config['local_device_path'] directly.
    """
    if config.get("use_test_device_path"):
        return config["local_test_device_path"]
    return config["local_device_path"]


def get_record_path(config: dict) -> str:
    """
    Device "records" are sessions made of many chunk files, not one base
    path + extension. This returns the device root folder for interface
    consistency; real chunk resolution always goes through
    build_device_manifest() / get_device_chunk_files() in p0_scan_datasets.py.
    """
    return get_active_device_path(config)


def make_subject_dirs(record_name: str, results_dir: str = "results_test",
                      dataset: str = "device") -> str:
    """
    Create and return the per-subject output directory:
        results_test/device/plots/per_subject/<session>/
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