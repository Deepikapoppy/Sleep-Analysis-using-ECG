import os
import sys

# .../database_sp/Testing_code/phase0_dataset_management_testing/__init__.py
_PACKAGE_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../Testing_code/phase0_dataset_management_testing
_TESTING_ROOT = os.path.dirname(_PACKAGE_DIR)                 # .../Testing_code
_PROJECT_ROOT = os.path.dirname(_TESTING_ROOT)                # .../database_sp  <- Database/ lives here

# Insert BOTH: _PROJECT_ROOT so `import Database` resolves regardless of cwd
# or how this package was imported (python -m unittest, pytest, a script in
# a different folder, etc.), and _TESTING_ROOT so this package itself is
# importable as a top-level name the same way. Without _PROJECT_ROOT here,
# only scripts that do their own path-fixing before importing this package
# (like run_phase0_test.py) could ever find Database — anything else
# (`python -m unittest phase0_dataset_management_testing...`) would fail
# with "ModuleNotFoundError: No module named 'Database'".
for _p in (_PROJECT_ROOT, _TESTING_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .p0_config import CONFIG, get_record_path, make_subject_dirs, get_active_device_path
from .p0_logging import setup_logger, log_phase_header, setup_environment
from .p0_scan_datasets import (
    scan_datasets, build_device_manifest, get_device_chunk_files
)
from .p0_register_records import register_all_records
from .p0_metadata_extraction import extract_all_metadata, _fs_from_chunk
from .p0_inspect_plots import run_phase0_record, load_record

__all__ = [
    "CONFIG", "get_record_path", "make_subject_dirs", "get_active_device_path",
    "setup_logger", "log_phase_header", "setup_environment",
    "scan_datasets", "build_device_manifest", "get_device_chunk_files",
    "register_all_records", "extract_all_metadata", "_fs_from_chunk",
    "run_phase0_record", "load_record",
]