"""
=============================================================================
dashboard_app.py — ECG Sleep-Stage Clinical Analytics Console  (v3)
Interactive, dark-themed, READ-ONLY visualization/analysis layer for the
ECG sleep-staging TESTING pipeline (phase0_dataset_management_testing ...
phase5_final_report_testing).

This dashboard never calls, imports, or modifies your phaseN_*_testing
processing packages, and never writes into Test_dataset or results_test.
It only reads (a) the raw device JSON chunks under TEST_DATASET_ROOT to
discover every Admission ID that exists, and (b) whatever files each phase
has already written to disk (npy / csv / json / png) under RESULTS_ROOT for
that ID. Nothing here re-runs any phase.

ADMISSION ID DISCOVERY (v3 fix)
--------------------------------
The Admission ID selector is now populated by scanning TEST_DATASET_ROOT
directly, using the same admissionId-grouping logic
phase0_dataset_management_testing/p0_scan_datasets.py uses (legacy single-
chunk dict JSON *and* the newer packetized list JSON are both supported), so
every session physically present in Test_dataset shows up immediately, even
before Phase 0 has ever been run for it. Each ID is then cross-referenced
against RESULTS_ROOT (matched by the record identity fields inside
record_info.json / phase1_meta.json / phase5_meta.json, not by folder-name
guessing) to attach a status badge:
    (dot) complete    - every phase (0,1,3,4B,5) has written its output
    (half) partial    - some, but not all, phases have run
    (open) unprocessed - the ID exists in Test_dataset but nothing has run yet
Selecting an unprocessed ID still opens cleanly - every panel shows an
explicit "run PhaseN first" state instead of an error.

NAVIGATION
----------
1. Overview         - KPI cards, pipeline trace (Phase 0 -> 5), quick status.
2. Signal Explorer   - full-night hypnogram + HR/RR trend + SQI, all sharing
   one synchronized timeline. Click, zoom, pan or scrub any track (or type
   a time / epoch directly) and every other panel - ECG, R-peaks, HR/RR,
   HRV, extracted features, signal quality, phase-wise results - jumps to
   that exact 30 s epoch instantly.
3. Phase Analysis    - every PNG the pipeline already generated, grouped by
   phase in collapsible panels, plus the raw metrics/logs backing each one.
4. Feature Table     - full Phase 3 HRV feature matrix, sortable / filterable
   / searchable; click any row to jump the Explorer to that epoch.
5. Raw JSON Inspector - drag-and-drop a device JSON export (legacy or new
   packetized format) to sanity-check it independent of the pipeline.

RUN
---
    pip install dash plotly pandas numpy scipy
    cd <project root>            # folder containing Test_dataset/, results_test/, ...
    python dashboard_app.py
Open http://127.0.0.1:8050

CONFIG
------
    export DASHBOARD_TEST_DATASET_ROOT=Test_dataset      # raw device JSON root
    export DASHBOARD_RESULTS_ROOT=results_test           # pipeline output root
    export DASHBOARD_DEVICE_ID_KEY=admissionId           # grouping key (rarely changed)
(mac/linux `export`; Windows `set`). Both roots are resolved relative to the
directory you launch `python dashboard_app.py` from, same as the phaseN_test.py
drivers - run it from your project root.
=============================================================================
"""

import os
import glob
import json
import base64
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.signal import find_peaks

import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx, Patch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────
def _resolve_project_path(default_name, env_name):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Prefer the actual project-root layout over any stale environment variable.
    preferred_candidates = []
    for base in [project_root, os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
        preferred_candidates.append(os.path.normpath(os.path.join(base, default_name)))

    if default_name == "results_test":
        preferred_candidates.extend([
            os.path.normpath(os.path.join(project_root, "results_test")),
            os.path.normpath(os.path.join(project_root, "Testing_code", "results_test")),
        ])
    elif default_name == "Test_Dataset":
        preferred_candidates.extend([
            os.path.normpath(os.path.join(project_root, "Test_Dataset")),
            os.path.normpath(os.path.join(project_root, "Testing_code", "Test_Dataset")),
        ])

    seen = set()
    for cand in preferred_candidates:
        norm = os.path.normpath(cand)
        if norm not in seen:
            seen.add(norm)
            if os.path.exists(norm):
                return norm

    env_value = os.environ.get(env_name)
    if env_value:
        candidate = os.path.abspath(os.path.expanduser(env_value))
        if os.path.exists(candidate):
            return candidate
        return candidate

    return os.path.normpath(os.path.join(project_root, default_name))

RESULTS_ROOT = _resolve_project_path("results_test", "DASHBOARD_RESULTS_ROOT")
TEST_DATASET_ROOT = _resolve_project_path("Test_Dataset", "DASHBOARD_TEST_DATASET_ROOT")
DEVICE_ID_KEY = os.environ.get("DASHBOARD_DEVICE_ID_KEY", "admissionId")

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_ORDER = [0, 1, 2, 3, 4]

# Dark-theme sleep-depth palette — chosen for contrast against a near-black
# background. Wake is warm/alert, N1->N3 deepen along a cool blue ramp,
# REM is a distinct magenta since it is physiologically an active state
# rather than "deeper" sleep.
STAGE_COLORS = {
    0: "#FB923C",   # Wake  - amber
    1: "#60A5FA",   # N1    - light blue
    2: "#22D3C8",   # N2    - teal
    3: "#6366F1",   # N3    - indigo
    4: "#F472B6",   # REM   - magenta
}
STAGE_COLOR_FALLBACK = "#64748B"

# Core dark clinical palette (shared by CSS + Plotly figures)
COLOR = {
    "bg_app":       "#0A0E16",
    "bg_sidebar":   "#0D1220",
    "bg_panel":     "#121826",
    "bg_panel_alt": "#161E30",
    "bg_surface":   "#1A2338",
    "bg_hover":     "#202B45",
    "border":       "#26314A",
    "border_soft":  "#1D2740",
    "text_primary": "#E7ECF5",
    "text_secondary": "#9AAAC4",
    "text_muted":   "#5E6E8C",
    "accent":       "#22D3C8",
    "accent_dim":   "#0E8388",
    "accent_blue":  "#4F9DF7",
    "accent_purple": "#A78BFA",
    "good":     "#34D399",
    "good_bg":  "rgba(52,211,153,0.12)",
    "warn":     "#FBBF24",
    "warn_bg":  "rgba(251,191,36,0.12)",
    "bad":      "#F87171",
    "bad_bg":   "rgba(248,113,113,0.12)",
    "flag":     "#C084FC",
    "flag_bg":  "rgba(192,132,252,0.12)",
    "neutral_bg": "rgba(148,163,184,0.10)",
}

PLOT_FONT = dict(family="Inter, -apple-system, Segoe UI, sans-serif",
                 color=COLOR["text_secondary"], size=12)
MONO_FONT = "'JetBrains Mono', 'Courier New', monospace"


# ──────────────────────────────────────────────────────────────────────────
# Safe loaders — tolerant of missing / partially-written files
# ──────────────────────────────────────────────────────────────────────────
def _safe_json(path):
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _safe_csv(path):
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def _safe_npy(path):
    if path and os.path.exists(path):
        try:
            return np.load(path, allow_pickle=True)
        except Exception:
            return None
    return None


def _first_glob(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def _safe_log_tail(path, n=200):
    if path and os.path.exists(path):
        try:
            with open(path, "r", errors="ignore") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception:
            return None
    return None


# ──────────────────────────────────────────────────────────────────────────
# Test_dataset scanning — the authoritative source of "which Admission IDs
# exist". This is a READ-ONLY mirror of the admissionId-grouping logic in
# phase0_dataset_management_testing/p0_scan_datasets.py (kept local so this
# dashboard has zero import dependency on the processing packages and can
# be pointed at any TEST_DATASET_ROOT). It supports both the legacy
# single-chunk dict JSON format and the newer packetized list format.
# ──────────────────────────────────────────────────────────────────────────
def _parse_iso8601_ms(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            pass
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return int(datetime.fromisoformat(s).timestamp() * 1000)
        except ValueError:
            return None
    if isinstance(value, dict):
        return _parse_iso8601_ms(value.get("$date"))
    return None


def _coerce_value_array(value):
    if value is None:
        return []
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], list):
            return list(value[0])
        if isinstance(value[0], (int, float)):
            return list(value)
    return []


def _is_new_packetized_json(payload):
    if not isinstance(payload, list) or not payload:
        return False
    first = payload[0]
    if not isinstance(first, dict) or "value" not in first:
        return False
    return isinstance(first.get("value"), list) and len(first.get("value")) > 0


def _read_device_chunk_meta_light(path, id_key=DEVICE_ID_KEY):
    """Scalar timing/id fields only from one device JSON file — never loads
    the ECG sample arrays themselves. Mirrors
    phase0_dataset_management_testing/p0_scan_datasets._read_device_chunk_meta."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None

    if isinstance(d, dict):
        sid = d.get(id_key) or d.get("admissionId")
        if not sid:
            return None
        return {
            "path": path, "session_id": str(sid),
            "window_start_ms": d.get("window_start_ms"),
            "window_end_ms": d.get("window_end_ms"),
            "duration_s": d.get("duration_s"),
            "sample_count": d.get("sample_count"),
        }

    if _is_new_packetized_json(d):
        timestamps, samples = [], 0
        for record in d:
            if not isinstance(record, dict):
                continue
            samples += len(_coerce_value_array(record.get("value")))
            ts = _parse_iso8601_ms(record.get("utcTimestamp"))
            if ts is not None:
                timestamps.append(ts)
        if not timestamps:
            return None
        sid = d[0].get(id_key) or d[0].get("admissionId")
        if not sid:
            return None
        start_ms, end_ms = min(timestamps), max(timestamps)
        return {
            "path": path, "session_id": str(sid),
            "window_start_ms": start_ms, "window_end_ms": end_ms,
            "duration_s": max((end_ms - start_ms) / 1000.0, 1 / 125),
            "sample_count": int(samples),
        }
    return None


_TEST_DATASET_CACHE = {"manifest": None}


def scan_test_dataset(root=TEST_DATASET_ROOT):
    """Walk TEST_DATASET_ROOT for every *.json chunk and group by Admission
    ID, sorted per-ID by window_start_ms. Returns {} (not an error) if the
    folder doesn't exist yet, so a misconfigured path degrades gracefully
    instead of crashing the whole dashboard."""
    manifest = {}
    if not root or not os.path.isdir(root):
        return manifest
    for fp in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        meta = _read_device_chunk_meta_light(fp)
        if meta is None or meta.get("window_start_ms") is None:
            continue
        manifest.setdefault(meta["session_id"], []).append(meta)
    for sid in manifest:
        manifest[sid] = sorted(manifest[sid], key=lambda m: m["window_start_ms"] or 0)
    return manifest


def get_test_dataset_manifest(force=False):
    if force or _TEST_DATASET_CACHE["manifest"] is None:
        _TEST_DATASET_CACHE["manifest"] = scan_test_dataset()
    return _TEST_DATASET_CACHE["manifest"]


# ──────────────────────────────────────────────────────────────────────────
# Results-side index — {admission_id: out_dir}. Matched primarily by the
# record identity written INSIDE each phase's own JSON (record_info.json /
# phase1_meta.json / phase5_meta.json), which is robust to whatever
# folder-naming convention make_subject_dirs() actually uses; falls back to
# the containing folder name only if no identity field is present.
# ──────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# Results-side index — {admission_id: out_dir}. Matched primarily by the
# record identity written INSIDE each phase's own JSON (record_info.json /
# phase1_meta.json / phase5_meta.json), which is robust to whatever
# folder-naming convention make_subject_dirs() actually uses; falls back to
# the containing folder name only if no identity field is present.
# ──────────────────────────────────────────────────────────────────────────
_RESULT_IDENTITY_KEYS = (
    "admissionId", "admission_id", "record", "record_name",
    "session_id", "record_id", "subject_id", "patient_id", "id"
)


def _identity_from_json(path):
    d = _safe_json(path)
    if not isinstance(d, dict):
        return None

    for k in _RESULT_IDENTITY_KEYS:
        value = d.get(k)
        if value is not None and str(value).strip():
            return str(value)

    # Some result JSON files nest data under "data" / "meta" / "record"
    for outer_key in ("data", "meta", "record", "patient"):
        nested = d.get(outer_key)
        if isinstance(nested, dict):
            for k in _RESULT_IDENTITY_KEYS:
                value = nested.get(k)
                if value is not None and str(value).strip():
                    return str(value)

    return None


def build_results_index(results_root=RESULTS_ROOT):
    index, seen_dirs = {}, set()

    if not results_root or not os.path.isdir(results_root):
        return index

    patterns = [
        "phase1_meta.json",
        "record_info.json",
        "phase3_hrv_features.csv",
        "phase5_meta.json",
        "phase4b_predict_meta.json",
        "phase4b_ecg_hypnogram_rf.npy",
    ]

    for pat in patterns:
        for hit in glob.glob(os.path.join(results_root, "**", pat), recursive=True):
            out_dir = os.path.dirname(hit)
            if out_dir in seen_dirs:
                continue
            seen_dirs.add(out_dir)

            sid = None
            for probe in (
                "record_info.json",
                "phase1_meta.json",
                "phase5_meta.json",
                "phase4b_predict_meta.json",
            ):
                probe_path = os.path.join(out_dir, probe)
                sid = _identity_from_json(probe_path)
                if sid:
                    break

            if not sid:
                sid = os.path.basename(os.path.normpath(out_dir))

            index[str(sid)] = out_dir

    # Fallback: if the directory itself is the admission ID
    for candidate_dir in sorted(glob.glob(os.path.join(results_root, "**"), recursive=True)):
        if os.path.isdir(candidate_dir):
            sid = os.path.basename(os.path.normpath(candidate_dir))
            if sid and sid.startswith("ADM") and sid not in index:
                index[sid] = candidate_dir

    return index


def _session_status(entry):
    out_dir = entry.get("out_dir")
    if not out_dir:
        return "unprocessed"
    flags = [
        os.path.exists(os.path.join(out_dir, "record_info.json")),
        os.path.exists(os.path.join(out_dir, "phase1_meta.json")),
        os.path.exists(os.path.join(out_dir, "phase3_hrv_features.csv")),
        os.path.exists(os.path.join(out_dir, "phase4b_predict_meta.json")),
        os.path.exists(os.path.join(out_dir, "phase5_meta.json")),
    ]
    n = sum(flags)
    if n == len(flags):
        return "complete"
    if n == 0:
        return "unprocessed"
    return "partial"


STATUS_ICON = {"complete": "\u2713", "partial": "\u25d0", "unprocessed": "\u25cb"}
STATUS_LABEL = {"complete": "Complete", "partial": "Partial", "unprocessed": "Not processed"}


def discover_sessions(results_root=RESULTS_ROOT, dataset_root=TEST_DATASET_ROOT):
    """
    The Admission ID registry that drives the entire dashboard.
    Primary source: every Admission ID physically present under
    TEST_DATASET_ROOT — this is what populates the selector, regardless of
    how far the pipeline has progressed for that ID. Cross-referenced
    against RESULTS_ROOT to attach the processed output dir + status badge.

    Returns {admission_id: {"out_dir": str|None, "raw_chunks": [...],
                            "status": "complete"|"partial"|"unprocessed"}}
    """
    raw_manifest = get_test_dataset_manifest() if dataset_root == TEST_DATASET_ROOT         else scan_test_dataset(dataset_root)
    results_index = build_results_index(results_root)

    registry = {sid: {"raw_chunks": chunks, "out_dir": results_index.get(sid)}
               for sid, chunks in raw_manifest.items()}

    # IDs that only exist in RESULTS_ROOT (e.g. raw chunks archived/pruned
    # after processing) still surface, flagged accordingly, so nothing that
    # was ever processed silently disappears from the selector.
    for sid, out_dir in results_index.items():
        registry.setdefault(sid, {"raw_chunks": [], "out_dir": out_dir})
        registry[sid]["out_dir"] = out_dir

    for entry in registry.values():
        entry["status"] = _session_status(entry)

    return dict(sorted(registry.items()))


def session_dropdown_options(results_root=RESULTS_ROOT, dataset_root=TEST_DATASET_ROOT):
    registry = discover_sessions(results_root, dataset_root)
    options = []
    for sid, entry in registry.items():
        icon = STATUS_ICON.get(entry["status"], "\u25cb")
        n_chunks = len(entry.get("raw_chunks") or [])
        suffix = f"  ·  {n_chunks} chunk(s), unprocessed" if not entry.get("out_dir") else ""
        options.append({"label": f"{icon}  {sid}{suffix}", "value": sid})
    return options


def dataset_scan_summary(results_root=RESULTS_ROOT, dataset_root=TEST_DATASET_ROOT):
    registry = discover_sessions(results_root, dataset_root)
    n_total = len(registry)
    n_complete = sum(1 for e in registry.values() if e["status"] == "complete")
    n_partial = sum(1 for e in registry.values() if e["status"] == "partial")
    if n_total == 0:
        return (f"No Admission IDs found under '{dataset_root}' — check "
               f"DASHBOARD_TEST_DATASET_ROOT / that you launched the dashboard "
               f"from your project root.")
    return (f"{n_total} Admission ID(s) in {dataset_root}  ·  "
           f"{n_complete} complete  ·  {n_partial} partial  ·  "
           f"{n_total - n_complete - n_partial} not processed")


# ──────────────────────────────────────────────────────────────────────────
# Per-session loader, cached in memory for the life of the server process
# ──────────────────────────────────────────────────────────────────────────
_SESSION_CACHE = {}
_EMPTY_PHASE_FIELDS = {
    "record_info": {}, "phase1_meta": {}, "sqi_df": None, "epochs": None,
    "bad_mask": None, "rr_raw": None, "phase2_consolidated_row": None,
    "hrv_df": None, "pred_smooth": None, "pred_raw": None,
    "predict_meta": {}, "phase5_meta": {}, "duration_df": None,
    "phase_pngs": [], "log_tail": None, "log_path": None,
    "epoch_sec": 30, "target_fs": 125, "recording_start": None,
}


def load_session(session_id, results_root=RESULTS_ROOT):
    if session_id in _SESSION_CACHE:
        return _SESSION_CACHE[session_id]

    registry = discover_sessions(results_root)
    entry = registry.get(session_id)
    if entry is None:
        return None

    out_dir = entry.get("out_dir")
    data = {"output_dir": out_dir, "session_id": session_id,
           "status": entry["status"], "raw_chunks": entry.get("raw_chunks") or []}

    if out_dir is None:
        # This Admission ID exists in Test_dataset but no phase has produced
        # output for it yet — every downstream tab already renders an
        # explicit "run PhaseN first" state for empty/None fields.
        data.update(_EMPTY_PHASE_FIELDS)
        _SESSION_CACHE[session_id] = data
        return data

    # Phase 0
    data["record_info"] = _safe_json(os.path.join(out_dir, "record_info.json")) or {}

    # Phase 1
    data["phase1_meta"] = _safe_json(os.path.join(out_dir, "phase1_meta.json")) or {}
    data["sqi_df"]      = _safe_csv(os.path.join(out_dir, "phase1_sqi.csv"))
    data["epochs"]      = _safe_npy(os.path.join(out_dir, "preprocessed_epochs.npy"))
    data["bad_mask"]    = _safe_npy(os.path.join(out_dir, "bad_epoch_mask.npy"))

    # Phase 2
    data["rr_raw"] = _safe_json(os.path.join(out_dir, "phase2_rr_full.json"))
    consolidated_path = _first_glob([os.path.join(results_root, "**", "*consolidated*.csv")])
    data["phase2_consolidated_row"] = None
    if consolidated_path:
        cdf = _safe_csv(consolidated_path)
        if cdf is not None:
            id_col = next((c for c in cdf.columns if c.lower() in
                          ("record_name", "session_id", "record", "admissionid")), None)
            if id_col is not None:
                match = cdf[cdf[id_col].astype(str) == str(session_id)]
                if len(match):
                    data["phase2_consolidated_row"] = match.iloc[0].to_dict()

    # Phase 3
    data["hrv_df"] = _safe_csv(os.path.join(out_dir, "phase3_hrv_features.csv"))

    # Phase 4b
    data["pred_smooth"]  = _safe_npy(os.path.join(out_dir, "phase4b_ecg_hypnogram_rf.npy"))
    data["pred_raw"]     = _safe_npy(os.path.join(out_dir, "phase4b_ecg_hypnogram_rf_raw.npy"))
    data["predict_meta"] = _safe_json(os.path.join(out_dir, "phase4b_predict_meta.json")) or {}

    # Phase 5
    data["phase5_meta"] = _safe_json(os.path.join(out_dir, "phase5_meta.json")) or {}
    dur_path = _first_glob([os.path.join(out_dir, "*duration*summary*.csv"),
                            os.path.join(out_dir, "*duration*.csv")])
    data["duration_df"] = _safe_csv(dur_path)
    data["phase_pngs"] = sorted(glob.glob(os.path.join(out_dir, "*.png")))

    # Logs (best-effort — any *.log under this session's output dir or the
    # shared logs/ folder mentioning the session id)
    log_hits = sorted(glob.glob(os.path.join(out_dir, "*.log")))
    if not log_hits:
        log_hits = sorted(glob.glob(os.path.join("logs", f"*{session_id}*.log")))
    data["log_tail"] = _safe_log_tail(log_hits[0]) if log_hits else None
    data["log_path"] = log_hits[0] if log_hits else None

    meta = data["phase1_meta"]
    data["epoch_sec"] = meta.get("epoch_sec", 30)
    data["target_fs"] = meta.get("target_fs", 125)
    data["recording_start"] = (meta.get("recording_start_time")
                               or data["record_info"].get("recording_start"))

    _SESSION_CACHE[session_id] = data
    return data


def clear_cache(session_id=None):
    if session_id:
        _SESSION_CACHE.pop(session_id, None)
    else:
        _SESSION_CACHE.clear()
        get_test_dataset_manifest(force=True)


# ──────────────────────────────────────────────────────────────────────────
# On-the-fly R-peak / HR / RR / HRV cross-check for the selected epoch
# (visualization only — does NOT touch Phase 2/3's own official values)
# ──────────────────────────────────────────────────────────────────────────
def detect_rpeaks_and_hrv(seg, fs):
    seg = np.asarray(seg, dtype=float)
    if len(seg) < fs:
        return {"peaks": np.array([]), "mean_hr": None, "sdnn": None, "rmssd": None}

    sig = seg - np.nanmedian(seg)
    min_distance = max(int(0.3 * fs), 1)          # refractory ~300 ms -> max ~200 bpm
    height = np.nanstd(sig) * 1.0
    peaks, _ = find_peaks(sig, distance=min_distance, height=height)

    if len(peaks) < 2:
        return {"peaks": peaks, "mean_hr": None, "sdnn": None, "rmssd": None}

    rr_ms = np.diff(peaks) / fs * 1000.0
    mean_hr = 60000.0 / np.mean(rr_ms) if len(rr_ms) else None
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else None
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 2 else None

    return {"peaks": peaks, "rr_ms": rr_ms, "mean_hr": mean_hr, "sdnn": sdnn, "rmssd": rmssd}


def epoch_time_label(epoch_idx, epoch_sec, recording_start=None):
    """Clock time (+ date, if known) if recording_start is parseable, else
    relative h/m/s elapsed. Returns (clock_str, date_str_or_None, is_absolute)."""
    total_sec = epoch_idx * epoch_sec
    if recording_start:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%H:%M:%S"):
            try:
                base = datetime.strptime(recording_start, fmt)
                clock = base + timedelta(seconds=total_sec)
                date_str = clock.strftime("%d %b %Y") if fmt == "%Y-%m-%d %H:%M:%S" else None
                return clock.strftime("%H:%M:%S"), date_str, True
            except ValueError:
                continue
    hh, rem = divmod(int(total_sec), 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}", None, False


# ──────────────────────────────────────────────────────────────────────────
# Session-level summary metrics (drives the Overview KPI cards)
# ──────────────────────────────────────────────────────────────────────────
def compute_summary_metrics(data):
    epoch_sec = data.get("epoch_sec", 30)
    pred = data.get("pred_smooth")
    pred_raw = data.get("pred_raw")
    n = len(pred) if pred is not None else 0

    total_sec = n * epoch_sec
    hh, rem = divmod(int(total_sec), 3600)
    mm = rem // 60
    duration_label = f"{hh}h {mm:02d}m" if n else "—"

    sqi_df = data.get("sqi_df")
    mean_sqi = None
    if sqi_df is not None and "overall_sqi" in sqi_df.columns and len(sqi_df):
        mean_sqi = float(pd.to_numeric(sqi_df["overall_sqi"], errors="coerce").mean())
    if mean_sqi is None or np.isnan(mean_sqi):
        sqi_status, sqi_label = "neutral", "n/a"
    else:
        sqi_label = f"{mean_sqi:.2f}"
        sqi_status = "good" if mean_sqi >= 0.7 else "warn" if mean_sqi >= 0.5 else "bad"

    hrv_df = data.get("hrv_df")
    mean_hr = mean_rr = None
    if hrv_df is not None and len(hrv_df):
        if "mean_hr" in hrv_df.columns:
            mean_hr = float(pd.to_numeric(hrv_df["mean_hr"], errors="coerce").mean())
        if "resp_rate_est" in hrv_df.columns:
            mean_rr = float(pd.to_numeric(hrv_df["resp_rate_est"], errors="coerce").mean())
    hr_label = f"{mean_hr:.0f} bpm" if mean_hr is not None and not np.isnan(mean_hr) else "n/a"
    rr_label = f"{mean_rr:.1f} /min" if mean_rr is not None and not np.isnan(mean_rr) else "n/a"

    stage_counts, dominant_label = {}, "n/a"
    if n:
        vals, counts = np.unique(pred.astype(int), return_counts=True)
        stage_counts = {int(v): int(c) for v, c in zip(vals, counts)}
        dominant_label = STAGE_NAMES.get(int(vals[np.argmax(counts)]), "?")

    bad_mask = data.get("bad_mask")
    n_bad = int(np.sum(bad_mask)) if bad_mask is not None else 0
    bad_pct = (n_bad / n * 100.0) if n else 0.0
    artifact_status = "good" if bad_pct < 5 else "warn" if bad_pct < 15 else "bad"

    n_mismatch = 0
    if pred is not None and pred_raw is not None and len(pred) == len(pred_raw):
        n_mismatch = int(np.sum(pred.astype(int) != pred_raw.astype(int)))

    # Phase 0 note: run_phase0_test.py's driver (scan -> register -> inspect)
    # does NOT call p0_metadata_extraction.extract_all_metadata, so
    # record_info.json is legitimately absent for many fully-processed
    # sessions — it's a separate, optional metadata-extraction step, not a
    # sign Phase 0 didn't run. phase0_raw_ecg_overview.png (written by
    # p0_inspect_plots.run_phase0_record) is what Phase 0 actually,
    # reliably produces, so either artifact counts as "Phase 0 done".
    phase0_png = any(os.path.basename(p).lower().startswith("phase0")
                     for p in data.get("phase_pngs", []))
    phase_flags = [
        bool(data.get("record_info")) or phase0_png,
        bool(data.get("phase1_meta")) or data.get("epochs") is not None,
        bool(data.get("phase2_consolidated_row")),
        data.get("hrv_df") is not None,
        bool(data.get("predict_meta")) or data.get("pred_smooth") is not None,
        bool(data.get("phase5_meta")) or data.get("duration_df") is not None,
    ]
    n_complete = sum(phase_flags)
    if n_complete == 6:
        proc_status, proc_label = "good", "Complete · 6 / 6"
    elif n_complete == 0:
        proc_status, proc_label = "bad", "Not started"
    else:
        proc_status, proc_label = "warn", f"Partial · {n_complete} / 6"

    return dict(
        n_epochs=n, duration_label=duration_label, epoch_sec=epoch_sec,
        mean_sqi=mean_sqi, sqi_label=sqi_label, sqi_status=sqi_status,
        hr_label=hr_label, rr_label=rr_label,
        stage_counts=stage_counts, dominant_label=dominant_label,
        n_bad=n_bad, bad_pct=bad_pct, artifact_status=artifact_status,
        n_mismatch=n_mismatch,
        n_complete=n_complete, proc_status=proc_status, proc_label=proc_label,
        phase_flags=phase_flags,
    )


# ──────────────────────────────────────────────────────────────────────────
# Figure builders — dark Plotly theme throughout
# ──────────────────────────────────────────────────────────────────────────
def _dark_layout(fig, height, extra=None):
    layout = dict(
        height=height, font=PLOT_FONT,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLOR["bg_panel"],
        margin=dict(l=48, r=24, t=44, b=40),
        hoverlabel=dict(bgcolor=COLOR["bg_surface"], font_color=COLOR["text_primary"],
                        font_family=MONO_FONT, bordercolor=COLOR["border"]),
    )
    if extra:
        layout.update(extra)
    fig.update_layout(**layout)


def build_overview_figure(data, epoch_idx=None):
    """
    3-row synchronized timeline: hypnogram / HR & RR trend / SQI trend.
    Sharing one x-axis (hours) with a single range-slider. A vertical guide
    line marks the epoch currently open in the inspector below, so every
    track visually agrees on "where we are" in the night.
    """
    pred = data["pred_smooth"]
    pred_raw = data["pred_raw"]
    epoch_sec = data["epoch_sec"]
    n = len(pred) if pred is not None else 0
    if n == 0:
        fig = go.Figure()
        fig.add_annotation(text="No phase4b_ecg_hypnogram_rf.npy found for this session yet — "
                                "run Phase 4b first.", showarrow=False,
                          font=dict(color=COLOR["text_muted"], size=13))
        _dark_layout(fig, 220)
        return fig

    t_hr = np.arange(n) * epoch_sec / 3600.0
    colors = [STAGE_COLORS.get(int(s), STAGE_COLOR_FALLBACK) for s in pred]
    stage_labels = [STAGE_NAMES.get(int(s), "?") for s in pred]

    hrv_df = data["hrv_df"]
    has_hr = hrv_df is not None and "mean_hr" in hrv_df.columns
    sqi_df = data["sqi_df"]
    has_sqi = sqi_df is not None and "overall_sqi" in sqi_df.columns

    rows, row_heights, titles = [1], [1.0], ["Predicted hypnogram — click any segment to inspect that epoch"]
    if has_hr:
        rows.append(2); row_heights.append(0.55); titles.append("Heart rate / respiration trend — Phase 3")
    if has_sqi:
        rows.append(3 if has_hr else 2); row_heights.append(0.45)
        titles.append("Signal quality index (SQI) — Phase 1")
    n_rows = len(rows)
    row_heights = [h / sum(row_heights) for h in row_heights]

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.10, row_heights=row_heights,
                        subplot_titles=titles)

    fig.add_trace(go.Bar(
        x=t_hr, y=[1] * n, width=(epoch_sec / 3600.0) * 0.98,
        marker_color=colors, marker_line_width=0,
        customdata=np.arange(n), text=stage_labels,
        hovertemplate="Epoch %{customdata}<br>t=%{x:.3f} h<br>Stage: %{text}<extra></extra>",
        name="Predicted stage",
    ), row=1, col=1)
    fig.update_yaxes(visible=False, range=[0, 1.35], row=1, col=1)

    bad_mask = data["bad_mask"]
    if bad_mask is not None:
        bad_idx = np.where(np.asarray(bad_mask[:n]).astype(bool))[0]
        if len(bad_idx):
            fig.add_trace(go.Scatter(
                x=t_hr[bad_idx], y=[1.16] * len(bad_idx), mode="markers",
                marker=dict(symbol="triangle-down", color=COLOR["bad"], size=8, line=dict(width=0)),
                name="Artifact / bad epoch",
                hovertemplate="Artifact epoch %{customdata}<extra></extra>",
                customdata=bad_idx,
            ), row=1, col=1)

    if pred_raw is not None and len(pred_raw) == n:
        mismatch_idx = np.where(pred.astype(int) != pred_raw.astype(int))[0]
        if len(mismatch_idx):
            fig.add_trace(go.Scatter(
                x=t_hr[mismatch_idx], y=[1.27] * len(mismatch_idx), mode="markers",
                marker=dict(symbol="diamond", color=COLOR["flag"], size=6, line=dict(width=0)),
                name="Raw ≠ smoothed (review)",
                hovertemplate="Epoch %{customdata}: raw/smoothed disagree<extra></extra>",
                customdata=mismatch_idx,
            ), row=1, col=1)

    row_cursor = 1
    if has_hr:
        row_cursor += 1
        n_hr = min(n, len(hrv_df))
        hr_vals = pd.to_numeric(hrv_df["mean_hr"], errors="coerce").values[:n_hr]
        fig.add_trace(go.Scatter(
            x=t_hr[:n_hr], y=hr_vals, mode="lines", line=dict(color=COLOR["accent_blue"], width=1.3),
            name="HR (bpm)", customdata=np.arange(n_hr),
            hovertemplate="Epoch %{customdata}<br>HR=%{y:.1f} bpm<extra></extra>",
        ), row=row_cursor, col=1)
        if "resp_rate_est" in hrv_df.columns:
            rr_vals = pd.to_numeric(hrv_df["resp_rate_est"], errors="coerce").values[:n_hr]
            fig.add_trace(go.Scatter(
                x=t_hr[:n_hr], y=rr_vals, mode="lines", line=dict(color=COLOR["accent_purple"], width=1.1, dash="dot"),
                name="Resp. rate (br/min)", yaxis=f"y{row_cursor}", customdata=np.arange(n_hr),
                hovertemplate="Epoch %{customdata}<br>RR=%{y:.1f} /min<extra></extra>",
            ), row=row_cursor, col=1)
        fig.update_yaxes(gridcolor=COLOR["border_soft"], zerolinecolor=COLOR["border"], row=row_cursor, col=1)

    if has_sqi:
        row_cursor += 1
        sqi_vals = pd.to_numeric(sqi_df["overall_sqi"], errors="coerce").values[:n]
        t_sqi = t_hr[:len(sqi_vals)]
        fig.add_trace(go.Scatter(
            x=t_sqi, y=sqi_vals, mode="lines", line=dict(color=COLOR["accent"], width=1.3),
            name="SQI", fill="tozeroy", fillcolor="rgba(34,211,200,0.08)",
            hovertemplate="t=%{x:.3f} h<br>SQI=%{y:.2f}<extra></extra>",
        ), row=row_cursor, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color=COLOR["bad"], line_width=1, row=row_cursor, col=1)
        fig.update_yaxes(range=[0, 1.05], gridcolor=COLOR["border_soft"], row=row_cursor, col=1)

    if epoch_idx is not None and n:
        t_sel = min(epoch_idx, n - 1) * epoch_sec / 3600.0
        fig.add_vline(x=t_sel, line_color=COLOR["text_primary"], line_width=1.4,
                     line_dash="dot", opacity=0.85)

    fig.update_xaxes(title_text="Time (hours)", gridcolor=COLOR["border_soft"],
                     linecolor=COLOR["border"], row=n_rows, col=1,
                     rangeslider=dict(visible=True, thickness=0.06, bgcolor=COLOR["bg_panel_alt"],
                                      bordercolor=COLOR["border"], borderwidth=1))
    for r in range(1, n_rows):
        fig.update_xaxes(gridcolor=COLOR["border_soft"], linecolor=COLOR["border"], row=r, col=1)

    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(family=PLOT_FONT["family"], size=12.5, color=COLOR["text_secondary"])
        ann["x"] = 0
        ann["xanchor"] = "left"

    _dark_layout(fig, 210 * n_rows + 60, extra=dict(
        showlegend=False, bargap=0, clickmode="event+select",
        margin=dict(l=48, r=24, t=40, b=40),
    ))
    return fig


def build_ecg_figure(data, epoch_idx):
    epochs = data["epochs"]
    fs = data["target_fs"]
    if epochs is None or epoch_idx is None or epoch_idx >= len(epochs):
        fig = go.Figure()
        fig.add_annotation(text="No preprocessed_epochs.npy for this session "
                                "(run Phase 1 first).", showarrow=False,
                          font=dict(color=COLOR["text_muted"], size=13))
        _dark_layout(fig, 300)
        return fig, {}

    seg = np.asarray(epochs[epoch_idx]).flatten()
    t = np.arange(len(seg)) / fs
    hrv_check = detect_rpeaks_and_hrv(seg, fs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=seg, mode="lines",
                             line=dict(color=COLOR["accent"], width=1.3),
                             name="ECG (DWT-cleaned)",
                             hovertemplate="t=%{x:.3f}s<br>amp=%{y:.4f}<extra></extra>"))
    peaks = hrv_check.get("peaks", np.array([]))
    if len(peaks):
        fig.add_trace(go.Scatter(x=t[peaks], y=seg[peaks], mode="markers",
                                 marker=dict(color=COLOR["bad"], size=8, symbol="x-thin",
                                            line=dict(width=2, color=COLOR["bad"])),
                                 name="R-peaks (recomputed, display-only)",
                                 hovertemplate="R-peak @ t=%{x:.3f}s<extra></extra>"))
    _dark_layout(fig, 320, extra=dict(
        title=dict(text=f"Epoch {epoch_idx} · ECG segment "
                        f"({data['epoch_sec']} s @ {fs:.1f} Hz)",
                  font=dict(size=13, color=COLOR["text_primary"])),
        xaxis=dict(title="Time within epoch (s)", gridcolor=COLOR["border_soft"], linecolor=COLOR["border"]),
        yaxis=dict(title="Amplitude (mV)", gridcolor=COLOR["border_soft"], linecolor=COLOR["border"]),
        legend=dict(orientation="h", y=1.22, font=dict(size=11, color=COLOR["text_secondary"]),
                   bgcolor="rgba(0,0,0,0)"),
    ))
    return fig, hrv_check


def build_feature_trend_figure(data, feature_col, epoch_idx=None):
    hrv_df = data["hrv_df"]
    if hrv_df is None or feature_col not in hrv_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Select a feature to plot its full-night trend.",
                          showarrow=False, font=dict(color=COLOR["text_muted"], size=12))
        _dark_layout(fig, 200)
        return fig

    epoch_sec = data["epoch_sec"]
    vals = pd.to_numeric(hrv_df[feature_col], errors="coerce").values
    t_hr = np.arange(len(vals)) * epoch_sec / 3600.0

    pred = data.get("pred_smooth")
    colors = None
    if pred is not None and len(pred) >= len(vals):
        colors = [STAGE_COLORS.get(int(s), STAGE_COLOR_FALLBACK) for s in pred[:len(vals)]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_hr, y=vals, mode="lines+markers",
        line=dict(color=COLOR["accent_blue"], width=1.0),
        marker=dict(size=4, color=colors if colors else COLOR["accent_blue"]),
        customdata=np.arange(len(vals)),
        hovertemplate=f"Epoch %{{customdata}}<br>{feature_col}=%{{y:.4f}}<extra></extra>",
    ))
    if epoch_idx is not None and epoch_idx < len(vals):
        fig.add_vline(x=epoch_idx * epoch_sec / 3600.0, line_color=COLOR["text_primary"],
                     line_width=1.2, line_dash="dot", opacity=0.8)
    _dark_layout(fig, 210, extra=dict(
        title=dict(text=f"{feature_col} — full-night trend (dot color = predicted stage)",
                  font=dict(size=12, color=COLOR["text_secondary"])),
        xaxis=dict(title="Time (hours)", gridcolor=COLOR["border_soft"], linecolor=COLOR["border"]),
        yaxis=dict(gridcolor=COLOR["border_soft"], linecolor=COLOR["border"]),
        margin=dict(l=48, r=24, t=36, b=36),
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Small reusable UI components
# ──────────────────────────────────────────────────────────────────────────
def metric_card(label, value, sub=None, status="neutral", icon=None):
    return html.Div(className="metric-card", children=[
        html.Div(className=f"metric-dot metric-dot--{status}"),
        html.Div(className="metric-card__body", children=[
            html.Div([html.Span(icon, className="metric-card__icon") if icon else None,
                     html.Span(label)], className="metric-card__label"),
            html.Div(value, className="metric-card__value"),
            html.Div(sub, className="metric-card__sub") if sub is not None else None,
        ]),
    ])


def stage_distribution_bar(stage_counts, n):
    if not stage_counts or not n:
        return html.Div("no prediction data", className="metric-card__sub")
    segments = []
    for stage_id in STAGE_ORDER:
        if stage_id not in stage_counts:
            continue
        pct = stage_counts[stage_id] / n * 100.0
        if pct <= 0:
            continue
        segments.append(html.Div(
            title=f"{STAGE_NAMES.get(stage_id, '?')}: {pct:.0f}%",
            style={"width": f"{pct}%", "background": STAGE_COLORS.get(stage_id, STAGE_COLOR_FALLBACK)},
            className="stage-bar__seg",
        ))
    legend = []
    for sid in STAGE_ORDER:
        if sid not in stage_counts:
            continue
        legend.append(html.Span([
            html.Span(className="legend-dot", style={"background": STAGE_COLORS.get(sid, STAGE_COLOR_FALLBACK)}),
            f" {STAGE_NAMES.get(sid, '?')}",
        ], className="legend-item"))
    return html.Div([
        html.Div(segments, className="stage-bar"),
        html.Div(legend, className="stage-bar__legend"),
    ])


def status_pill(text, status="neutral", title=None):
    return html.Span(text, className=f"pill pill--{status}", title=title)


def timeline_legend():
    """
    Custom HTML legend for the synchronized timeline — replaces Plotly's
    built-in legend, which used to sit at y=1.06 directly on top of the
    row-1 subplot title ("Predicted hypnogram — click any segment…"),
    producing the overlapping, hard-to-read strip from earlier screenshots.
    Rendered once, above the graph, so it never re-flows or collides again.
    """
    def swatch_row(items):
        return html.Div(items, className="legend-strip")

    def dot(label, color):
        return html.Span([html.Span(className="legend-swatch legend-swatch--dot",
                                   style={"background": color}), label], className="legend-chip")

    def marker(label, color, symbol):
        return html.Span([html.Span(symbol, className="legend-swatch legend-swatch--marker",
                                   style={"color": color}), label], className="legend-chip")

    def line(label, color, dashed=False):
        return html.Span([html.Span(className="legend-swatch legend-swatch--line"
                                   + (" legend-swatch--dashed" if dashed else ""),
                                   style={"background": color if not dashed else "transparent",
                                         "borderTopColor": color}),
                          label], className="legend-chip")

    stage_dots = [dot(STAGE_NAMES[s], STAGE_COLORS[s]) for s in STAGE_ORDER]
    marker_items = [
        marker("Artifact / bad epoch", COLOR["bad"], "▽"),
        marker("Raw ≠ smoothed (review)", COLOR["flag"], "◇"),
        line("HR (bpm)", COLOR["accent_blue"]),
        line("Resp. rate (br/min)", COLOR["accent_purple"], dashed=True),
        line("SQI", COLOR["accent"]),
    ]
    return html.Div([
        swatch_row(stage_dots),
        swatch_row(marker_items),
    ], className="timeline-legend")


def kv_grid(pairs):
    rows = []
    for k, v in pairs:
        rows.append(html.Div(str(k), className="kv-key"))
        rows.append(html.Div("—" if v is None or v == "" else str(v), className="kv-val"))
    return html.Div(rows, className="kv-grid")


def phase_step(index, title, status, body):
    return html.Div(className="phase-step", children=[
        html.Div(className="phase-step__rail", children=[
            html.Div(str(index), className=f"phase-step__marker phase-step__marker--{status}"),
            html.Div(className="phase-step__line"),
        ]),
        html.Div(className="phase-step__content", children=[
            html.Div([title, status_pill("done", "good") if status == "good" else status_pill("missing", "bad")],
                    className="phase-step__title"),
            body,
        ]),
    ])


def section_card(title, children, subtitle=None, right=None):
    header_children = [html.Div(title, className="section-card__title")]
    if subtitle:
        header_children.append(html.Div(subtitle, className="section-card__subtitle"))
    header = [html.Div(header_children, className="section-card__header-text")]
    if right is not None:
        header.append(html.Div(right, className="section-card__header-right"))
    return html.Div(className="section-card", children=[
        html.Div(header, className="section-card__header"),
        html.Div(children, className="section-card__body"),
    ])


# ──────────────────────────────────────────────────────────────────────────
# Epoch detail lookups — the "why was this predicted?" root-cause panel
# ──────────────────────────────────────────────────────────────────────────
def build_epoch_detail(data, epoch_idx, hrv_check):
    pred, pred_raw = data["pred_smooth"], data["pred_raw"]
    epoch_sec = data["epoch_sec"]
    recording_start = data.get("recording_start")

    smoothed = int(pred[epoch_idx]) if pred is not None and epoch_idx < len(pred) else None
    raw = int(pred_raw[epoch_idx]) if pred_raw is not None and epoch_idx < len(pred_raw) else None
    mismatch = smoothed is not None and raw is not None and smoothed != raw

    bad_mask = data["bad_mask"]
    is_bad = bool(bad_mask is not None and epoch_idx < len(bad_mask) and bad_mask[epoch_idx])

    sqi_df = data["sqi_df"]
    sqi_val = None
    if sqi_df is not None and epoch_idx < len(sqi_df) and "overall_sqi" in sqi_df.columns:
        sqi_val = float(sqi_df["overall_sqi"].iloc[epoch_idx])

    badges = []
    if smoothed is not None:
        badges.append(html.Span(f"● {STAGE_NAMES.get(smoothed, '?')}", className="pill pill--stage",
                                style={"borderColor": STAGE_COLORS.get(smoothed, STAGE_COLOR_FALLBACK),
                                      "color": STAGE_COLORS.get(smoothed, STAGE_COLOR_FALLBACK)}))
    if is_bad:
        badges.append(status_pill("Artifact / excluded upstream", "bad"))
    if mismatch:
        badges.append(status_pill(f"Raw prediction was {STAGE_NAMES.get(raw, '?')} — review", "flag"))
    if sqi_val is not None:
        sqi_status = "good" if sqi_val >= 0.7 else "warn" if sqi_val >= 0.5 else "bad"
        badges.append(status_pill(f"SQI {sqi_val:.2f}", sqi_status))
    if not badges:
        badges.append(status_pill("No prediction data for this epoch", "neutral"))

    clock, date_str, is_absolute = epoch_time_label(epoch_idx, epoch_sec, recording_start)
    time_line = html.Div(
        f"Epoch {epoch_idx}  ·  {'IST' if is_absolute else 't+'} {clock}"
        + (f"  ·  {date_str}" if date_str else "")
        + f"  ·  {epoch_idx * epoch_sec / 3600:.3f} h into session",
        className="epoch-detail__time",
        title=None if is_absolute else
             "No recording_start_time in phase1_meta.json — showing time elapsed since "
             "session start instead of a clock time.",
    )

    hrv_df = data["hrv_df"]
    hr_official_val = rr_official_val = None
    if hrv_df is not None and epoch_idx < len(hrv_df):
        row = hrv_df.iloc[epoch_idx]
        if "mean_hr" in hrv_df.columns and pd.notna(row["mean_hr"]):
            hr_official_val = float(row["mean_hr"])
        if "resp_rate_est" in hrv_df.columns and pd.notna(row["resp_rate_est"]):
            rr_official_val = float(row["resp_rate_est"])

    # "At a glance" — the single strip the person actually wants when they
    # click a stage on the hypnogram: exactly when this is, what stage it
    # was scored as, and its HR / RR / SQI, all in one place before the
    # deeper root-cause comparison below.
    glance = html.Div(className="epoch-glance", children=[
        html.Div(className="epoch-glance__item", children=[
            html.Div("EPOCH", className="epoch-glance__label"),
            html.Div(str(epoch_idx), className="epoch-glance__value"),
        ]),
        html.Div(className="epoch-glance__item", children=[
            html.Div("TIME (IST)" if is_absolute else "TIME (RELATIVE)", className="epoch-glance__label"),
            html.Div(clock, className="epoch-glance__value"),
        ]),
        html.Div(className="epoch-glance__item", children=[
            html.Div("STAGE", className="epoch-glance__label"),
            html.Div(STAGE_NAMES.get(smoothed, "—"), className="epoch-glance__value",
                    style={"color": STAGE_COLORS.get(smoothed, COLOR["text_primary"])} if smoothed is not None else {}),
        ]),
        html.Div(className="epoch-glance__item", children=[
            html.Div("HR", className="epoch-glance__label"),
            html.Div(f"{hr_official_val:.0f} bpm" if hr_official_val is not None else "n/a",
                    className="epoch-glance__value"),
        ]),
        html.Div(className="epoch-glance__item", children=[
            html.Div("RESP. RATE", className="epoch-glance__label"),
            html.Div(f"{rr_official_val:.1f} /min" if rr_official_val is not None else "n/a",
                    className="epoch-glance__value"),
        ]),
        html.Div(className="epoch-glance__item", children=[
            html.Div("SQI", className="epoch-glance__label"),
            html.Div(f"{sqi_val:.2f}" if sqi_val is not None else "n/a",
                    className="epoch-glance__value",
                    style={"color": COLOR["good"] if (sqi_val or 0) >= 0.7 else
                          COLOR["warn"] if (sqi_val or 0) >= 0.5 else COLOR["bad"]} if sqi_val is not None else {}),
        ]),
    ])

    official = {}
    if hr_official_val is not None:
        official["HR (bpm)"] = f"{hr_official_val:.1f}"
    if hrv_df is not None and epoch_idx < len(hrv_df):
        row = hrv_df.iloc[epoch_idx]
        for key, label in [("sdnn", "SDNN (ms)"), ("rmssd", "RMSSD (ms)")]:
            if key in hrv_df.columns:
                val = row[key]
                official[label] = f"{val:.1f}" if pd.notna(val) else "NaN"

    recomputed = {
        "HR (bpm)": f"{hrv_check['mean_hr']:.1f}" if hrv_check.get("mean_hr") else "n/a",
        "SDNN (ms)": f"{hrv_check['sdnn']:.1f}" if hrv_check.get("sdnn") else "n/a",
        "RMSSD (ms)": f"{hrv_check['rmssd']:.1f}" if hrv_check.get("rmssd") else "n/a",
    }

    hr_official = official.get("HR (bpm)")
    hr_recomputed = recomputed.get("HR (bpm)")
    discrepancy = None
    try:
        if hr_official not in (None, "NaN") and hr_recomputed != "n/a":
            diff = abs(float(hr_official) - float(hr_recomputed))
            if diff > 15:
                discrepancy = f"⚠ {diff:.0f} bpm difference between official (Phase 3) and recomputed HR"
    except ValueError:
        pass

    compare = html.Div(className="compare-grid", children=[
        html.Div(className="compare-col", children=[
            html.Div("Phase 3 · official", className="compare-col__title"),
            kv_grid(list(official.items())) if official else
            html.Div("phase3_hrv_features.csv not found", className="metric-card__sub"),
        ]),
        html.Div(className="compare-col", children=[
            html.Div("Recomputed · display-only cross-check", className="compare-col__title"),
            kv_grid(list(recomputed.items())),
            html.Div(f"{len(hrv_check.get('peaks', []))} R-peaks detected",
                     className="metric-card__sub"),
        ]),
    ])

    extra_pairs = []
    if hrv_df is not None and epoch_idx < len(hrv_df):
        row = hrv_df.iloc[epoch_idx]
        for key, label in [("lf_hf_ratio", "LF/HF ratio"), ("hf_power", "HF power"),
                           ("resp_rate_est", "Resp rate (br/min)"), ("sampen", "SampEn"),
                           ("pnn20", "pNN20 (%)"), ("perm_en", "PermEn")]:
            if key in hrv_df.columns:
                val = row[key]
                extra_pairs.append((label, f"{val:.3f}" if pd.notna(val) else "NaN"))

    children = [
        glance,
        html.Div(badges, className="badge-row"),
        time_line,
        html.Div(discrepancy, className="discrepancy-note") if discrepancy else None,
        compare,
        html.Div("Additional Phase 3 features", className="compare-col__title",
                style={"marginTop": "14px"}) if extra_pairs else None,
        kv_grid(extra_pairs) if extra_pairs else None,
    ]
    return html.Div([c for c in children if c is not None], className="epoch-detail")


def hrv_feature_table_row(data, epoch_idx):
    hrv_df = data["hrv_df"]
    if hrv_df is None or epoch_idx >= len(hrv_df):
        return []
    row = hrv_df.iloc[epoch_idx]
    numeric_cols = [c for c in row.index if pd.api.types.is_number(row[c])]
    out = pd.DataFrame({"feature": numeric_cols, "value": [row[c] for c in numeric_cols]})
    out["value"] = out["value"].astype(float).round(4)
    return out.to_dict("records")


def encode_image(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


# ──────────────────────────────────────────────────────────────────────────
# Raw JSON preview parsing (standalone — does not depend on the pipeline
# package being importable, so this tab works even outside project root)
# ──────────────────────────────────────────────────────────────────────────
def summarize_raw_device_json(payload):
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) \
            and "value" in payload[0] and "admissionId" in payload[0]:
        admission_ids, facility_ids, lens, ts_list, nonzero = set(), set(), set(), [], 0
        for rec in payload:
            admission_ids.add(rec.get("admissionId"))
            facility_ids.add(rec.get("facilityId"))
            v = rec.get("value")
            packet = v[0] if isinstance(v, list) and v and isinstance(v[0], list) else v
            if isinstance(packet, list):
                lens.add(len(packet))
                if any(x != 0 for x in packet):
                    nonzero += 1
            ts = rec.get("utcTimestamp")
            if isinstance(ts, dict):
                ts = ts.get("$date")
            if ts:
                ts_list.append(ts)
        return {
            "format": "new packetized (list of {admissionId, value, utcTimestamp, ...})",
            "n_packets": len(payload),
            "admissionIds": list(admission_ids),
            "facilityIds": list(facility_ids),
            "packet_inner_lengths_seen": list(lens),
            "packets_with_nonzero_signal": f"{nonzero}/{len(payload)}",
            "first_timestamp": min(ts_list) if ts_list else None,
            "last_timestamp": max(ts_list) if ts_list else None,
        }
    if isinstance(payload, dict):
        return {
            "format": "legacy single-chunk dict",
            "admissionId": payload.get("admissionId"),
            "window_start_ms": payload.get("window_start_ms"),
            "window_end_ms": payload.get("window_end_ms"),
            "duration_s": payload.get("duration_s"),
            "sample_count": payload.get("sample_count"),
            "keys": list(payload.keys()),
        }
    return {"format": "UNRECOGNIZED", "top_level_type": str(type(payload))}


# ──────────────────────────────────────────────────────────────────────────
# App shell / dark-theme styling
# ──────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="SleepECG · Clinical Console", suppress_callback_exceptions=True)
server = app.server

APP_CSS = f"""
:root {{
  --bg-app: {COLOR['bg_app']}; --bg-sidebar: {COLOR['bg_sidebar']}; --bg-panel: {COLOR['bg_panel']};
  --bg-panel-alt: {COLOR['bg_panel_alt']}; --bg-surface: {COLOR['bg_surface']}; --bg-hover: {COLOR['bg_hover']};
  --border: {COLOR['border']}; --border-soft: {COLOR['border_soft']};
  --text-primary: {COLOR['text_primary']}; --text-secondary: {COLOR['text_secondary']}; --text-muted: {COLOR['text_muted']};
  --accent: {COLOR['accent']}; --accent-dim: {COLOR['accent_dim']}; --accent-blue: {COLOR['accent_blue']}; --accent-purple: {COLOR['accent_purple']};
  --good: {COLOR['good']}; --good-bg: {COLOR['good_bg']}; --warn: {COLOR['warn']}; --warn-bg: {COLOR['warn_bg']};
  --bad: {COLOR['bad']}; --bad-bg: {COLOR['bad_bg']}; --flag: {COLOR['flag']}; --flag-bg: {COLOR['flag_bg']};
  --neutral-bg: {COLOR['neutral_bg']};
  --mono: 'JetBrains Mono', 'Courier New', monospace;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; background: var(--bg-app); color: var(--text-primary);
       font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }}
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: var(--bg-app); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}

.app-shell {{ display: flex; min-height: 100vh; }}

/* ── Sidebar ── */
.sidebar {{ width: 236px; flex: 0 0 236px; background: var(--bg-sidebar); color: #fff;
           display: flex; flex-direction: column; border-right: 1px solid var(--border-soft); }}
.sidebar-brand {{ padding: 22px 20px 16px; border-bottom: 1px solid var(--border-soft);
                  display: flex; align-items: center; gap: 10px; }}
.sidebar-brand__mark {{ width: 30px; height: 30px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent-blue));
    display: flex; align-items: center; justify-content: center; font-weight: 700; color: #06111a; font-size: 14px; flex: 0 0 30px; }}
.sidebar-brand__title {{ font-weight: 700; font-size: 15px; letter-spacing: 0.2px; color: var(--text-primary); }}
.sidebar-brand__sub {{ font-size: 11px; color: var(--text-muted); margin-top: 1px; }}
.sidebar-foot {{ margin-top: auto; padding: 14px 20px; font-size: 10.5px; color: var(--text-muted);
                border-top: 1px solid var(--border-soft); line-height: 1.5; }}
.sidebar-foot__dot {{ display: inline-block; width: 6px; height: 6px; border-radius: 50%;
                      background: var(--good); margin-right: 6px; }}

.nav-tabs-wrapper {{ border: none !important; flex: 1; padding: 14px 10px; }}
.nav-tabs {{ border: none !important; }}
.nav-tab {{ background: transparent !important; border: none !important;
           color: var(--text-secondary) !important; padding: 10px 13px !important; font-size: 13px !important;
           text-align: left !important; border-radius: 9px !important; margin-bottom: 3px !important;
           line-height: 1.4 !important; letter-spacing: 0.1px; display: flex !important;
           align-items: center !important; gap: 10px !important; transition: background 0.15s, color 0.15s; }}
.nav-tab:hover {{ background: var(--bg-hover) !important; color: var(--text-primary) !important; }}
.nav-tab--selected {{ background: var(--bg-panel-alt) !important; color: #fff !important;
                      box-shadow: inset 3px 0 0 var(--accent); font-weight: 600 !important; }}

/* ── Main column ── */
.main {{ flex: 1; min-width: 0; display: flex; flex-direction: column; }}
.topbar {{ background: var(--bg-panel); border-bottom: 1px solid var(--border-soft);
          padding: 14px 28px; display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
          position: sticky; top: 0; z-index: 20; }}
.topbar-label {{ font-size: 12px; color: var(--text-muted); margin-right: 2px; }}
.content {{ padding: 24px 28px 48px; flex: 1; }}

button.refresh-btn {{ background: var(--bg-surface); border: 1px solid var(--border);
    color: var(--text-secondary); padding: 7px 14px; border-radius: 7px; font-size: 12.5px;
    cursor: pointer; transition: all 0.15s; }}
button.refresh-btn:hover {{ border-color: var(--accent); color: var(--accent); }}

.scan-summary {{ font-size: 11px; color: var(--text-muted); font-family: var(--mono);
                 white-space: nowrap; }}

.jump-input {{ background: var(--bg-surface) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 7px !important; padding: 6px 10px !important;
    font-family: var(--mono) !important; font-size: 12.5px !important; width: 100px !important; }}

/* ── Dropdown (dark) ── */
.Select-control, .dash-dropdown .Select-control {{ background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important; }}
#session-dropdown .Select-menu-outer, #feature-select .Select-menu-outer {{ background: var(--bg-surface) !important; z-index: 50 !important;}}
.dash-dropdown div, .dash-dropdown input {{ color: var(--text-primary) !important; }}
.VirtualizedSelectOption {{ background: var(--bg-surface) !important; color: var(--text-primary) !important; }}
.VirtualizedSelectFocusedOption {{ background: var(--bg-hover) !important; }}

/* ── Metric / KPI cards ── */
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(196px, 1fr));
               gap: 14px; margin-bottom: 26px; }}
.metric-card {{ background: var(--bg-panel); border: 1px solid var(--border-soft); border-radius: 12px;
               padding: 16px 17px; display: flex; gap: 11px; align-items: flex-start;
               transition: border-color 0.15s; }}
.metric-card:hover {{ border-color: var(--border); }}
.metric-dot {{ width: 9px; height: 9px; border-radius: 50%; margin-top: 6px; flex: 0 0 9px;
              box-shadow: 0 0 8px currentColor; }}
.metric-dot--good {{ background: var(--good); color: var(--good); }}
.metric-dot--warn {{ background: var(--warn); color: var(--warn); }}
.metric-dot--bad {{ background: var(--bad); color: var(--bad); }}
.metric-dot--neutral {{ background: var(--text-muted); color: var(--text-muted); box-shadow: none; }}
.metric-card__label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase;
                       letter-spacing: 0.5px; margin-bottom: 5px; }}
.metric-card__value {{ font-family: var(--mono); font-size: 22px; font-weight: 600; color: var(--text-primary); }}
.metric-card__sub {{ font-size: 11.5px; color: var(--text-muted); margin-top: 5px; }}

.stage-bar {{ display: flex; height: 7px; border-radius: 4px; overflow: hidden; margin-top: 7px; width: 100%; }}
.stage-bar__seg {{ height: 100%; }}
.stage-bar__legend {{ font-size: 10.5px; color: var(--text-muted); margin-top: 6px; }}
.legend-dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 3px; }}
.legend-item {{ margin-right: 10px; }}

/* ── Timeline legend (custom, replaces Plotly's built-in legend to avoid
   it colliding with subplot titles) ── */
.timeline-legend {{ display: flex; flex-direction: column; gap: 8px; padding: 4px 4px 14px; }}
.legend-strip {{ display: flex; flex-wrap: wrap; gap: 8px 18px; }}
.legend-chip {{ display: inline-flex; align-items: center; gap: 6px; font-size: 11.5px;
               color: var(--text-secondary); white-space: nowrap; }}
.legend-swatch {{ display: inline-block; flex: 0 0 auto; }}
.legend-swatch--dot {{ width: 10px; height: 10px; border-radius: 50%; }}
.legend-swatch--marker {{ font-size: 13px; line-height: 1; width: 14px; text-align: center; }}
.legend-swatch--line {{ width: 16px; height: 2px; border-radius: 1px; }}
.legend-swatch--dashed {{ height: 0; border-top: 2px dashed; background: transparent !important; }}

/* ── Pills / badges ── */
.pill {{ display: inline-block; font-size: 11.5px; padding: 3px 10px; border-radius: 999px;
        border: 1px solid transparent; margin-right: 6px; margin-bottom: 4px; font-weight: 500; }}
.pill--good {{ background: var(--good-bg); color: var(--good); }}
.pill--warn {{ background: var(--warn-bg); color: var(--warn); }}
.pill--bad {{ background: var(--bad-bg); color: var(--bad); }}
.pill--flag {{ background: var(--flag-bg); color: var(--flag); }}
.pill--neutral {{ background: var(--neutral-bg); color: var(--text-secondary); }}
.pill--stage {{ background: var(--bg-surface); border-width: 1.5px; font-weight: 600; }}
.badge-row {{ margin-bottom: 10px; }}

/* ── Section cards / phase timeline ── */
.section-card {{ background: var(--bg-panel); border: 1px solid var(--border-soft); border-radius: 12px;
                margin-bottom: 18px; overflow: hidden; }}
.section-card__header {{ padding: 14px 18px; border-bottom: 1px solid var(--border-soft);
                        display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
.section-card__title {{ font-weight: 600; font-size: 14.5px; color: var(--text-primary); }}
.section-card__subtitle {{ font-size: 12px; color: var(--text-muted); margin-top: 2px; }}
.section-card__body {{ padding: 16px 18px; }}

.phase-step {{ display: flex; gap: 14px; }}
.phase-step__rail {{ display: flex; flex-direction: column; align-items: center; }}
.phase-step__marker {{ width: 27px; height: 27px; border-radius: 50%; display: flex;
    align-items: center; justify-content: center; font-family: var(--mono); font-size: 12px;
    font-weight: 700; color: #06111a; flex: 0 0 27px; }}
.phase-step__marker--good {{ background: var(--good); }}
.phase-step__marker--bad {{ background: var(--text-muted); color: #0a0e16; }}
.phase-step__line {{ width: 2px; flex: 1; background: var(--border); margin-top: 4px; min-height: 14px; }}
.phase-step:last-child .phase-step__line {{ display: none; }}
.phase-step__content {{ padding-bottom: 22px; flex: 1; min-width: 0; }}
.phase-step__title {{ font-weight: 600; font-size: 13.5px; margin: 3px 0 8px; color: var(--text-primary);
                      display: flex; align-items: center; gap: 8px; }}

.kv-grid {{ display: grid; grid-template-columns: max-content 1fr; gap: 5px 14px; }}
.kv-grid--wide {{ margin-top: 10px; }}
.kv-key {{ font-size: 12px; color: var(--text-muted); }}
.kv-val {{ font-family: var(--mono); font-size: 12px; color: var(--text-primary); word-break: break-word; }}

/* ── Explorer ── */
.epoch-toolbar {{ display: grid; grid-template-columns: 90px 1fr 90px; align-items: center;
                  gap: 14px; margin: 4px 0 18px; }}
.nav-btn {{ background: var(--bg-surface); color: var(--text-primary); border: 1px solid var(--border);
           padding: 9px 0; border-radius: 8px; cursor: pointer; font-size: 13px; transition: all 0.15s; }}
.nav-btn:hover {{ background: var(--accent-dim); border-color: var(--accent); color: #fff; }}

.explorer-grid {{ display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }}
.explorer-grid > div:first-child {{ flex: 1.35; min-width: 420px; }}
.explorer-grid > div:last-child {{ flex: 1; min-width: 340px; }}

.epoch-detail__time {{ font-family: var(--mono); font-size: 12px; color: var(--text-secondary);
                       margin-bottom: 12px; }}
.epoch-glance {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(84px, 1fr));
                gap: 10px; background: var(--bg-panel-alt); border: 1px solid var(--border-soft);
                border-radius: 10px; padding: 12px 14px; margin-bottom: 14px; }}
.epoch-glance__item {{ text-align: left; }}
.epoch-glance__label {{ font-size: 9.5px; letter-spacing: 0.5px; color: var(--text-muted);
                        text-transform: uppercase; margin-bottom: 3px; }}
.epoch-glance__value {{ font-family: var(--mono); font-size: 15px; font-weight: 600;
                        color: var(--text-primary); }}
.discrepancy-note {{ background: var(--warn-bg); color: var(--warn); font-size: 12px;
                     padding: 8px 12px; border-radius: 7px; margin-bottom: 12px; }}
.compare-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 6px; }}
.compare-col__title {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px;
                       color: var(--text-muted); margin-bottom: 8px; }}

.feature-select-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }}

/* ── Phase analysis ── */
details.phase-group {{ background: var(--bg-panel); border: 1px solid var(--border-soft);
                       border-radius: 12px; margin-bottom: 14px; padding: 4px 18px; }}
details.phase-group > summary {{ padding: 13px 0; font-weight: 600; font-size: 13.5px;
                                 cursor: pointer; list-style: none; display: flex;
                                 align-items: center; justify-content: space-between; color: var(--text-primary); }}
details.phase-group > summary::-webkit-details-marker {{ display: none; }}
details.phase-group > summary::before {{ content: "▸ "; color: var(--accent); }}
details.phase-group[open] > summary::before {{ content: "▾ "; color: var(--accent); }}
.phase-group__count {{ font-size: 11.5px; color: var(--text-muted); font-weight: 400; }}
.phase-img-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
                   gap: 14px; padding-bottom: 16px; }}
.phase-img-card {{ background: var(--bg-panel-alt); border-radius: 9px; padding: 8px; border: 1px solid var(--border-soft); }}
.phase-img-card img {{ width: 100%; border-radius: 6px; border: 1px solid var(--border-soft); display: block;
                       background: #fff; }}
.phase-img-card__caption {{ font-size: 11.5px; color: var(--text-secondary); margin-top: 8px;
                            font-family: var(--mono); }}
.log-box {{ background: #05080e; border: 1px solid var(--border-soft); border-radius: 8px; padding: 12px 14px;
           font-family: var(--mono); font-size: 11px; color: var(--text-secondary); white-space: pre-wrap;
           max-height: 320px; overflow-y: auto; line-height: 1.5; }}

/* ── Upload ── */
.upload-zone {{ border: 1.5px dashed var(--border); border-radius: 12px; padding: 30px;
               text-align: center; color: var(--text-secondary); background: var(--bg-panel); cursor: pointer; }}
.upload-zone:hover {{ border-color: var(--accent); color: var(--accent); }}

.empty-state {{ color: var(--text-muted); font-size: 13px; padding: 48px 0; text-align: center; }}
.empty-state__icon {{ font-size: 26px; display: block; margin-bottom: 10px; opacity: 0.6; }}

/* ── DataTable dark overrides ── */
.dash-table-container .dash-spreadsheet-container {{ font-family: var(--mono) !important; }}
.dash-spreadsheet-container .dash-spreadsheet-inner table {{ background: var(--bg-panel) !important; }}
.dash-spreadsheet-container th {{ background: var(--bg-panel-alt) !important; color: var(--text-secondary) !important;
    border-color: var(--border-soft) !important; }}
.dash-spreadsheet-container td {{ background: var(--bg-panel) !important; color: var(--text-primary) !important;
    border-color: var(--border-soft) !important; }}
.dash-spreadsheet-container input {{ background: var(--bg-surface) !important; color: var(--text-primary) !important; }}
.dash-filter input {{ color: var(--text-primary) !important; background: var(--bg-surface) !important;
    border-radius: 5px !important; }}
.dash-filter input::placeholder {{ color: var(--text-muted) !important; opacity: 1 !important; }}
.previous-page, .next-page, .page-number {{ color: var(--text-secondary) !important; }}
.dash-spreadsheet-container .dash-freeze-left {{ background: var(--bg-panel) !important;
    box-shadow: 2px 0 6px rgba(0,0,0,0.35); }}

/* ── KPI chip in topbar ── */
.chip-row {{ display: flex; gap: 6px; flex-wrap: wrap; }}

@media (max-width: 880px) {{
  .app-shell {{ flex-direction: column; }}
  .sidebar {{ width: 100%; flex: none; flex-direction: row; overflow-x: auto; }}
  .sidebar-brand, .sidebar-foot {{ display: none; }}
  .nav-tabs-wrapper {{ padding: 8px; }}
  .compare-grid {{ grid-template-columns: 1fr; }}
  .explorer-grid > div {{ min-width: 100%; }}
}}
"""

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>{APP_CSS}</style>
    {{%css%}}
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>
"""

GRAPH_CONFIG = dict(scrollZoom=True, displaylogo=False,
                    modeBarButtonsToRemove=["lasso2d", "select2d"])

session_selector = html.Div([
    html.Span("Admission ID", className="topbar-label"),
    dcc.Dropdown(id="session-dropdown", options=session_dropdown_options(), value=None,
                 placeholder="Search / select an Admission ID…", searchable=True,
                 style={"width": "360px", "fontSize": "13px"}, clearable=False),
    html.Button("↻ Refresh", id="refresh-btn", n_clicks=0, className="refresh-btn",
               title="Re-scan Test_dataset + results_test for new / updated sessions"),
    html.Div(id="dataset-scan-summary", children=dataset_scan_summary(),
            className="scan-summary"),
    html.Div(id="session-meta", className="chip-row"),
], style={"display": "flex", "alignItems": "center", "gap": "10px", "flexWrap": "wrap"})

sidebar = html.Div(className="sidebar", children=[
    html.Div(className="sidebar-brand", children=[
        html.Div("◈", className="sidebar-brand__mark"),
        html.Div([
            html.Div("SleepECG", className="sidebar-brand__title"),
            html.Div("Clinical Analytics Console", className="sidebar-brand__sub"),
        ]),
    ]),
    dcc.Tabs(id="tabs", value="tab-overview", vertical=True,
             parent_className="nav-tabs-wrapper", className="nav-tabs",
             children=[
        dcc.Tab(label="📊 Overview", value="tab-overview",
               className="nav-tab", selected_className="nav-tab--selected"),
        dcc.Tab(label="🫀 Signal Explorer", value="tab-explorer",
               className="nav-tab", selected_className="nav-tab--selected"),
        dcc.Tab(label="🗂 Phase Analysis", value="tab-phases",
               className="nav-tab", selected_className="nav-tab--selected"),
        dcc.Tab(label="📈 Feature Table", value="tab-features",
               className="nav-tab", selected_className="nav-tab--selected"),
        dcc.Tab(label="🧾 Raw JSON Inspector", value="tab-raw",
               className="nav-tab", selected_className="nav-tab--selected"),
    ]),
    html.Div([
        html.Span(className="sidebar-foot__dot"),
        "Read-only layer — Testing_code pipeline is never modified or re-run.",
    ], className="sidebar-foot"),
])

app.layout = html.Div(className="app-shell", children=[
    sidebar,
    html.Div(className="main", children=[
        html.Div(className="topbar", children=[session_selector]),
        dcc.Loading(html.Div(id="tab-content", className="content"),
                   type="circle", color=COLOR["accent"]),
    ]),
    dcc.Store(id="epoch-index-store", data=0),
    dcc.Store(id="n-epochs-store", data=0),
])


# ── Tab content renderers ───────────────────────────────────────────────
def render_overview_tab(session_id):
    if not session_id:
        return html.Div([
            html.Span("🩺", className="empty-state__icon"),
            "Select an Admission ID above to view its pipeline status.",
        ], className="empty-state")
    data = load_session(session_id)
    if data is None:
        return html.Div(f"Admission ID '{session_id}' not found under {TEST_DATASET_ROOT} "
                        f"or {RESULTS_ROOT}.", className="empty-state")

    m = compute_summary_metrics(data)

    cards = html.Div(className="metric-grid", children=[
        metric_card("Session Duration", m["duration_label"],
                   f"{m['n_epochs']} epochs @ {m['epoch_sec']}s", icon="⏱"),
        metric_card("ECG Signal Quality", m["sqi_label"], "mean SQI · Phase 1", m["sqi_status"], icon="📶"),
        metric_card("Mean Heart Rate", m["hr_label"], "Phase 3 HRV", icon="❤"),
        metric_card("Mean Respiration Rate", m["rr_label"], "Phase 3 HRV", icon="🌬"),
        metric_card("Dominant Stage", m["dominant_label"],
                   stage_distribution_bar(m["stage_counts"], m["n_epochs"]), icon="🌙"),
        metric_card("Artifact Epochs", f"{m['n_bad']} · {m['bad_pct']:.1f}%",
                   "flagged by Phase 1 bad_epoch_mask", m["artifact_status"], icon="⚠"),
        metric_card("Processing Status", m["proc_label"], "Phase 0 → Phase 5", m["proc_status"], icon="✓"),
    ])

    extra = []
    if m["n_mismatch"]:
        extra.append(html.Div(status_pill(
            f"{m['n_mismatch']} epoch(s) where the raw RF prediction disagrees with the "
            "smoothed hypnogram — flagged for review in Signal Explorer.", "flag"),
            style={"marginTop": "-10px", "marginBottom": "22px"}))

    ri = data["record_info"]
    m1 = data["phase1_meta"]
    p2row = data["phase2_consolidated_row"]
    hrv_df = data["hrv_df"]
    pm = data["predict_meta"]
    p5m = data["phase5_meta"]
    phase0_png = any(os.path.basename(p).lower().startswith("phase0")
                     for p in data.get("phase_pngs", []))
    phase0_done = bool(ri) or phase0_png

    if ri:
        phase0_body = kv_grid(list(ri.items()))
    elif phase0_png:
        phase0_body = html.Div([
            html.Div("record_info.json not found, but phase0_raw_ecg_overview.png exists — "
                    "Phase 0 (scan + inspect) completed for this session. record_info.json is "
                    "only written by the separate p0_metadata_extraction.extract_all_metadata "
                    "step, which run_phase0_test.py doesn't call by default.",
                    className="metric-card__sub"),
            html.Div("See the ECG overview plot under Phase Analysis → Phase 0.",
                    className="metric-card__sub", style={"marginTop": "4px"}),
        ])
    else:
        phase0_body = html.Div("Neither record_info.json nor phase0_raw_ecg_overview.png found "
                               "— Phase 0 has not produced output for this session yet.",
                               className="metric-card__sub")

    timeline = html.Div([
        phase_step(0, "Dataset scan / metadata", "good" if phase0_done else "bad", phase0_body),
        phase_step(1, "Preprocessing — DWT, SQI, epoching",
                  "good" if (m1 or data["epochs"] is not None) else "bad",
                  kv_grid([
                      ("epoch_sec", m1.get("epoch_sec", "?")),
                      ("target_fs", m1.get("target_fs", "?")),
                      ("dwt_wavelet", m1.get("dwt_wavelet", "?")),
                      ("dwt_level", m1.get("dwt_level", "?")),
                      ("mean_sqi", m1.get("mean_sqi", "?")),
                      ("n_epochs (preprocessed_epochs.npy)",
                       len(data["epochs"]) if data["epochs"] is not None else "N/A"),
                      ("bad epochs",
                       int(np.sum(data["bad_mask"])) if data["bad_mask"] is not None else "N/A"),
                  ]) if (m1 or data["epochs"] is not None) else
                  html.Div("phase1_meta.json not found.", className="metric-card__sub")),
        phase_step(2, "R-peak / RR / HR", "good" if p2row else "bad",
                  kv_grid(list(p2row.items())) if p2row else
                  html.Div("No matching row found in a *consolidated*.csv under RESULTS_ROOT.",
                          className="metric-card__sub")),
        phase_step(3, "HRV feature extraction", "good" if hrv_df is not None else "bad",
                  kv_grid([
                      ("n_epochs", len(hrv_df)),
                      ("n_feature_columns", hrv_df.shape[1]),
                      ("columns (first 12)", ", ".join(map(str, hrv_df.columns[:12]))),
                  ]) if hrv_df is not None else
                  html.Div("phase3_hrv_features.csv not found.", className="metric-card__sub")),
        phase_step("4B", "Random-forest sleep-stage prediction", "good" if pm else "bad",
                  html.Div([
                      kv_grid(list(pm.items())) if pm else
                      html.Div("phase4b_predict_meta.json not found.", className="metric-card__sub"),
                      html.Div(f"raw vs smoothed disagreement: {m['n_mismatch']} epoch(s)",
                              className="metric-card__sub", style={"marginTop": "8px"}),
                  ])),
        phase_step(5, "Final report", "good" if (p5m or data["duration_df"] is not None) else "bad",
                  html.Div([
                      kv_grid(list(p5m.items())) if p5m else
                      html.Div("phase5_meta.json not found.", className="metric-card__sub"),
                      dash_table.DataTable(
                          data=data["duration_df"].round(3).to_dict("records"),
                          columns=[{"name": c, "id": c} for c in data["duration_df"].columns],
                          style_table={"marginTop": "10px", "overflowX": "auto"},
                          style_cell={"fontFamily": "JetBrains Mono, monospace", "fontSize": 11,
                                     "padding": "6px 10px", "backgroundColor": COLOR["bg_panel"],
                                     "color": COLOR["text_primary"], "border": f"1px solid {COLOR['border_soft']}"},
                          style_header={"backgroundColor": COLOR["bg_panel_alt"], "fontWeight": "600",
                                       "color": COLOR["text_secondary"], "border": f"1px solid {COLOR['border_soft']}"},
                          page_size=8,
                      ) if data["duration_df"] is not None else None,
                  ])),
    ])

    return html.Div([
        *extra,
        cards,
        section_card("Pipeline trace", timeline,
                    subtitle=f"Output directory: {data['output_dir']}"),
    ])


def render_explorer_tab(session_id, n_epochs, epoch_idx):
    if not session_id:
        return html.Div([
            html.Span("🫀", className="empty-state__icon"),
            "Select an Admission ID above to explore its hypnogram, ECG and HRV.",
        ], className="empty-state")
    data = load_session(session_id)
    hrv_df = data["hrv_df"] if data else None
    feature_options = []
    if hrv_df is not None:
        feature_options = [{"label": c, "value": c} for c in hrv_df.columns
                           if pd.api.types.is_numeric_dtype(hrv_df[c])]
    default_feature = next((o["value"] for o in feature_options
                            if o["value"] in ("mean_hr", "rmssd", "sdnn")), None) \
                       or (feature_options[0]["value"] if feature_options else None)

    return html.Div([
        section_card("Full-night synchronized timeline",
                    html.Div([
                        timeline_legend(),
                        dcc.Graph(id="hypnogram-graph", config=GRAPH_CONFIG),
                    ]),
                    subtitle="Hypnogram / HR & RR / SQI share one clock — scroll, drag the range "
                             "slider, or click any segment to jump the inspector below"),
        html.Div([
            html.Button("⏮ Prev", id="prev-btn", n_clicks=0, className="nav-btn"),
            dcc.Slider(id="epoch-slider", min=0, max=max(n_epochs - 1, 0), step=1,
                      value=epoch_idx or 0, tooltip={"placement": "bottom"}, updatemode="mouseup"),
            html.Button("Next ⏭", id="next-btn", n_clicks=0, className="nav-btn"),
        ], className="epoch-toolbar"),
        html.Div(className="explorer-grid", children=[
            html.Div(section_card("30-second ECG window",
                                  dcc.Graph(id="ecg-graph", config=GRAPH_CONFIG))),
            html.Div(section_card("Epoch inspector — root-cause view", html.Div(id="epoch-detail"))),
        ]),
        section_card(
            "Feature trend explorer",
            html.Div([
                html.Div([
                    html.Span("Feature:", className="topbar-label"),
                    dcc.Dropdown(id="feature-select", options=feature_options, value=default_feature,
                               clearable=False, style={"width": "280px", "fontSize": "12.5px"}),
                ], className="feature-select-row"),
                dcc.Graph(id="feature-trend-graph", config=GRAPH_CONFIG),
            ]),
            subtitle="Any Phase 3 feature plotted across the whole night, colored by predicted stage",
        ),
        section_card("Full HRV feature vector for this epoch (all Phase 3 columns)",
                    dash_table.DataTable(id="hrv-table",
                        columns=[{"name": "feature", "id": "feature"},
                                {"name": "value", "id": "value"}],
                        page_size=12, style_table={"maxWidth": "520px"},
                        style_cell={"fontFamily": "JetBrains Mono, monospace", "fontSize": 12,
                                   "textAlign": "left", "padding": "6px 10px",
                                   "backgroundColor": COLOR["bg_panel"], "color": COLOR["text_primary"],
                                   "border": f"1px solid {COLOR['border_soft']}"},
                        style_header={"backgroundColor": COLOR["bg_panel_alt"], "fontWeight": "600",
                                     "color": COLOR["text_secondary"], "border": f"1px solid {COLOR['border_soft']}"},
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": COLOR["bg_panel_alt"]}],
                    )),
    ])


PHASE_GROUPS = [
    ("phase0", "Phase 0 — Dataset scan"),
    ("phase1", "Phase 1 — Preprocessing"),
    ("phase2", "Phase 2 — R-peak / RR / HR"),
    ("phase3", "Phase 3 — HRV features"),
    ("phase4b", "Phase 4B — RF sleep-stage prediction"),
    ("phase5", "Phase 5 — Final report"),
]


def render_phase_analysis_tab(session_id):
    if not session_id:
        return html.Div([
            html.Span("🗂", className="empty-state__icon"),
            "Select an Admission ID above.",
        ], className="empty-state")
    data = load_session(session_id)
    if data is None:
        return html.Div(f"Session '{session_id}' not found.", className="empty-state")

    log_section = None
    if data.get("log_tail"):
        log_section = section_card("Processing log (tail)", html.Pre(data["log_tail"], className="log-box"),
                                   subtitle=data.get("log_path"))

    if not data["phase_pngs"]:
        return html.Div([
            html.Div([
                html.Span("🗂", className="empty-state__icon"),
                "No PNG plots found in this session's output directory yet "
                "(run the phaseN_test.py drivers first).",
            ], className="empty-state"),
            log_section,
        ])

    used = set()
    groups = []
    for prefix, title in PHASE_GROUPS:
        matches = [p for p in data["phase_pngs"]
                  if os.path.basename(p).lower().startswith(prefix)]
        used.update(matches)
        if matches:
            groups.append((title, matches))
    leftover = [p for p in data["phase_pngs"] if p not in used]
    if leftover:
        groups.append(("Other plots", leftover))

    sections = []
    for i, (title, files) in enumerate(groups):
        cards = [
            html.Div(className="phase-img-card", children=[
                html.Img(src=encode_image(p)),
                html.Div(os.path.basename(p), className="phase-img-card__caption"),
            ]) for p in files
        ]
        sections.append(html.Details(open=(i == 0), className="phase-group", children=[
            html.Summary([title, html.Span(f"{len(files)} plot(s)", className="phase-group__count")]),
            html.Div(cards, className="phase-img-grid"),
        ]))
    return html.Div([*sections, log_section] if log_section else sections)


def render_features_tab(session_id):
    if not session_id:
        return html.Div([
            html.Span("📈", className="empty-state__icon"),
            "Select an Admission ID above.",
        ], className="empty-state")
    data = load_session(session_id)
    hrv_df = data["hrv_df"] if data else None
    if hrv_df is None:
        return html.Div("phase3_hrv_features.csv not found for this session.", className="empty-state")

    df = hrv_df.copy()

    # phase3_hrv_features.csv already carries its own "epoch_idx" column
    # (see p3_pipeline.py's _build_feature_matrix) — reuse it instead of
    # inserting a duplicate (that raised ValueError: cannot insert epoch_idx,
    # already exists), but still guarantee it exists and sits first.
    if "epoch_idx" in df.columns:
        df["epoch_idx"] = pd.to_numeric(df["epoch_idx"], errors="coerce")
        if df["epoch_idx"].isna().any():
            df["epoch_idx"] = np.arange(len(df))
        df["epoch_idx"] = df["epoch_idx"].astype(int)
        df.insert(0, "epoch_idx", df.pop("epoch_idx"))
    else:
        df.insert(0, "epoch_idx", np.arange(len(df)))

    pred = data.get("pred_smooth")
    if pred is not None and len(pred) >= len(df) and "predicted_stage" not in df.columns:
        df.insert(1, "predicted_stage",
                 [STAGE_NAMES.get(int(pred[i]), "?") if i < len(pred) else "?"
                  for i in df["epoch_idx"]])

    cols = []
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            cols.append({"name": c, "id": c, "type": "text"})
        elif pd.api.types.is_numeric_dtype(df[c]):
            cols.append({"name": c, "id": c, "type": "numeric"})
        else:
            cols.append({"name": c, "id": c, "type": "text"})
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    round_map = {c: 4 for c in numeric_cols}
    display_df = df.copy()
    for c in numeric_cols:
        display_df[c] = display_df[c].round(round_map[c])
    # Booleans render as True/False objects that the native text filter
    # can't match reliably — stringify so typing "true"/"false" works.
    for c in display_df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            display_df[c] = display_df[c].astype(str)

    n_frozen = 2 if "predicted_stage" in df.columns else 1
    return section_card(
        "Phase 3 · HRV feature vectors",
        html.Div([
            html.Div("Click any row to jump Signal Explorer to that epoch. Numeric columns "
                    "support filter expressions like >50, <=0.2, or =3 — text columns match "
                    "on contains (case-insensitive).",
                    style={"fontSize": "11.5px", "color": COLOR["text_muted"], "marginBottom": "8px"}),
            dash_table.DataTable(
                id="features-table",
                columns=cols, data=display_df.to_dict("records"),
                page_size=20, sort_action="native", sort_mode="multi",
                filter_action="native", filter_options={"case": "insensitive"},
                row_selectable=False, cell_selectable=True,
                fixed_columns={"headers": True, "data": n_frozen},
                style_table={"overflowX": "auto", "minWidth": "100%"},
                style_cell={"fontFamily": "JetBrains Mono, monospace", "fontSize": 11, "minWidth": 110,
                           "maxWidth": 220, "padding": "6px 8px", "backgroundColor": COLOR["bg_panel"],
                           "color": COLOR["text_primary"], "border": f"1px solid {COLOR['border_soft']}"},
                style_header={"backgroundColor": COLOR["bg_panel_alt"], "fontWeight": "600",
                             "textTransform": "uppercase", "fontSize": 10, "color": COLOR["text_secondary"],
                             "border": f"1px solid {COLOR['border_soft']}"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": COLOR["bg_panel_alt"]},
                    {"if": {"state": "active"}, "backgroundColor": COLOR["bg_hover"],
                     "border": f"1px solid {COLOR['accent']}"},
                    {"if": {"column_id": "epoch_idx"}, "fontWeight": "600", "color": COLOR["accent"]},
                ],
            ),
        ]),
        subtitle=f"{len(df)} epochs × {hrv_df.shape[1]} feature columns — sortable / filterable / searchable",
    )


def render_raw_tab():
    return section_card("Raw device JSON inspector", html.Div([
        html.P("Drag & drop a device JSON export here — both the legacy chunk format and the "
              "new packetized format are supported, matching the existing p0_scan_datasets logic. "
              "This is a standalone sanity check; it does not touch the pipeline.",
              style={"color": COLOR["text_secondary"], "fontSize": "13px", "marginTop": 0}),
        dcc.Upload(id="raw-json-upload",
                  children=html.Div(["Drag and drop, or ", html.A("select a JSON file")]),
                  className="upload-zone"),
        html.Div(id="raw-json-summary", style={"marginTop": "16px"}),
    ]))


# ── Callbacks ────────────────────────────────────────────────────────────
@app.callback(
    Output("session-dropdown", "options"),
    Output("dataset-scan-summary", "children"),
    Input("refresh-btn", "n_clicks"),
)
def refresh_sessions(_):
    clear_cache()
    return session_dropdown_options(), dataset_scan_summary()


@app.callback(
    Output("session-dropdown", "options", allow_duplicate=True),
    Output("dataset-scan-summary", "children", allow_duplicate=True),
    Input("session-dropdown", "id"), prevent_initial_call="initial_duplicate",
)
def initial_sessions(_):
    return session_dropdown_options(), dataset_scan_summary()


@app.callback(Output("session-meta", "children"), Input("session-dropdown", "value"))
def update_session_meta(session_id):
    if not session_id:
        return None
    data = load_session(session_id)
    if data is None:
        return status_pill("Admission ID not found in Test_dataset or results_test", "bad")
    if data.get("output_dir") is None:
        n_chunks = len(data.get("raw_chunks") or [])
        return [
            status_pill(f"{n_chunks} raw chunk(s) in Test_dataset", "neutral"),
            status_pill("Not processed — run Testing_code Phase 0 → 5 for this ID", "warn"),
        ]
    m = compute_summary_metrics(data)
    return [
        status_pill(m["duration_label"], "neutral"),
        status_pill(f"SQI {m['sqi_label']}", m["sqi_status"]),
        status_pill(m["hr_label"], "neutral"),
        status_pill(m["proc_label"], m["proc_status"]),
    ]


@app.callback(Output("n-epochs-store", "data"), Input("session-dropdown", "value"))
def on_session_change(session_id):
    if not session_id:
        return 0
    data = load_session(session_id)
    if data is None or data["pred_smooth"] is None:
        return 0
    return len(data["pred_smooth"])


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("session-dropdown", "value"),
    Input("n-epochs-store", "data"),
    State("epoch-index-store", "data"),
)
def render_tab(tab, session_id, n_epochs, epoch_idx):
    if tab == "tab-overview":
        return render_overview_tab(session_id)
    if tab == "tab-explorer":
        return render_explorer_tab(session_id, n_epochs or 0, epoch_idx)
    if tab == "tab-phases":
        return render_phase_analysis_tab(session_id)
    if tab == "tab-features":
        return render_features_tab(session_id)
    if tab == "tab-raw":
        return render_raw_tab()
    return html.Div()


# Synchronized timeline — rebuilt on session change / tab open, and its
# selection-cursor line is refreshed whenever the inspected epoch changes.
@app.callback(
    Output("hypnogram-graph", "figure"),
    Input("session-dropdown", "value"),
    Input("tabs", "value"),
    Input("epoch-index-store", "data"),
    prevent_initial_call=True,
)
def update_hypnogram(session_id, tab, epoch_idx):
    if tab != "tab-explorer" or not session_id:
        return dash.no_update
    data = load_session(session_id)
    if data is None:
        return go.Figure()
    return build_overview_figure(data, epoch_idx=epoch_idx)


# Single callback owns BOTH the epoch-index store and the slider value —
# avoids the circular dependency of store -> slider -> store. Clicking the
# hypnogram/HR/SQI tracks, dragging the slider, Prev/Next, and clicking a
# row in the Feature Table all funnel through here.
@app.callback(
    Output("epoch-index-store", "data"),
    Output("epoch-slider", "value"),
    Input("hypnogram-graph", "clickData"),
    Input("epoch-slider", "value"),
    Input("prev-btn", "n_clicks"),
    Input("next-btn", "n_clicks"),
    State("epoch-index-store", "data"),
    State("n-epochs-store", "data"),
    prevent_initial_call=True,
)
def update_epoch_index(clickData, slider_val, prev_clicks, next_clicks, current, n_epochs):
    trig = ctx.triggered_id
    n_epochs, current = n_epochs or 0, current or 0

    if trig == "hypnogram-graph" and clickData:
        idx = int(clickData["points"][0].get("customdata", current))
    elif trig == "epoch-slider":
        idx = int(slider_val)
    elif trig == "prev-btn":
        idx = max(0, current - 1)
    elif trig == "next-btn":
        idx = min(max(n_epochs - 1, 0), current + 1)
    else:
        idx = current

    idx = max(0, min(idx, max(n_epochs - 1, 0)))
    return idx, idx


@app.callback(
    Output("ecg-graph", "figure"),
    Output("epoch-detail", "children"),
    Output("hrv-table", "data"),
    Input("tabs", "value"),
    Input("epoch-index-store", "data"),
    State("session-dropdown", "value"),
    prevent_initial_call=True,
)
def on_epoch_change(tab, epoch_idx, session_id):
    # ecg-graph / epoch-detail / hrv-table only exist in the DOM while the
    # Signal Explorer tab is rendered. epoch-index-store can change from
    # OTHER tabs too (e.g. a Feature Table row click), so this callback
    # must no-op — not emit a real value — whenever that tab isn't active,
    # exactly like update_hypnogram already does. Including "tabs" as an
    # Input (not just State) means switching INTO the Explorer tab also
    # re-evaluates this and paints the epoch that was selected elsewhere.
    if tab != "tab-explorer" or not session_id or epoch_idx is None:
        return dash.no_update, dash.no_update, dash.no_update
    data = load_session(session_id)
    if data is None:
        return go.Figure(), None, []

    ecg_fig, hrv_check = build_ecg_figure(data, epoch_idx)
    detail = build_epoch_detail(data, epoch_idx, hrv_check)
    table_data = hrv_feature_table_row(data, epoch_idx)
    return ecg_fig, detail, table_data


@app.callback(
    Output("feature-trend-graph", "figure"),
    Input("tabs", "value"),
    Input("feature-select", "value"),
    Input("epoch-index-store", "data"),
    State("session-dropdown", "value"),
    prevent_initial_call=True,
)
def on_feature_change(tab, feature_col, epoch_idx, session_id):
    # Same reasoning as on_epoch_change — feature-trend-graph is Explorer-
    # tab-only, but epoch-index-store can be nudged from the Feature Table.
    if tab != "tab-explorer" or not session_id or not feature_col:
        return dash.no_update
    data = load_session(session_id)
    if data is None:
        return go.Figure()
    return build_feature_trend_figure(data, feature_col, epoch_idx=epoch_idx)


# Clicking a row in the Feature Table jumps the shared epoch-index-store.
# IMPORTANT: this callback must NOT target epoch-slider directly — that
# component only exists in the DOM while the Signal Explorer tab is
# rendered, and this click happens from the Feature Table tab, where
# epoch-slider has been removed from the page. Setting a real value (not
# dash.no_update) on a component id that isn't currently in the layout is
# exactly what raises "A nonexistent object was used in an `Output`...".
# epoch-slider re-reads its initial value from epoch-index-store (via the
# State in render_explorer_tab) the moment the user switches to that tab,
# so a single store update is sufficient and safe from any tab.
@app.callback(
    Output("epoch-index-store", "data", allow_duplicate=True),
    Input("features-table", "active_cell"),
    State("features-table", "data"),
    prevent_initial_call=True,
)
def on_feature_row_click(active_cell, table_data):
    if not active_cell or not table_data:
        return dash.no_update
    row = table_data[active_cell["row"]]
    return int(row.get("epoch_idx", 0))


@app.callback(
    Output("raw-json-summary", "children"),
    Input("raw-json-upload", "contents"),
    State("raw-json-upload", "filename"),
    prevent_initial_call=True,
)
def on_raw_upload(contents, filename):
    if contents is None:
        return None
    try:
        _, b64data = contents.split(",", 1)
        payload = json.loads(base64.b64decode(b64data))
    except Exception as exc:
        return status_pill(f"Could not parse '{filename}' as JSON: {exc}", "bad")
    summary = summarize_raw_device_json(payload)
    return html.Div([
        html.Div(f"File: {filename}", className="epoch-detail__time"),
        kv_grid(list(summary.items())),
    ])


if __name__ == "__main__":
    print(f"Test_dataset root : {os.path.abspath(TEST_DATASET_ROOT)}")
    print(f"Results root      : {os.path.abspath(RESULTS_ROOT)}")
    registry = discover_sessions()
    print(f"Found {len(registry)} Admission ID(s):")
    for sid, entry in registry.items():
        print(f"  {STATUS_ICON[entry['status']]}  {sid:<22s} "
             f"[{STATUS_LABEL[entry['status']]}]"
             f"{'  -> ' + entry['out_dir'] if entry['out_dir'] else ''}")
    if not registry:
        print("  (none — check DASHBOARD_TEST_DATASET_ROOT / DASHBOARD_RESULTS_ROOT "
             "and that you're running this from your project root)")
    app.run(debug=True, port=8050)