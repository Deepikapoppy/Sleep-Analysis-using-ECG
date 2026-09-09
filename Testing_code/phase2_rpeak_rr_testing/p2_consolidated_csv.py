"""
=============================================================================
phase2_rpeak_rr_testing/p2_consolidated_csv.py
Build a consolidated CSV across ALL device test sessions.

Run this AFTER p2_pipeline.py has completed for all sessions.

Output files
────────────
results_test/consolidated/
    phase2_consolidated_all_records.csv   ← one row per session
    phase2_method_pattern_summary.csv     ← which method dominates where
    phase2_consolidated_report.txt        ← human-readable summary

Consolidated CSV columns
────────────────────────
record_name, dataset,
total_epochs, valid_epochs, bad_epochs,
mean_hr, mean_quality, mean_rr_ms,

# Method win counts (how many epochs each method won)
neurokit_wins, elgendi_wins, scipy_wins, failed_wins,

# Method win percentages
neurokit_pct, elgendi_pct, scipy_pct,

# Dominant method (most wins in this record)
dominant_method, dominant_method_pct,

# Per-method average peak counts across all epochs
neurokit_avg_peaks, elgendi_avg_peaks, scipy_avg_peaks,

# Quality stats
high_quality_epochs,    (quality >= 0.9)
medium_quality_epochs,  (0.7 <= quality < 0.9)
low_quality_epochs,     (quality < 0.7)
mean_n_artifacts,

# Sleep-stage transition indicator
hr_min, hr_max, hr_std,  (high std = more sleep stage transitions)

phase1_dwt_wavelet, phase1_dwt_level, phase1_polarity_inverted
=============================================================================
"""

import os
import sys
import json
import ast
import logging
import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
#  Parse one record's epoch summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def _parse_method_counts(val) -> dict:
    """Safely parse the all_method_counts column (stored as dict string)."""
    if pd.isna(val) or val == "":
        return {}
    if isinstance(val, dict):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return {}


def _summarise_one_record(record_name: str,
                           dataset: str,
                           output_dir: str) -> Optional[dict]:
    """
    Read phase2_rr_epoch_summary.csv + phase2_metrics.json for one record.
    Returns a summary dict or None if files not found.
    """
    epoch_csv  = os.path.join(output_dir, "phase2_rr_epoch_summary.csv")
    metric_json = os.path.join(output_dir, "phase2_metrics.json")

    if not os.path.exists(epoch_csv):
        return None

    df = pd.read_csv(epoch_csv)

    # Load meta
    meta = {}
    if os.path.exists(metric_json):
        with open(metric_json) as f:
            meta = json.load(f)

    total_epochs = len(df)
    valid_mask   = df["n_rr"] > 0
    valid_epochs = int(valid_mask.sum())
    bad_epochs   = int(df["is_bad"].sum())

    # ── HR stats ──────────────────────────────────────────────────────────────
    hr_vals = df.loc[valid_mask, "mean_hr"].dropna()
    mean_hr = round(float(hr_vals.mean()), 2) if len(hr_vals) else float("nan")
    hr_min  = round(float(hr_vals.min()),  2) if len(hr_vals) else float("nan")
    hr_max  = round(float(hr_vals.max()),  2) if len(hr_vals) else float("nan")
    hr_std  = round(float(hr_vals.std()),  2) if len(hr_vals) else float("nan")

    # ── RR / quality stats ────────────────────────────────────────────────────
    mean_rr      = round(float(df.loc[valid_mask, "mean_rr"].mean()), 2) \
                   if valid_mask.sum() else float("nan")
    mean_quality = round(float(df["quality"].mean()), 3)
    mean_artifacts = round(float(df["n_artifacts"].mean()), 2)

    hq = int((df["quality"] >= 0.9).sum())
    mq = int(((df["quality"] >= 0.7) & (df["quality"] < 0.9)).sum())
    lq = int((df["quality"] < 0.7).sum())

    # ── Method win counts ─────────────────────────────────────────────────────
    method_wins = {
    "neurokit"        : 0,
    "elgendi2010"     : 0,
    "scipy_prominence": 0,
    "failed"          : 0,
    }
    for m in df["method"]:
        key = str(m).strip()
        if key in method_wins:
            method_wins[key] += 1
        else:
            method_wins["failed"] += 1

    # ── Per-method average peak counts ────────────────────────────────────────
    avg_peaks = {m: 0.0 for m in
                 ["neurokit", "elgendi2010", "scipy_prominence"]}
    if "all_method_counts" in df.columns:
        parsed = df["all_method_counts"].apply(_parse_method_counts)
        for m in avg_peaks:
            vals = [d.get(m, 0) for d in parsed if isinstance(d, dict)]
            avg_peaks[m] = round(float(np.mean(vals)), 2) if vals else 0.0

    # ── Dominant method ───────────────────────────────────────────────────────
    valid_methods = {k: v for k, v in method_wins.items() if k != "failed"}
    dominant      = max(valid_methods, key=valid_methods.get)
    dominant_pct  = round(100 * method_wins[dominant] / max(total_epochs, 1), 1)

    # ── Method percentages ────────────────────────────────────────────────────
    def _pct(m): return round(100 * method_wins[m] / max(total_epochs, 1), 1)

    row = {
        "record_name"          : record_name,
        "dataset"              : dataset,
        "total_epochs"         : total_epochs,
        "valid_epochs"         : valid_epochs,
        "bad_epochs"           : bad_epochs,
        "mean_hr"              : mean_hr,
        "hr_min"               : hr_min,
        "hr_max"               : hr_max,
        "hr_std"               : hr_std,
        "mean_rr_ms"           : mean_rr,
        "mean_quality"         : mean_quality,
        "mean_n_artifacts"     : mean_artifacts,

        # Method wins
        "neurokit_wins"        : method_wins["neurokit"],
        "elgendi_wins"         : method_wins["elgendi2010"],
        "scipy_wins"           : method_wins["scipy_prominence"],
        "failed_wins"          : method_wins["failed"],

        # Method percentages
        "neurokit_pct"         : _pct("neurokit"),
        "elgendi_pct"          : _pct("elgendi2010"),
        "scipy_pct"            : _pct("scipy_prominence"),

        # Dominant method
        "dominant_method"      : dominant,
        "dominant_method_pct"  : dominant_pct,

        # Average peaks per method
        "neurokit_avg_peaks"   : avg_peaks["neurokit"],
        "elgendi_avg_peaks"    : avg_peaks["elgendi2010"],
        "scipy_avg_peaks"      : avg_peaks["scipy_prominence"],

        # Quality distribution
        "high_quality_epochs"  : hq,
        "medium_quality_epochs": mq,
        "low_quality_epochs"   : lq,

        # Phase 1 metadata
        "phase1_dwt_wavelet"       : meta.get("phase1_dwt_wavelet",    "db4"),
        "phase1_dwt_level"         : meta.get("phase1_dwt_level",       5),
        "phase1_polarity_inverted" : meta.get("phase1_polarity_inverted", False),
    }
    return row


# ─────────────────────────────────────────────────────────────────────────────
#  Build consolidated CSV across all records
# ─────────────────────────────────────────────────────────────────────────────

def build_consolidated_csv(config: dict,
                            logger: logging.Logger,
                            output_root: str = "results_test",
                            db_path: str = None) -> pd.DataFrame:
    """
    Scan all per-record output directories, build consolidated summary.

    Parameters
    ----------
    config      : CONFIG dict (needs 'dataset', 'results_dir')
    logger      : logger
    output_root : root of results_test/ folder
    db_path     : isolated test DB path (e.g. "test_pipeline.db") — pass
                  explicitly to keep this reading from test data only,
                  never your production DB

    Returns
    -------
    df_consolidated : pd.DataFrame — one row per record
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Database.db_manager import get_all_records, DB_PATH

    db_path = db_path or DB_PATH
    dataset = config.get("dataset", "device")
    db_records = get_all_records(dataset, db_path)

    if db_records:
        session_ids = [r["record_name"] for r in db_records]
    else:
        # DB registration is optional throughout this testing suite — fall
        # back to scanning the device folder directly, same pattern used
        # by p1_pipeline.py / p2_pipeline.py's __main__ blocks.
        logger.warning(
            "No sessions found in SQLite — falling back to a disk scan "
            "(registration is optional for device test sessions)."
        )
        from phase0_dataset_management_testing import scan_datasets
        session_ids = scan_datasets(config, logger)

    if not session_ids:
        logger.warning("No device sessions found (SQLite empty and disk scan empty).")
        return pd.DataFrame()

    logger.info(f"Building consolidated CSV for {len(session_ids)} {dataset} session(s)...")

    rows = []
    missing = []
    for rec_name in session_ids:
        out_dir   = os.path.join(output_root, dataset, "plots",
                                  "per_subject", rec_name)
        row = _summarise_one_record(rec_name, dataset, out_dir)
        if row:
            rows.append(row)
            logger.info(f"  ✓ {rec_name}  dominant={row['dominant_method']}  "
                        f"mean_hr={row['mean_hr']}")
        else:
            missing.append(rec_name)
            logger.warning(f"  ✗ {rec_name}  — no phase2 CSV found (skipped)")

    if missing:
        logger.warning(f"Missing Phase-2 outputs for {len(missing)} records: "
                       f"{missing[:5]}{'...' if len(missing)>5 else ''}")

    df = pd.DataFrame(rows)

    # ── Save consolidated CSV ─────────────────────────────────────────────────
    out_dir = os.path.join(output_root, "consolidated")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "phase2_consolidated_all_records.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Consolidated CSV saved → {csv_path}")

    # ── Method pattern summary ────────────────────────────────────────────────
    if not df.empty:
        _save_method_pattern_summary(df, out_dir, logger)
        _save_text_report(df, out_dir, dataset, logger)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Method pattern summary
# ─────────────────────────────────────────────────────────────────────────────

def _save_method_pattern_summary(df: pd.DataFrame,
                                  out_dir: str,
                                  logger: logging.Logger) -> None:
    """
    Per-method statistics across all records:
    How many records chose each method as dominant?
    What is the average win% per method?
    """
    methods = ["neurokit", "elgendi2010", "scipy_prominence"]

    rows = []
    for m in methods:
        wins_col = f"{m.replace('2010','').replace('2017','').replace('_prominence','').replace('modified_','')}_wins"
        pct_col  = wins_col.replace("wins", "pct")

        # Map to actual column names
        col_map = {
            "neurokit"         : ("neurokit_wins",  "neurokit_pct"),
            "elgendi2010"      : ("elgendi_wins",   "elgendi_pct"),
            "scipy_prominence" : ("scipy_wins",     "scipy_pct"),
        }
        wc, pc = col_map[m]

        n_dominant = int((df["dominant_method"] == m).sum())
        rows.append({
            "method"                : m,
            "n_records_dominant"    : n_dominant,
            "pct_records_dominant"  : round(100 * n_dominant / max(len(df), 1), 1),
            "mean_win_pct"          : round(float(df[pc].mean()), 1),
            "min_win_pct"           : round(float(df[pc].min()), 1),
            "max_win_pct"           : round(float(df[pc].max()), 1),
            "total_epoch_wins"      : int(df[wc].sum()),
        })

    pat_df   = pd.DataFrame(rows).sort_values("total_epoch_wins", ascending=False)
    pat_path = os.path.join(out_dir, "phase2_method_pattern_summary.csv")
    pat_df.to_csv(pat_path, index=False)
    logger.info(f"Method pattern summary saved → {pat_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Human-readable text report
# ─────────────────────────────────────────────────────────────────────────────

def _save_text_report(df: pd.DataFrame, out_dir: str,
                       dataset: str, logger: logging.Logger) -> None:
    lines = []
    lines.append("=" * 65)
    lines.append(f"  PHASE 2 CONSOLIDATED REPORT  |  Dataset: {dataset.upper()}")
    lines.append("=" * 65)
    lines.append(f"  Total records processed : {len(df)}")
    lines.append(f"  Total epochs            : {int(df['total_epochs'].sum())}")
    lines.append(f"  Valid epochs            : {int(df['valid_epochs'].sum())}")
    lines.append(f"  Mean HR (all records)   : {df['mean_hr'].mean():.1f} bpm")
    lines.append(f"  Mean quality score      : {df['mean_quality'].mean():.3f}")
    lines.append("")
    lines.append("  Method Dominance (which method won most epochs per record):")
    lines.append("  " + "-" * 55)
    dom = df["dominant_method"].value_counts()
    for m, cnt in dom.items():
        pct = round(100 * cnt / len(df), 1)
        lines.append(f"  {m:<28} {cnt:>4} records  ({pct}%)")
    lines.append("")
    lines.append("  Total Epoch Wins across all records:")
    lines.append("  " + "-" * 55)
    for col, label in [
        ("neurokit_wins",  "neurokit"),
        ("elgendi_wins",   "elgendi2010"),
        ("scipy_wins",     "scipy_prominence"),
        ("failed_wins",    "failed"),
    ]:
        total = int(df[col].sum())
        total_ep = int(df["total_epochs"].sum())
        pct = round(100 * total / max(total_ep, 1), 1)
        lines.append(f"  {label:<28} {total:>6} epochs  ({pct}%)")
    lines.append("")
    lines.append("  Top 5 records by highest mean HR:")
    top_hr = df.nlargest(5, "mean_hr")[["record_name", "mean_hr", "dominant_method"]]
    for _, r in top_hr.iterrows():
        lines.append(f"    {r['record_name']:<12} HR={r['mean_hr']:.1f}  method={r['dominant_method']}")
    lines.append("")
    lines.append("  Top 5 records by lowest mean quality:")
    low_q = df.nsmallest(5, "mean_quality")[["record_name", "mean_quality", "mean_n_artifacts"]]
    for _, r in low_q.iterrows():
        lines.append(f"    {r['record_name']:<12} quality={r['mean_quality']:.3f}  artifacts/ep={r['mean_n_artifacts']:.1f}")
    lines.append("=" * 65)

    rpt_path = os.path.join(out_dir, "phase2_consolidated_report.txt")
    with open(rpt_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Text report saved → {rpt_path}")
    # Also print to console
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from phase0_dataset_management_testing import CONFIG
    from .p2_logging import setup_logger

    TEST_DB_PATH = "test_pipeline.db"

    logger = setup_logger("consolidated_test", log_dir="logs",
                          log_file="phase2_consolidated_test.log")
    df = build_consolidated_csv(CONFIG, logger,
                                 output_root=CONFIG.get("results_dir", "results_test"),
                                 db_path=TEST_DB_PATH)

    print(f"\nConsolidated CSV: {len(df)} records")
    print(df[["record_name", "dominant_method", "mean_hr",
              "mean_quality", "valid_epochs"]].to_string(index=False))