"""
=============================================================================
phase4b_ml_classification/p4b_ml_classify.py
Random Forest sleep-stage classifier — ML alternative to Phase 4's
rule-based scorer, evaluated the same way (vs Phase 5 PSG labels).

Data flow (matches your existing disk-based architecture)
───────────────────────────────────────────────────────────
  Reads   : phase3_hrv_features.csv  (per subject, from disk, via p4_load_phase3)
            psg_hypnogram.npy        (per subject, from disk, Phase 5 output)
  Writes  : phase4b_ecg_hypnogram_rf.npy   → each subject's output_dir
            phase4b_rf_model.pkl           → results/<dataset>/consolidated/
            phase4b_feature_importance.csv → results/<dataset>/consolidated/
  SQLite  : adds phase4b_done / phase4b_at / phase4b_output /
            phase4b_overall_acc / phase4b_cohen_kappa columns to
            processing_status (auto-added if missing, same pattern as
            create_database.py's _add_column_if_missing)

Why pooled + subject-level CV (not per-subject like Phases 1-6)
─────────────────────────────────────────────────────────────
RF needs to learn across subjects. Splitting by EPOCH (not subject) would
leak the same subject's physiology into both train and test and inflate
accuracy — GroupKFold groups by subject_id so entire subjects are held out.

Run AFTER Phase 3 and Phase 5 are complete for all subjects you want to
train/evaluate on. Independent of Phase 4 (rule-based) — run in either order.
=============================================================================
"""

import os
import sys
import json
import logging
import pickle
import gc

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.db_manager import get_all_records, get_connection, DB_PATH
from phase4_sleep_classification.p4_load_phase3 import load_features
from phase4_sleep_classification.p4_classify import smooth_hypnogram
from phase0_dataset_management.p0_config import make_subject_dirs

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# Columns to drop before feeding into the model — administrative, not physiological
_NON_FEATURE_COLS = {
    "epoch_idx", "is_bad", "n_rr", "detection_method", "sqi_flag",
    "__wake_hr_p75",
}


# ─────────────────────────────────────────────────────────────────────────────
#  SQLite: ensure phase4b columns exist
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_phase4b_columns(db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        existing = {row[1] for row in
                    conn.execute("PRAGMA table_info(processing_status)")}
        additions = {
            "phase4b_done":         "INTEGER DEFAULT 0",
            "phase4b_at":           "TEXT",
            "phase4b_output":       "TEXT",
            "phase4b_overall_acc":  "REAL",
            "phase4b_cohen_kappa":  "REAL",
        }
        for col, coltype in additions.items():
            if col not in existing:
                conn.execute(
                    f"ALTER TABLE processing_status ADD COLUMN {col} {coltype}"
                )
        conn.commit()


def _mark_phase4b(record_id: int, output_dir: str, acc: float, kappa: float,
                   db_path: str = DB_PATH) -> None:
    from datetime import datetime
    with get_connection(db_path) as conn:
        conn.execute(
            """UPDATE processing_status
               SET phase4b_done=1, phase4b_at=?, phase4b_output=?,
                   phase4b_overall_acc=?, phase4b_cohen_kappa=?, updated_at=?
               WHERE record_id=?""",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), output_dir,
             float(acc), float(kappa),
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"), record_id),
        )
        conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  Load + pool every subject's Phase 3 features + Phase 5 PSG labels
# ─────────────────────────────────────────────────────────────────────────────

def _load_subject_data(row: dict, config: dict, logger: logging.Logger):
    """
    Returns (df_features, psg_labels, output_dir) for one subject,
    or None if either Phase 3 or Phase 5 output is missing.
    """
    output_dir = make_subject_dirs(
        row["record_name"], config.get("results_dir", "results"),
        config["dataset"],
    )
    cfg = dict(config)
    cfg["record_name"] = row["record_name"]
    cfg["output_dir"]  = output_dir

    feat_path = os.path.join(output_dir, "phase3_hrv_features.csv")
    psg_path  = os.path.join(output_dir, "psg_hypnogram.npy")   # adjust if your
                                                                  # p5_save uses a
                                                                  # different filename
    if not (os.path.exists(feat_path) and os.path.exists(psg_path)):
        logger.warning(f"Skipping {row['record_name']} — missing Phase3/Phase5 output")
        return None

    df  = load_features(cfg, logger)
    psg = np.load(psg_path)

    n = min(len(df), len(psg))
    df, psg = df.iloc[:n].reset_index(drop=True), psg[:n]
    df["subject_id"] = row["record_name"]
    df["psg_label"]  = psg
    return df, output_dir


def build_pooled_dataset(config: dict, logger: logging.Logger) -> pd.DataFrame:
    records = get_all_records(config["dataset"])
    frames  = []
    for row in records:
        result = _load_subject_data(row, config, logger)
        if result is not None:
            df, _ = result
            frames.append(df)
    if not frames:
        raise RuntimeError("No subjects with both Phase3 + Phase5 output found.")
    pooled = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Pooled dataset: {len(pooled)} epochs from "
        f"{pooled['subject_id'].nunique()} subjects"
    )
    return pooled


def _feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in _NON_FEATURE_COLS | {"subject_id", "psg_label"}
            and pd.api.types.is_numeric_dtype(df[c])]


# ─────────────────────────────────────────────────────────────────────────────
#  Train + evaluate (subject-level GroupKFold — no epoch leakage)
# ─────────────────────────────────────────────────────────────────────────────

def train_rf(pooled: pd.DataFrame, config: dict, logger: logging.Logger,
             n_splits: int = None, random_state: int = 42,
             n_estimators: int = 300, max_depth: int = 20, n_jobs: int = 4):
    """
    Trains RF with subject-grouped CV, returns (model, imputer, feature_cols,
    oof_predictions_df, overall_metrics).

    n_splits=None (default) → true Leave-One-Subject-Out (LOSO).

    max_depth=20 (was None/unbounded) and n_jobs=4 (was -1/all cores) —
    unbounded trees across 154 repeated folds were accumulating memory
    until a MemoryError hit mid-run. Bounding depth also usually helps
    generalization (less overfitting per fold), not just memory.

    CRASH-SAFE CHECKPOINTING: progress (oof_pred so far + next fold index)
    is saved to disk after EVERY fold. If the process crashes or is killed,
    simply re-run the same call and it will resume from the last completed
    fold instead of starting over. Checkpoint is deleted automatically once
    all folds finish successfully.
    """
    consolidated_dir = os.path.join(config.get("results_dir", "results"),
                                     config["dataset"], "consolidated")
    os.makedirs(consolidated_dir, exist_ok=True)
    ckpt_path = os.path.join(consolidated_dir, "phase4b_checkpoint.npz")

    feat_cols = _feature_columns(pooled)
    X_all = pooled[feat_cols].values.astype(float)
    X_all[np.isinf(X_all)] = np.nan

    # Drop any feature that is (almost) entirely missing across ALL subjects —
    # SimpleImputer can't compute a median with zero observed values, and was
    # silently + inconsistently dropping such columns fold-by-fold (that's
    # the recurring "Skipping features without any observed values" warning).
    finite_frac = 1.0 - (np.isnan(X_all).mean(axis=0))
    dead_mask   = finite_frac < 0.01   # <1% observed values = effectively dead
    if dead_mask.any():
        dead_feats = [f for f, d in zip(feat_cols, dead_mask) if d]
        logger.warning(
            f"Dropping {len(dead_feats)} feature(s) with <1% observed values "
            f"across the whole pooled dataset (likely a Phase 3 computation "
            f"issue for these): {dead_feats}"
        )
        feat_cols = [f for f, d in zip(feat_cols, dead_mask) if not d]
        X_all = X_all[:, ~dead_mask]

    X = X_all
    y = pooled["psg_label"].values
    groups = pooled["subject_id"].values

    n_subjects = pooled["subject_id"].nunique()
    n_splits   = n_subjects if n_splits is None else min(n_splits, n_subjects)
    is_loso    = (n_splits == n_subjects)
    gkf = GroupKFold(n_splits=n_splits)

    # ── Resume from checkpoint if one exists and matches this run's shape ──
    start_fold = 0
    oof_pred = np.full(len(pooled), -1, dtype=int)
    if os.path.exists(ckpt_path):
        try:
            ckpt = np.load(ckpt_path)
            if int(ckpt["n_epochs"]) == len(pooled) and int(ckpt["n_splits"]) == n_splits:
                oof_pred  = ckpt["oof_pred"]
                start_fold = int(ckpt["next_fold"])
                logger.info(
                    f"RESUMING from checkpoint → {start_fold} fold(s) already "
                    f"done, continuing from fold {start_fold + 1}/{n_splits}"
                )
            else:
                logger.warning(
                    "Checkpoint found but doesn't match this run's shape "
                    "(different n_epochs/n_splits) — ignoring, starting fresh."
                )
        except Exception as exc:
            logger.warning(f"Could not read checkpoint ({exc}) — starting fresh.")

    logger.info(
        f"Training RF — {n_subjects} subjects, "
        f"{'LOSO (' + str(n_splits) + ' folds)' if is_loso else str(n_splits) + '-fold GroupKFold'}, "
        f"{len(feat_cols)} features, {len(pooled)} epochs, "
        f"max_depth={max_depth}, n_jobs={n_jobs}"
    )

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        if fold < start_fold:
            continue   # already done in a previous (crashed) run

        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X[tr_idx])
        X_te = imputer.transform(X[te_idx])

        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=n_jobs,
            random_state=random_state,
        )
        clf.fit(X_tr, y[tr_idx])
        oof_pred[te_idx] = clf.predict(X_te)

        fold_acc = accuracy_score(y[te_idx], oof_pred[te_idx])
        held_out = sorted(set(groups[te_idx]))
        label = "held-out subject" if is_loso else "held-out subjects"
        logger.info(f"  Fold {fold+1}/{n_splits}: acc={fold_acc:.3f} "
                    f"({label}: {held_out})")

        # Explicit cleanup — release this fold's model/data before the next
        del imputer, clf, X_tr, X_te
        gc.collect()

        # Checkpoint after every fold — a crash after this point loses at
        # most the NEXT fold's work, not everything done so far
        np.savez(ckpt_path, oof_pred=oof_pred, next_fold=fold + 1,
                  n_splits=n_splits, n_epochs=len(pooled))

    # All folds done — checkpoint no longer needed
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    valid = oof_pred >= 0
    overall_acc   = accuracy_score(y[valid], oof_pred[valid])
    overall_kappa = cohen_kappa_score(y[valid], oof_pred[valid])
    logger.info(f"Overall OOF accuracy={overall_acc:.3f}  kappa={overall_kappa:.3f}")
    logger.info("\n" + classification_report(
        y[valid], oof_pred[valid],
        target_names=[STAGE_NAMES[k] for k in sorted(set(y[valid]))],
        zero_division=0,
    ))

    # Final model trained on ALL data (for production predictions)
    # X here is the same inf-cleaned array built above — reused, not re-pulled
    imputer_full = SimpleImputer(strategy="median")
    X_full = imputer_full.fit_transform(X)
    final_clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=n_jobs, random_state=random_state,
    )
    final_clf.fit(X_full, y)

    importance_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": final_clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    pooled = pooled.copy()
    pooled["rf_pred"] = oof_pred

    metrics = {"overall_acc": overall_acc, "overall_kappa": overall_kappa,
               "n_subjects": n_subjects, "n_epochs": len(pooled),
               "n_features": len(feat_cols)}
    return final_clf, imputer_full, feat_cols, pooled, importance_df, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Save per-subject predictions + consolidated artifacts + SQLite
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(pooled_with_preds: pd.DataFrame, model, imputer, feat_cols,
                  importance_df: pd.DataFrame, metrics: dict,
                  config: dict, logger: logging.Logger, db_path: str = DB_PATH):
    consolidated_dir = os.path.join(config.get("results_dir", "results"),
                                     config["dataset"], "consolidated")
    os.makedirs(consolidated_dir, exist_ok=True)

    with open(os.path.join(consolidated_dir, "phase4b_rf_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "imputer": imputer,
                     "feature_cols": feat_cols}, f)
    importance_df.to_csv(
        os.path.join(consolidated_dir, "phase4b_feature_importance.csv"),
        index=False,
    )
    with open(os.path.join(consolidated_dir, "phase4b_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved model + feature importance → {consolidated_dir}")

    _ensure_phase4b_columns(db_path)
    records = {r["record_name"]: r for r in get_all_records(config["dataset"])}

    for subj, grp in pooled_with_preds.groupby("subject_id"):
        output_dir = make_subject_dirs(
            subj, config.get("results_dir", "results"), config["dataset"]
        )
        preds_raw = grp["rf_pred"].values
        preds_smooth = smooth_hypnogram(preds_raw, kernel_size=5,
                                         min_run_epochs=2, logger=None)

        np.save(os.path.join(output_dir, "phase4b_ecg_hypnogram_rf_raw.npy"), preds_raw)
        np.save(os.path.join(output_dir, "phase4b_ecg_hypnogram_rf.npy"), preds_smooth)

        subj_acc_raw    = accuracy_score(grp["psg_label"], preds_raw)
        subj_kappa_raw  = cohen_kappa_score(grp["psg_label"], preds_raw)
        subj_acc        = accuracy_score(grp["psg_label"], preds_smooth)
        subj_kappa      = cohen_kappa_score(grp["psg_label"], preds_smooth)

        row = records.get(subj)
        if row:
            _mark_phase4b(row["record_id"], output_dir, subj_acc, subj_kappa, db_path)
        logger.info(
            f"  {subj}: raw acc={subj_acc_raw:.3f} kappa={subj_kappa_raw:.3f}  →  "
            f"smoothed acc={subj_acc:.3f} kappa={subj_kappa:.3f} "
            f"→ {output_dir}/phase4b_ecg_hypnogram_rf.npy"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_phase4b(config: dict, logger: logging.Logger, db_path: str = DB_PATH,
                 n_splits: int = None, n_estimators: int = 300,
                 max_depth: int = 20, n_jobs: int = 4):
    pooled = build_pooled_dataset(config, logger)
    model, imputer, feat_cols, pooled_with_preds, importance_df, metrics = \
        train_rf(pooled, config, logger, n_splits=n_splits,
                 n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs)
    save_outputs(pooled_with_preds, model, imputer, feat_cols,
                 importance_df, metrics, config, logger, db_path)
    logger.info(f"✓ Phase 4b (RF) complete — dataset={config['dataset']} | {metrics}")
    return metrics


if __name__ == "__main__":
    from phase0_dataset_management.p0_config  import CONFIG
    from phase0_dataset_management.p0_logging import setup_logger

    logger = setup_logger("phase4b", log_dir="logs", log_file="phase4b.log")

    # This call is CRASH-SAFE: if it's interrupted (memory error, closed
    # terminal, power loss, etc.), just run this exact same line again —
    # it will detect the checkpoint and resume from the next fold instead
    # of starting over. Checkpoint lives at
    # results/<dataset>/consolidated/phase4b_checkpoint.npz and is deleted
    # automatically once all folds complete successfully.
    run_phase4b(CONFIG, logger, n_splits=None, n_estimators=300,
                max_depth=20, n_jobs=4)