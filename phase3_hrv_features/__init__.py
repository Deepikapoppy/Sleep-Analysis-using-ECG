"""
=============================================================================
phase3_hrv_features  —  Phase 3 HRV Feature Extraction Package

Module layout
─────────────
  p3_logging.py          Logging setup
  p3_load_phase2.py      Load Phase 2 RR results + Phase 1 meta/SQI
  p3_sqi_gate.py         SQI Gate (4 features) — always applied first
  p3_tier1_features.py   T1 Primary Screamer (18 features)
  p3_tier2_features.py   T2 Confirmatory (19 features)
  p3_tier3_features.py   T3 Resolver (29 features)
  p3_postprocess.py      Derived features, rolling median smooth, z-scores
  p3_plots.py            Diagnostic plots per tier
  p3_save.py             Save CSV, meta JSON, SQLite update
  p3_pipeline.py         Master orchestrator
=============================================================================
"""
