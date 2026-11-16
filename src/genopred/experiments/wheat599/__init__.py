"""Wheat599 experiment orchestration (WP-C).

Contract (method-stable names; phase expressed via run config/results):
- quantile_tabm_dev: run/_init_fit_worker/_fit_worker (from neural/quantile_tabm.py).
- quantile_tabm_outer: from neural/quantile_tabm_final.py (freeze/validate/fit_one/summarize/run).
- shape_quantile_outer: from neural/shape_quantile_final.py.
- independent_tabm_outer: from neural/independent_final.py (ensure_frozen_manifest etc.).
- genomic_tabm_dev: from neural/genomic_tabm_run.py (model_source_sha etc.).
- ragp_dev: from neural/ragp_run.py.
- background_ridge: nested background-ridge/RR experiment (WP-A, from scripts/run_background_nested.py).
"""