# churn-guard

[![Retrain Gate](https://github.com/JohnMahfouz/Churn-Guard/actions/workflows/retrain-gate.yml/badge.svg)](https://github.com/JohnMahfouz/Churn-Guard/actions/workflows/retrain-gate.yml)

A promotion-gated churn prediction pipeline on the IBM Telco Customer Churn dataset: leakage-audited EDA, feature engineering with a leave-one-out region-churn-rate encoding, a LightGBM baseline tracked in MLflow, a promotion gate that blocks non-improving retrains, and Evidently-based drift monitoring.

## Pipeline

| Script | Purpose |
| --- | --- |
| `src/eda.py` | Drops target-leakage columns, flags CLTV for review |
| `src/feature_engineering.py` | Builds `data/telco_features.parquet`: leak-free `region_churn_rate`, cleaned `Total Charges`, categorical dtypes |
| `src/train.py` | Baseline LightGBM model, logged to the MLflow `churn-guard` experiment |
| `src/retrain.py` | Retrains and promotes only if F1 beats the current `stage: production` run by 2+ points; otherwise rejects (see [Retrain Gate](#retrain-gate) below) |
| `src/drift_check.py` | Evidently feature/prediction/performance drift reports (requires the separate `.venv-drift` Python 3.11 environment -- see note below) |

## Retrain Gate

`.github/workflows/retrain-gate.yml` runs `src/feature_engineering.py` then `src/retrain.py` on every push to `main` and on demand (`workflow_dispatch`). The job's pass/fail state *is* `retrain.py`'s exit code: 0 (promoted) passes, 1 (rejected) fails -- a model that doesn't clearly improve cannot become production without the Actions run visibly failing. Each run uploads `retrain_summary.txt` and the MLflow run artifacts (including the feature importance plot and model) so the promotion/rejection reasoning is visible from the Actions tab.

### What's committed vs. DVC-tracked vs. gitignored, and why

CI runs in a clean checkout with no access to this machine's files, so anything the workflow needs must either be committed, DVC-pullable from somewhere CI can reach, or reconstructed. See `.gitignore` for the full list; the key calls:

- **`data/telco_features.parquet` is DVC-tracked (`data/telco_features.parquet.dvc` is committed), not committed directly.** DVC's remote is a local folder on the maintainer's machine (`dvc remote add -d localremote <path>`) -- fine for local dataset versioning across pipeline changes, but a GitHub-hosted runner has no path to reach it, so **CI does not rely on DVC at all**. Instead, the workflow runs `src/feature_engineering.py` against the committed raw source to regenerate an identical parquet on every run (verified byte-identical to the DVC-tracked version). DVC and CI intentionally use two different paths to the same data here.
- **`mlflow.db` and `mlruns/` are gitignored, not committed.** A SQLite file has no meaningful diff or merge in git, and committing it would freeze the "current production model" at whatever it was when you committed. Instead, the workflow persists them across CI runs with `actions/cache`, so promotion history builds up naturally between workflow runs without touching version control.
- **`data/telco_customer_churn_raw.xlsx` (raw source) is committed** -- this is what CI actually builds `data/telco_features.parquet` from; `archive.zip` (a redundant copy of the same public dataset) has been removed entirely.
- **`reports/`, `feature_importance.png`, `.venv-drift/`, `chroma_db/`** are all regenerated outputs or a local-only environment -- gitignored, not source.

### Note on `drift_check.py`

Evidently is incompatible with Python 3.14 (a Pydantic v1 issue affecting every Evidently release as of this writing). `src/drift_check.py` runs in a separate Python 3.11 virtual environment (`.venv-drift/`, not committed -- see `requirements-drift.txt` to recreate it) and is not part of the CI workflow above, which targets the main Python 3.14 environment.
