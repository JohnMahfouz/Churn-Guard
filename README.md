# churn-guard

[![Retrain Gate](https://github.com/JohnMahfouz/Churn-Guard/actions/workflows/retrain-gate.yml/badge.svg)](https://github.com/JohnMahfouz/Churn-Guard/actions/workflows/retrain-gate.yml)

A promotion-gated churn prediction pipeline on the IBM Telco Customer Churn dataset: leakage-audited EDA, feature engineering with a leave-one-out region-churn-rate encoding, a LightGBM baseline tracked in MLflow, a promotion gate that blocks non-improving retrains, and Evidently-based drift monitoring.

## Pipeline

| Script | Purpose |
| --- | --- |
| `EDA.py` | Drops target-leakage columns, flags CLTV for review |
| `feature_engineering.py` | Builds `telco_features.parquet`: leak-free `region_churn_rate`, cleaned `Total Charges`, categorical dtypes |
| `train.py` | Baseline LightGBM model, logged to the MLflow `churn-guard` experiment |
| `retrain.py` | Retrains and promotes only if F1 beats the current `stage: production` run by 2+ points; otherwise rejects (see [Retrain Gate](#retrain-gate) below) |
| `drift_check.py` | Evidently feature/prediction/performance drift reports (requires the separate `.venv-drift` Python 3.11 environment -- see note below) |

## Retrain Gate

`.github/workflows/retrain-gate.yml` runs `retrain.py` on every push to `main` and on demand (`workflow_dispatch`). The job's pass/fail state *is* `retrain.py`'s exit code: 0 (promoted) passes, 1 (rejected) fails -- a model that doesn't clearly improve cannot become production without the Actions run visibly failing. Each run uploads `retrain_summary.txt` and the MLflow run artifacts (including the feature importance plot and model) so the promotion/rejection reasoning is visible from the Actions tab.

### What's committed vs. gitignored, and why

CI runs in a clean checkout with no access to this machine's files, so anything `retrain.py` needs to read must either be committed or reconstructed in the workflow. See `.gitignore` for the full list; the key calls:

- **`telco_features.parquet` is committed.** `retrain.py` loads it directly and the workflow only runs that script (not `feature_engineering.py`), so without it the job has nothing to train on. It's a small (~140KB), fully engineered sample of a public IBM dataset -- there's no privacy/scale reason to keep it out of a portfolio repo.
- **`mlflow.db` and `mlruns/` are gitignored, not committed.** A SQLite file has no meaningful diff or merge in git, and committing it would freeze the "current production model" at whatever it was when you committed. Instead, the workflow persists them across CI runs with `actions/cache`, so promotion history builds up naturally between workflow runs without touching version control.
- **`Telco_customer_churn.xlsx` (raw source) is committed** for anyone who wants to re-run `feature_engineering.py` from scratch; `archive.zip` (a redundant copy of the same public dataset) is gitignored.
- **`reports/`, `feature_importance.png`, `.venv-drift/`** are all regenerated outputs or a local-only environment -- gitignored, not source.

### Note on `drift_check.py`

Evidently is incompatible with Python 3.14 (a Pydantic v1 issue affecting every Evidently release as of this writing). `drift_check.py` runs in a separate Python 3.11 virtual environment (`.venv-drift/`, not committed -- see `requirements-drift.txt` to recreate it) and is not part of the CI workflow above, which targets the main Python 3.14 environment.
