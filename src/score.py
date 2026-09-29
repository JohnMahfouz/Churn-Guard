import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import mlflow.lightgbm
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "churn-guard"

# telco_features.parquet's schema, minus the target column -- see feature_engineering.py.
# Kept explicit here rather than derived from the model: LightGBM sanitizes feature
# names internally (spaces become underscores), so model.feature_name_ doesn't give
# back the original column names a caller's CSV actually uses.
FEATURE_COLUMNS = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months",
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charges", "Total Charges", "CLTV", "region_churn_rate",
]
CATEGORICAL_COLUMNS = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service",
    "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
]


def load_production_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' does not exist -- run train.py/retrain.py first.")

    prod_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'production'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not prod_runs:
        raise RuntimeError(
            "No run tagged stage='production' in the churn-guard experiment -- "
            "run retrain.py to establish a production model before starting this API."
        )
    run = prod_runs[0]
    model = mlflow.lightgbm.load_model(f"runs:/{run.info.run_id}/model")

    # pandas_categorical preserves the exact training-time category levels (and their
    # order) for each categorical column, in fit order -- matched positionally against
    # CATEGORICAL_COLUMNS below, so a batch missing some category values, or a batch
    # that's all one value, still gets encoded identically to how the model was trained.
    pandas_categorical = model.booster_.pandas_categorical
    return model, run, pandas_categorical


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, run, pandas_categorical = load_production_model()
    app.state.model = model
    app.state.run = run
    app.state.pandas_categorical = pandas_categorical
    yield


app = FastAPI(title="churn-guard batch scoring", lifespan=lifespan)


def _promoted_at(run) -> str:
    tagged = run.data.tags.get("promoted_at")
    if tagged is not None:
        return tagged
    # Runs promoted before the promoted_at tag existed: fall back to when the run
    # itself was logged, which is when promotion happened in the same retrain.py call.
    return datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).isoformat()


def _prepare_features(df: pd.DataFrame, pandas_categorical: list) -> pd.DataFrame:
    uploaded_cols = set(df.columns)
    expected_cols = set(FEATURE_COLUMNS)
    missing = sorted(expected_cols - uploaded_cols)
    unexpected = sorted(uploaded_cols - expected_cols)
    if missing or unexpected:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "CSV columns do not match the expected schema",
                "missing_columns": missing,
                "unexpected_columns": unexpected,
            },
        )

    df = df[FEATURE_COLUMNS].copy()
    for col, categories in zip(CATEGORICAL_COLUMNS, pandas_categorical):
        df[col] = pd.Categorical(df[col], categories=categories)
    return df


@app.post("/score-batch")
async def score_batch(file: UploadFile = File(...), threshold: float = Query(0.5, ge=0.0, le=1.0)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv")

    try:
        raw_df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if raw_df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows")

    X = _prepare_features(raw_df, app.state.pandas_categorical)
    probabilities = app.state.model.predict_proba(X)[:, 1]

    results = [
        {
            "row_index": i,
            "churn_probability": round(float(p), 6),
            "high_risk": bool(p >= threshold),
        }
        for i, p in enumerate(probabilities)
    ]
    return {"threshold": threshold, "count": len(results), "predictions": results}


@app.get("/health")
def health():
    run = app.state.run
    return {
        "status": "ok",
        "production_run_id": run.info.run_id,
        "promoted_at": _promoted_at(run),
    }


@app.get("/model-info")
def model_info():
    run = app.state.run
    metrics = run.data.metrics
    return {
        "run_id": run.info.run_id,
        "metrics": {
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "roc_auc": metrics.get("roc_auc"),
        },
    }
