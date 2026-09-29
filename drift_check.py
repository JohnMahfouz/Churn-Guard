import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import sys

import mlflow
import pandas as pd
from evidently.legacy.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score

FEATURES_PATH = "telco_features.parquet"
TARGET = "Churn Value"
EXPERIMENT_NAME = "churn-guard"
REPORTS_DIR = "reports"
RANDOM_STATE = 42
SHIFT_FRAC = 0.2
# Evidently's DataDriftPreset ships a documented default (share of drifted columns
# >= 0.5); ClassificationPreset has no equivalent single verdict, so this check
# reuses retrain.py's own promotion margin as the "meaningful change" bar.
PERFORMANCE_DRIFT_MARGIN = 0.02

os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_parquet(FEATURES_PATH)
train_df = df[df["split"] == "train"].drop(columns=["split"])
test_df = df[df["split"] == "test"].drop(columns=["split"])

feature_cols = [c for c in df.columns if c not in {TARGET, "split", "label_available"}]
categorical_cols = [c for c in feature_cols if str(train_df[c].dtype) == "category"]
numerical_cols = [c for c in feature_cols if c not in categorical_cols]

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
prod_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.stage = 'production'",
    order_by=["start_time DESC"],
    max_results=1,
)
if not prod_runs:
    print("No production model found -- run retrain.py first to establish one.")
    sys.exit(1)
prod_run = prod_runs[0]
model = mlflow.lightgbm.load_model(f"runs:/{prod_run.info.run_id}/model")

# Same 20% of test rows, before and after the synthetic shift, so the prediction-drift
# check below isolates the effect of the shift itself rather than which rows were sampled.
shift_idx = test_df.sample(frac=SHIFT_FRAC, random_state=RANDOM_STATE).index
original_batch = test_df.loc[shift_idx, feature_cols].copy()
shifted_batch = original_batch.copy()
shifted_batch["Monthly Charges"] *= 1.10  # simulated price increase
shifted_batch["Total Charges"] *= 1.10  # the price increase flows through to total billed
shifted_batch["Tenure Months"] = (shifted_batch["Tenure Months"] * 0.85).round()  # newer cohort skew

# ============================================================
# 1. Feature drift: shifted batch vs. the original training distribution
# ============================================================
feature_drift_report = Report(metrics=[DataDriftPreset()])
feature_drift_report.run(
    reference_data=train_df[feature_cols],
    current_data=shifted_batch,
    column_mapping=ColumnMapping(
        target=None, prediction=None, numerical_features=numerical_cols, categorical_features=categorical_cols
    ),
)
feature_drift_report.save_html(os.path.join(REPORTS_DIR, "feature_drift.html"))
feature_drift_detected = feature_drift_report.as_dict()["metrics"][0]["result"]["dataset_drift"]

# ============================================================
# 2. Prediction drift: does the shift change what the model actually outputs
# ============================================================
original_proba = model.predict_proba(original_batch)[:, 1]
shifted_proba = model.predict_proba(shifted_batch)[:, 1]

prediction_drift_report = Report(metrics=[DataDriftPreset()])
prediction_drift_report.run(
    reference_data=pd.DataFrame({"churn_probability": original_proba}),
    current_data=pd.DataFrame({"churn_probability": shifted_proba}),
    column_mapping=ColumnMapping(target=None, prediction=None, numerical_features=["churn_probability"]),
)
prediction_drift_report.save_html(os.path.join(REPORTS_DIR, "prediction_drift.html"))
prediction_drift_detected = prediction_drift_report.as_dict()["metrics"][0]["result"]["dataset_drift"]

# ============================================================
# 3. Delayed-label simulation: the test split was "scored 30 days ago", and only
# some of the true labels have "come back" yet -- monitoring has to work with that.
# Which rows are resolved vs. pending is read from the persisted label_available
# column (set once, deterministically, in feature_engineering.py) rather than
# redrawn here -- that column is the single source of truth for reveal state.
# ============================================================
X_test_full = test_df[feature_cols]
y_test_full = test_df[TARGET]
y_pred_full = model.predict(X_test_full)

resolved_mask = test_df["label_available"].to_numpy()
resolved_count = int(resolved_mask.sum())
pending_count = len(test_df) - resolved_count

y_test_resolved = y_test_full[resolved_mask]
y_pred_resolved = y_pred_full[resolved_mask]

training_f1 = prod_run.data.metrics.get("f1")
resolved_precision = precision_score(y_test_resolved, y_pred_resolved)
resolved_recall = recall_score(y_test_resolved, y_pred_resolved)
resolved_f1 = f1_score(y_test_resolved, y_pred_resolved)

performance_report = Report(metrics=[ClassificationPreset()])
performance_report.run(
    reference_data=pd.DataFrame({"target": y_test_full, "prediction": y_pred_full}),
    current_data=pd.DataFrame({"target": y_test_resolved.to_numpy(), "prediction": y_pred_resolved}),
    column_mapping=ColumnMapping(target="target", prediction="prediction", pos_label=1, task="classification"),
)
performance_report.save_html(os.path.join(REPORTS_DIR, "delayed_label_performance.html"))
performance_drift_detected = abs(resolved_f1 - training_f1) >= PERFORMANCE_DRIFT_MARGIN

# ============================================================
# Summary -- kept minimal on purpose; full detail lives in reports/*.html
# ============================================================
print(f"Feature drift:     {'DETECTED' if feature_drift_detected else 'OK'}")
print(f"Prediction drift:  {'DETECTED' if prediction_drift_detected else 'OK'}")
print(f"Performance drift: {'DETECTED' if performance_drift_detected else 'OK'}")
print(f"Labels: {resolved_count} resolved, {pending_count} pending -- "
      f"F1 training={training_f1:.4f} resolved={resolved_f1:.4f} "
      f"(precision={resolved_precision:.4f}, recall={resolved_recall:.4f})")
