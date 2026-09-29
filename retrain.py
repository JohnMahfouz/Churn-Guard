import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FEATURES_PATH = "telco_features.parquet"
TARGET = "Churn Value"
EXPERIMENT_NAME = "churn-guard"
PROMOTION_MARGIN = 0.02  # new model must beat production F1 by at least this much to be promoted

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "is_unbalance": True,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "random_state": 42,
}

df = pd.read_parquet(FEATURES_PATH)
train_df = df[df["split"] == "train"].drop(columns=["split"])
test_df = df[df["split"] == "test"].drop(columns=["split"])

feature_cols = [c for c in df.columns if c not in {TARGET, "split"}]
categorical_cols = [c for c in feature_cols if str(train_df[c].dtype) == "category"]

X_train, y_train = train_df[feature_cols], train_df[TARGET]
X_test, y_test = test_df[feature_cols], test_df[TARGET]

mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

prod_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.stage = 'production'",
    order_by=["start_time DESC"],
    max_results=1,
)
prod_run = prod_runs[0] if prod_runs else None
prod_f1 = prod_run.data.metrics.get("f1") if prod_run else None

model = lgb.LGBMClassifier(**PARAMS, verbose=-1)
model.fit(
    X_train,
    y_train,
    eval_X=X_test,
    eval_y=y_test,
    eval_metric="auc",
    categorical_feature=categorical_cols,
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
)

if model.best_iteration_ >= PARAMS["n_estimators"] - 1:
    print(f"NOTE: early stopping never triggered (best_iteration={model.best_iteration_} of "
          f"{PARAMS['n_estimators']} max) -- consider raising n_estimators.")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
new_f1 = metrics["f1"]

fig, ax = plt.subplots(figsize=(8, 6))
lgb.plot_importance(model, ax=ax, max_num_features=20, importance_type="gain")
fig.tight_layout()
importance_path = "feature_importance.png"
fig.savefig(importance_path)
plt.close(fig)

if prod_run is None:
    promote = True
    reason = "no existing production model -- promoting unconditionally as the baseline"
else:
    gap = new_f1 - prod_f1
    promote = gap >= PROMOTION_MARGIN
    reason = (f"F1 gap {gap:+.4f} vs required margin +{PROMOTION_MARGIN:.2f} -- "
              f"{'promoting' if promote else 'not promoting'}")

with mlflow.start_run() as run:
    mlflow.log_params(PARAMS)
    mlflow.log_param("best_iteration", model.best_iteration_)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(importance_path)
    mlflow.lightgbm.log_model(model, name="model")
    new_run_id = run.info.run_id

    if promote:
        client.set_tag(new_run_id, "stage", "production")
        if prod_run is not None:
            client.set_tag(prod_run.info.run_id, "stage", "archived")
    else:
        client.set_tag(new_run_id, "stage", "rejected")

print("\n" + "=" * 50)
print("RETRAIN SUMMARY")
print("=" * 50)
print(f"Old production F1: {f'{prod_f1:.4f}' if prod_f1 is not None else 'N/A (no prior production model)'}")
print(f"New model F1:      {new_f1:.4f}")
if promote:
    print(f"Decision: PROMOTED -- new run {new_run_id} is now production ({reason})")
else:
    print(f"Decision: REJECTED -- new run {new_run_id} tagged rejected ({reason})")

sys.exit(0 if promote else 1)
