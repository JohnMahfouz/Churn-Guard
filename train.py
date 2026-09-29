import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import pandas as pd
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

feature_cols = [c for c in df.columns if c not in {TARGET, "split", "label_available"}]
categorical_cols = [c for c in feature_cols if str(train_df[c].dtype) == "category"]

X_train, y_train = train_df[feature_cols], train_df[TARGET]
X_test, y_test = test_df[feature_cols], test_df[TARGET]

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

fig, ax = plt.subplots(figsize=(8, 6))
lgb.plot_importance(model, ax=ax, max_num_features=20, importance_type="gain")
fig.tight_layout()
importance_path = "feature_importance.png"
fig.savefig(importance_path)
plt.close(fig)

mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run() as run:
    mlflow.log_params(PARAMS)
    mlflow.log_param("best_iteration", model.best_iteration_)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(importance_path)
    mlflow.lightgbm.log_model(model, name="model")
    run_id = run.info.run_id

print(f"\nMLflow run ID: {run_id}")
print("Test set metrics (churn is imbalanced -- weight precision/recall/F1 over accuracy):")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  F1:        {metrics['f1']:.4f}")
print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print("\nTo view the MLflow UI locally, run:")
print("  mlflow ui")
print("then open http://127.0.0.1:5000 in your browser.")
