import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# Explicit rather than relying on a CLI launcher's implicit sys.path insertion --
# that behavior isn't consistent across ways this script gets run (e.g. Streamlit's
# own AppTest harness doesn't add it), so the sibling import below would fail there
# even though `streamlit run` happens to work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import json
from pathlib import Path

import pandas as pd
import requests
import shap
import streamlit as st
from mlflow.tracking import MlflowClient

import retention_agent as ra

FEATURES_PATH = "data/telco_features.parquet"
TARGET = "Churn Value"
EXPERIMENT_NAME = "churn-guard"
SCORE_API_URL = "http://localhost:8000"
DRIFT_SUMMARY_PATH = "reports/drift_summary.json"
BRIEFS_LOG_PATH = "logs/retention_briefs.jsonl"
# The customer whose brief best demonstrates retrieval working correctly (see
# retention_agent.py fix history): its most distinctive SHAP feature was
# Online Backup=No, it retrieved a past case about an unactivated backup add-on,
# and the recommended action correctly tied the two together.
FEATURED_CUSTOMER_ID = "test-5266"

st.set_page_config(page_title="churn-guard dashboard", layout="wide")
st.title("churn-guard")


# ============================================================
# Shared cached resources
# ============================================================
@st.cache_resource
def get_production_model_and_run():
    return ra.load_production_model(), _get_production_run()


def _get_production_run():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    prod_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'production'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return prod_runs[0] if prod_runs else None


@st.cache_resource
def get_explainer_and_baseline():
    model, _ = get_production_model_and_run()
    explainer = shap.TreeExplainer(model)
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in {TARGET, "split", "label_available"}]
    categorical_cols = [c for c in feature_cols if str(df[c].dtype) == "category"]
    test_df = df[df["split"] == "test"]
    X_test = test_df[feature_cols]
    at_risk_mask = model.predict_proba(X_test)[:, 1] >= ra.AT_RISK_THRESHOLD
    baseline = ra.population_baseline(explainer, X_test[at_risk_mask])
    return explainer, baseline, feature_cols, categorical_cols


@st.cache_resource
def get_agent_graph():
    return ra.build_graph()


def _promoted_at(run) -> str:
    tagged = run.data.tags.get("promoted_at")
    if tagged is not None:
        return tagged
    from datetime import datetime, timezone
    return datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).isoformat()


# ============================================================
# Section 1 -- Model health
# ============================================================
st.header("1. Model Health")

model, prod_run = get_production_model_and_run()

if prod_run is None:
    st.error("No production model found -- run retrain.py first.")
else:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("F1", f"{prod_run.data.metrics.get('f1', 0):.4f}")
    col2.metric("Precision", f"{prod_run.data.metrics.get('precision', 0):.4f}")
    col3.metric("Recall", f"{prod_run.data.metrics.get('recall', 0):.4f}")
    col4.metric("ROC-AUC", f"{prod_run.data.metrics.get('roc_auc', 0):.4f}")
    col5.metric("Run ID", prod_run.info.run_id[:8])
    st.caption(f"Full run ID: `{prod_run.info.run_id}` -- promoted at {_promoted_at(prod_run)}")

st.subheader("Promotion history")
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
all_runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
history_rows = [
    {
        "run_id": run.info.run_id[:8],
        "stage": run.data.tags.get("stage", "(untagged)"),
        "f1": run.data.metrics.get("f1"),
        "started_at": pd.to_datetime(run.info.start_time, unit="ms", utc=True),
    }
    for run in all_runs
]
history_df = pd.DataFrame(history_rows)


def _highlight_stage(row):
    color = {"production": "background-color: #c6f6c6", "rejected": "background-color: #f6c6c6"}.get(
        row["stage"], ""
    )
    return [color] * len(row)


st.dataframe(history_df.style.apply(_highlight_stage, axis=1), use_container_width=True)

st.subheader("Latest drift check")
if Path(DRIFT_SUMMARY_PATH).exists():
    with open(DRIFT_SUMMARY_PATH, encoding="utf-8") as f:
        drift = json.load(f)
    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    dcol1.metric("Feature drift", "DETECTED" if drift["feature_drift_detected"] else "OK")
    dcol2.metric("Prediction drift", "DETECTED" if drift["prediction_drift_detected"] else "OK")
    dcol3.metric("Performance drift", "DETECTED" if drift["performance_drift_detected"] else "OK")
    dcol4.metric("Labels resolved/pending", f"{drift['resolved_count']} / {drift['pending_count']}")
    st.caption(
        f"Resolved-subset F1 {drift['resolved_f1']:.4f} vs. training F1 {drift['training_f1']:.4f} "
        f"-- generated {drift['generated_at']}"
    )
else:
    st.info(f"No drift summary found at {DRIFT_SUMMARY_PATH} -- run drift_check.py first.")

st.divider()

# ============================================================
# Section 2 -- Live scoring demo
# ============================================================
st.header("2. Live Scoring Demo")
st.caption(f"Calls {SCORE_API_URL}/score-batch -- make sure score.py is running.")

if "scored_df" not in st.session_state:
    st.session_state["scored_df"] = None

upload_col, sample_col = st.columns(2)
uploaded_file = upload_col.file_uploader("Upload a CSV of customers to score", type="csv")
use_sample = sample_col.button("Use 10 sample test customers")

csv_bytes = None
original_index = None

if uploaded_file is not None:
    csv_bytes = uploaded_file.getvalue()
elif use_sample:
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = get_explainer_and_baseline()[2]
    test_df = df[df["split"] == "test"]
    sample = test_df[feature_cols].sample(10, random_state=None)
    original_index = sample.index
    csv_bytes = sample.to_csv(index=False).encode("utf-8")

if csv_bytes is not None:
    try:
        response = requests.post(
            f"{SCORE_API_URL}/score-batch",
            files={"file": ("customers.csv", csv_bytes, "text/csv")},
            timeout=30,
        )
        response.raise_for_status()
        predictions = response.json()["predictions"]

        raw_df = pd.read_csv(io.BytesIO(csv_bytes))
        if original_index is not None:
            customer_ids = [f"test-{i}" for i in original_index]
        else:
            customer_ids = [f"row-{p['row_index']}" for p in predictions]

        result_df = raw_df.copy()
        result_df.insert(0, "customer_id", customer_ids)
        result_df["churn_probability"] = [p["churn_probability"] for p in predictions]
        result_df["high_risk"] = [p["high_risk"] for p in predictions]
        result_df = result_df.sort_values("churn_probability", ascending=False).reset_index(drop=True)

        st.session_state["scored_df"] = result_df
    except requests.exceptions.RequestException as exc:
        detail = None
        if getattr(exc, "response", None) is not None:
            try:
                detail = exc.response.json().get("detail")
            except Exception:
                pass
        st.error(f"Could not reach the scoring API at {SCORE_API_URL}: {detail or exc}")

if st.session_state["scored_df"] is not None:
    result_df = st.session_state["scored_df"]
    display_cols = ["customer_id", "churn_probability", "high_risk"] + [
        c for c in result_df.columns if c not in {"customer_id", "churn_probability", "high_risk"}
    ]

    def _highlight_risk(row):
        return ["background-color: #f6c6c6" if row["high_risk"] else ""] * len(row)

    st.dataframe(
        result_df[display_cols].style.apply(_highlight_risk, axis=1).format({"churn_probability": "{:.1%}"}),
        use_container_width=True,
    )

st.divider()

# ============================================================
# Section 3 -- Retention briefs
# ============================================================
st.header("3. Retention Briefs")


def _load_recent_briefs(path: str, limit: int = 10) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    latest_by_customer = {}
    for entry in entries:
        latest_by_customer[entry["customer_id"]] = entry  # later lines overwrite -> keeps most recent
    ordered = sorted(latest_by_customer.values(), key=lambda e: e["timestamp"], reverse=True)
    return ordered[:limit]


briefs = _load_recent_briefs(BRIEFS_LOG_PATH)

featured = [b for b in briefs if b["customer_id"] == FEATURED_CUSTOMER_ID]
other = [b for b in briefs if b["customer_id"] != FEATURED_CUSTOMER_ID]


def _render_brief(entry: dict, label_suffix: str = ""):
    out = entry["output"]
    cases = ", ".join(str(c) for c in out.get("similar_cases_referenced", []))
    with st.expander(
        f"{entry['customer_id']} -- {entry['churn_probability']:.1%} churn risk{label_suffix}"
    ):
        st.markdown(f"**Retrieved similar cases:** {cases}")
        st.markdown(f"**Risk summary:** {out.get('risk_summary', '')}")
        st.markdown(f"**Recommended action:** {out.get('recommended_action', '')}")
        st.markdown(f"**Confidence note:** {out.get('confidence_note', '')}")
        if entry.get("unmentioned_features"):
            st.warning(f"Brief never addressed: {entry['unmentioned_features']}")
        st.caption(entry["timestamp"])


if featured:
    st.subheader("Best example of retrieval working correctly")
    _render_brief(featured[0], label_suffix=" [BEST EXAMPLE]")

if other:
    st.subheader("Recent briefs")
    for entry in other:
        _render_brief(entry)

if not briefs:
    st.info(f"No briefs logged yet at {BRIEFS_LOG_PATH} -- run retention_agent.py first.")

st.subheader("Generate a new brief on demand")
if st.session_state["scored_df"] is None:
    st.caption("Score some customers in Section 2 first to pick one here.")
else:
    result_df = st.session_state["scored_df"]
    selected_id = st.selectbox("Customer", result_df["customer_id"].tolist())
    if st.button("Generate retention brief"):
        with st.spinner("Running SHAP + retrieval + local LLM..."):
            row_data = result_df[result_df["customer_id"] == selected_id].iloc[0]
            churn_probability = float(row_data["churn_probability"])

            explainer, baseline, feature_cols, categorical_cols = get_explainer_and_baseline()
            # Same reconstruction score.py uses: cast to the exact training-time category
            # levels (in fit order) rather than a bare .astype("category"), which would
            # derive categories from just this one row and break the model's encoding.
            feature_row = pd.DataFrame([row_data[feature_cols].to_dict()])
            for col, categories in zip(categorical_cols, model.booster_.pandas_categorical):
                feature_row[col] = pd.Categorical(feature_row[col], categories=categories)

            shap_features = ra.top_shap_features(explainer, feature_row, top_n=ra.N_SHAP_FEATURES)
            query_features = ra.select_query_features(explainer, feature_row, baseline)

            graph = get_agent_graph()
            initial_state = {
                "customer_id": selected_id,
                "churn_probability": churn_probability,
                "shap_features": shap_features,
                "query_features": query_features,
                "retrieved_cases": [],
                "brief": {},
                "unmentioned_features": [],
            }
            final_state = graph.invoke(initial_state)
            ra.log_brief(selected_id, final_state)

        st.success("Brief generated and logged -- refresh the page to see it above.")
        _render_brief(
            {
                "customer_id": selected_id,
                "timestamp": "just now",
                "churn_probability": churn_probability,
                "output": final_state["brief"],
                "unmentioned_features": final_state["unmentioned_features"],
            }
        )
