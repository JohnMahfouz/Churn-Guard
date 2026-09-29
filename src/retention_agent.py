import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import json
from datetime import datetime, timezone
from typing import TypedDict

import chromadb
import mlflow
import numpy as np
import pandas as pd
import shap
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

FEATURES_PATH = "data/telco_features.parquet"
TARGET = "Churn Value"
EXPERIMENT_NAME = "churn-guard"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "retention_cases"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"
LOG_PATH = "logs/retention_briefs.jsonl"
N_RETRIEVED_CASES = 2
N_SHAP_FEATURES = 4          # top-4 by raw magnitude -- shown to the rep, checked for completeness
N_QUERY_CANDIDATES = 8       # wider pool to pick distinctive query features from
N_QUERY_FEATURES = 4         # top-4 by distinctiveness -- used to build the retrieval query
AT_RISK_THRESHOLD = 0.5      # population used for the "typical high-risk profile" baseline

# The same 5 categories retention_cases.md is organized by.
CATEGORY_LABELS = {
    "price": "Price Sensitivity / Cost Complaints",
    "service": "Poor Service Experience (Outages, Support Issues)",
    "competitor": "Competitor Offer / Switching",
    "life_change": "Life Change (Moving, Downsizing, No Longer Needs Service)",
    "engagement": "Low Engagement / Underuse of Paid Features",
}


class RetentionBrief(BaseModel):
    risk_summary: str
    similar_cases_referenced: list[int]
    recommended_action: str
    confidence_note: str


class AgentState(TypedDict):
    customer_id: str
    churn_probability: float
    shap_features: list[dict]        # top-4 by magnitude -- for the brief itself
    query_features: list[dict]        # top-4 by distinctiveness -- for retrieval only
    retrieved_cases: list[dict]
    brief: dict
    unmentioned_features: list[str]


def load_production_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    prod_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'production'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not prod_runs:
        raise RuntimeError("No production model found -- run retrain.py first.")
    return mlflow.lightgbm.load_model(f"runs:/{prod_runs[0].info.run_id}/model")


def _raw_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # some shap/lightgbm combinations return [class0, class1]
    return shap_values


def top_shap_features(explainer: shap.TreeExplainer, row: pd.DataFrame, top_n: int) -> list[dict]:
    contributions = _raw_shap_values(explainer, row)[0]
    ranked = sorted(zip(row.columns, row.iloc[0], contributions), key=lambda t: abs(t[2]), reverse=True)
    return [
        {"feature": feat, "value": str(val), "contribution": round(float(contrib), 4)}
        for feat, val, contrib in ranked[:top_n]
    ]


def population_baseline(explainer: shap.TreeExplainer, X_at_risk: pd.DataFrame) -> dict[str, float]:
    """Mean |SHAP contribution| per feature across the at-risk population (predicted
    probability >= AT_RISK_THRESHOLD). Contract and tenure turned out to dominate the
    raw top-SHAP list for nearly every at-risk customer -- ranking retrieval queries by
    raw magnitude alone made every query (and therefore every retrieval) nearly
    identical. This baseline lets us instead ask "is this feature unusually large for
    THIS customer" rather than "is this feature large" -- Contract will score high on
    the latter for almost everyone, and low on the former.
    """
    contributions = _raw_shap_values(explainer, X_at_risk)
    mean_abs = np.abs(contributions).mean(axis=0)
    return dict(zip(X_at_risk.columns, mean_abs))


def select_query_features(
    explainer: shap.TreeExplainer, row: pd.DataFrame, baseline: dict[str, float], top_n: int = N_QUERY_FEATURES
) -> list[dict]:
    """Pick the features most distinctive for this specific customer -- large
    relative to how much that feature typically moves predictions for at-risk
    customers generally -- from a wider candidate pool than the raw top-4."""
    candidates = top_shap_features(explainer, row, top_n=N_QUERY_CANDIDATES)
    for f in candidates:
        typical = baseline.get(f["feature"], 1e-6)
        f["distinctiveness"] = abs(f["contribution"]) / max(typical, 1e-6)
    candidates.sort(key=lambda f: f["distinctiveness"], reverse=True)
    return candidates[:top_n]


# Best-effort mapping from a SHAP feature+value to the categories it plausibly
# signals. Several features are genuinely ambiguous alone (tenure, contract type
# on their own don't say *why* someone is at risk) -- those return no category
# vote here and only contribute concrete detail text to the query; category
# assignment for this customer comes from combining whichever OTHER distinctive
# features are present, as the task called for.
def feature_category_votes(feature: str, value: str) -> list[str]:
    if feature in ("Monthly Charges", "Total Charges"):
        return [CATEGORY_LABELS["price"]]
    if feature == "Payment Method" and value == "Electronic check":
        return [CATEGORY_LABELS["price"]]
    if feature == "Tech Support" and value == "No":
        return [CATEGORY_LABELS["service"]]
    if feature in ("Online Security", "Online Backup", "Device Protection"):
        return [CATEGORY_LABELS["service"]] if value == "No" else [CATEGORY_LABELS["engagement"]]
    if feature == "Internet Service" and value == "Fiber optic":
        return [CATEGORY_LABELS["service"], CATEGORY_LABELS["price"]]
    if feature in ("Streaming TV", "Streaming Movies") and value == "Yes":
        return [CATEGORY_LABELS["engagement"]]
    if feature == "region_churn_rate":
        return [CATEGORY_LABELS["competitor"]]
    if feature in ("Senior Citizen", "Partner", "Dependents"):
        return [CATEGORY_LABELS["life_change"]]
    if feature == "Contract" and value == "Month-to-month":
        return [CATEGORY_LABELS["competitor"]]
    return []


# ============================================================
# Node 1: retrieve similar past cases from Chroma
# ============================================================
_embed_model = SentenceTransformer(EMBEDDING_MODEL)


def retrieve_similar_cases(state: AgentState) -> AgentState:
    votes: list[str] = []
    for f in state["query_features"]:
        votes.extend(feature_category_votes(f["feature"], f["value"]))
    categories = sorted(set(votes), key=votes.count, reverse=True)

    category_text = f"Likely category: {', '.join(categories)}. " if categories else ""
    factor_text = "; ".join(
        f"{f['feature']}={f['value']} ({'increases' if f['contribution'] > 0 else 'decreases'} risk)"
        for f in state["query_features"]
    )
    query_text = f"{category_text}Distinctive risk factors for this customer: {factor_text}"

    query_embedding = _embed_model.encode([query_text], normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.query(query_embeddings=query_embedding, n_results=N_RETRIEVED_CASES)

    retrieved = [
        {"case_id": int(case_id), "similarity": round(1 - distance, 4), **meta}
        for case_id, distance, meta in zip(
            results["ids"][0], results["distances"][0], results["metadatas"][0]
        )
    ]
    return {**state, "retrieved_cases": retrieved}


# ============================================================
# Node 2: generate the retention brief with a local LLM
# ============================================================
PROMPT_TEMPLATE = """You are a churn retention analyst writing a brief for a customer service rep.

CUSTOMER
Churn probability: {churn_probability:.0%}
Top risk factors (SHAP contributions -- positive pushes risk up, negative pushes it down):
{shap_summary}

SIMILAR PAST CASES (retrieved by similarity to this customer's risk profile)
{cases_summary}

Write a retention brief with exactly these three lines, no markdown, no extra commentary:
RISK_SUMMARY: <two to three sentences on why this customer is at risk>
RECOMMENDED_ACTION: <one specific, concrete action to try with this customer>
CONFIDENCE_NOTE: <one sentence on how confident you are and why, referencing the retrieved cases by number>

Requirements:
- RISK_SUMMARY must explicitly mention EVERY one of the {n_features} risk factors listed above by name,
  including any that DECREASE risk -- say what role each one plays, don't just cover the largest one.
- If a similar case above is marked FAILED ATTEMPT, do not recommend repeating that same action --
  either propose something different or explicitly justify why this customer differs enough to try it anyway.
"""


def check_unmentioned_features(brief: RetentionBrief, shap_features: list[dict]) -> list[str]:
    """Lightweight, not exhaustive: flags a feature only if NONE of its significant
    name-words appear anywhere in the brief text. Matching on any one word (e.g.
    'tenure' for 'Tenure Months') rather than the full exact phrase avoids flagging
    natural paraphrases like "short tenure of 1 month" as a silent drop -- a feature
    CAN still be addressed in wording this misses entirely (false negative), but it
    cannot be silently skipped without at least one of its words appearing, which is
    the failure mode this guards against."""
    full_text = " ".join([brief.risk_summary, brief.recommended_action, brief.confidence_note]).lower()
    unmentioned = []
    for f in shap_features:
        words = [w for w in f["feature"].lower().split() if len(w) > 2]
        if not any(w in full_text for w in words):
            unmentioned.append(f["feature"])
    return unmentioned


def generate_brief(state: AgentState) -> AgentState:
    shap_summary = "\n".join(
        f"- {f['feature']} = {f['value']} (contribution {f['contribution']:+.3f})"
        for f in state["shap_features"]
    )
    cases_summary = "\n\n".join(
        f"Case {c['case_id']} [{c['category']}] -- {'SUCCESS' if c['success'] else 'FAILED ATTEMPT'} "
        f"(similarity {c['similarity']:.2f})\n"
        f"Profile: {c['customer_profile']}\n"
        f"Flagged reason: {c['flagged_reason']}\n"
        f"Action taken: {c['action_taken']}\n"
        f"Outcome: {c['outcome']}"
        for c in state["retrieved_cases"]
    )
    prompt = PROMPT_TEMPLATE.format(
        churn_probability=state["churn_probability"],
        shap_summary=shap_summary,
        cases_summary=cases_summary,
        n_features=len(state["shap_features"]),
    )

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
    response = llm.invoke(prompt).content

    fields = {"risk_summary": "", "recommended_action": "", "confidence_note": ""}
    labels = {
        "RISK_SUMMARY:": "risk_summary",
        "RECOMMENDED_ACTION:": "recommended_action",
        "CONFIDENCE_NOTE:": "confidence_note",
    }
    for line in response.splitlines():
        stripped = line.strip()
        for label, key in labels.items():
            if stripped.upper().startswith(label):
                fields[key] = stripped.split(":", 1)[1].strip()

    # If the model didn't follow the format, keep the raw response visible rather
    # than silently shipping an empty risk_summary.
    brief = RetentionBrief(
        risk_summary=fields["risk_summary"] or response.strip(),
        similar_cases_referenced=[c["case_id"] for c in state["retrieved_cases"]],
        recommended_action=fields["recommended_action"],
        confidence_note=fields["confidence_note"],
    )
    unmentioned = check_unmentioned_features(brief, state["shap_features"])
    return {**state, "brief": brief.model_dump(), "unmentioned_features": unmentioned}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_similar_cases)
    graph.add_node("generate", generate_brief)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def select_test_customers(df: pd.DataFrame, model, feature_cols: list[str], n_high: int = 3, n_borderline: int = 2):
    test_df = df[df["split"] == "test"]
    X = test_df[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    scored = pd.DataFrame({"proba": proba}, index=test_df.index)

    high_risk = scored.sort_values("proba", ascending=False).head(n_high)
    remaining = scored.drop(index=high_risk.index)
    borderline = remaining.assign(dist=(remaining["proba"] - 0.5).abs()).sort_values("dist").head(n_borderline)

    selected = pd.concat([high_risk[["proba"]], borderline[["proba"]]])
    return selected, X.loc[selected.index]


def log_brief(customer_id: str, state: AgentState) -> None:
    entry = {
        "customer_id": customer_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "churn_probability": state["churn_probability"],
        "shap_features": state["shap_features"],
        "query_features": state["query_features"],
        "output": state["brief"],
        "unmentioned_features": state["unmentioned_features"],
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in {TARGET, "split", "label_available"}]

    model = load_production_model()
    explainer = shap.TreeExplainer(model)
    graph = build_graph()

    scored, X_selected = select_test_customers(df, model, feature_cols)

    # Baseline computed once over the full at-risk test population, reused for every customer.
    test_df = df[df["split"] == "test"]
    X_test = test_df[feature_cols]
    at_risk_mask = model.predict_proba(X_test)[:, 1] >= AT_RISK_THRESHOLD
    baseline = population_baseline(explainer, X_test[at_risk_mask])

    for row_index, proba in scored["proba"].items():
        customer_id = f"test-{row_index}"
        row = X_selected.loc[[row_index]]
        shap_features = top_shap_features(explainer, row, top_n=N_SHAP_FEATURES)
        query_features = select_query_features(explainer, row, baseline)

        initial_state: AgentState = {
            "customer_id": customer_id,
            "churn_probability": float(proba),
            "shap_features": shap_features,
            "query_features": query_features,
            "retrieved_cases": [],
            "brief": {},
            "unmentioned_features": [],
        }
        final_state = graph.invoke(initial_state)
        log_brief(customer_id, final_state)

        print("=" * 70)
        print(f"Customer {customer_id} -- churn probability {proba:.1%}")
        print("Top SHAP factors (shown to rep):")
        for f in shap_features:
            print(f"  {f['feature']} = {f['value']} ({f['contribution']:+.3f})")
        print("Query factors (by distinctiveness, used for retrieval):")
        for f in query_features:
            print(f"  {f['feature']} = {f['value']} (distinctiveness {f['distinctiveness']:.2f}x)")
        print(f"Retrieved cases: {[c['case_id'] for c in final_state['retrieved_cases']]}")
        print("\nRISK_SUMMARY:      ", final_state["brief"]["risk_summary"])
        print("RECOMMENDED_ACTION:", final_state["brief"]["recommended_action"])
        print("CONFIDENCE_NOTE:   ", final_state["brief"]["confidence_note"])
        if final_state["unmentioned_features"]:
            print(f"WARNING: brief never mentioned: {final_state['unmentioned_features']}")
        print()

    print(f"Logged {len(scored)} briefs to {LOG_PATH}")
