import re

import chromadb
from sentence_transformers import SentenceTransformer

CASES_PATH = "docs/retention_cases.md"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "retention_cases"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CATEGORY_PATTERN = re.compile(r"^## \d+\.\s*(.+?)\s*$", re.MULTILINE)
CASE_PATTERN = re.compile(
    r"### Case (\d+)\s*\n\n"
    r"\*\*Customer Profile:\*\*\s*(.*?)\n\n"
    r"\*\*Flagged Reason:\*\*\s*(.*?)\n\n"
    r"\*\*Action Taken:\*\*\s*(.*?)\n\n"
    r"\*\*Outcome:\*\*\s*(.*?)\s*(?:\n\n---|\Z)",
    re.DOTALL,
)


def parse_cases(path: str) -> list[dict]:
    text = open(path, encoding="utf-8").read()

    # Split on category headers; re.split with a capturing group interleaves the
    # preamble, each category name, and the block of cases that follows it.
    parts = CATEGORY_PATTERN.split(text)
    cases = []
    for category, block in zip(parts[1::2], parts[2::2]):
        for match in CASE_PATTERN.finditer(block):
            case_id, profile, reason, action, outcome = match.groups()
            cases.append({
                "case_id": int(case_id),
                "category": category,
                "customer_profile": profile.strip(),
                "flagged_reason": reason.strip(),
                "action_taken": action.strip(),
                "outcome": outcome.strip(),
                # "churn" only ever appears in outcomes that end in failure (including
                # the one case -- #10 -- that was temporarily retained before churning
                # later); every genuine save's outcome text avoids the word entirely.
                "success": "churn" not in outcome.lower(),
            })
    return cases


def case_text(case: dict) -> str:
    # These four fields describe one coherent short narrative per case (who they
    # were, why flagged, what was done, what happened) rather than independent
    # documents, so one embedding per case -- not one per field -- keeps that
    # narrative coherence instead of fragmenting it into four disjoint vectors.
    return (
        f"Customer profile: {case['customer_profile']} "
        f"Flagged reason: {case['flagged_reason']} "
        f"Action taken: {case['action_taken']} "
        f"Outcome: {case['outcome']}"
    )


if __name__ == "__main__":
    cases = parse_cases(CASES_PATH)
    print(f"Parsed {len(cases)} cases from {CASES_PATH}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    # normalize_embeddings=True + an explicit cosine space below means Chroma's
    # returned "distance" is exactly 1 - cosine_similarity, so similarity = 1 - distance
    # with no fudge factor -- Chroma defaults to squared L2, which wouldn't have that
    # clean a relationship to similarity.
    embeddings = model.encode([case_text(c) for c in cases], normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[str(c["case_id"]) for c in cases],
        embeddings=embeddings,
        documents=[case_text(c) for c in cases],
        metadatas=[
            {
                "category": c["category"],
                "customer_profile": c["customer_profile"],
                "flagged_reason": c["flagged_reason"],
                "action_taken": c["action_taken"],
                "outcome": c["outcome"],
                "success": c["success"],
            }
            for c in cases
        ],
    )
    print(f"Stored {collection.count()} embeddings in Chroma collection '{COLLECTION_NAME}' at {CHROMA_DIR}/")

    # Sanity check
    query = "customer complaining about price, threatening to leave"
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    print(f"\nSanity check -- query: {query!r}")
    for rank, (case_id, distance, meta) in enumerate(
        zip(results["ids"][0], results["distances"][0], results["metadatas"][0]), start=1
    ):
        similarity = 1 - distance  # cosine space: distance == 1 - cosine_similarity
        outcome_tag = "SUCCESS" if meta["success"] else "FAILED"
        print(f"{rank}. Case {case_id} ({meta['category']}, {outcome_tag}) -- similarity {similarity:.3f}")
        print(f"   Profile: {meta['customer_profile']}")
        print(f"   Reason:  {meta['flagged_reason']}")
