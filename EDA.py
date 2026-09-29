import re

import pandas as pd

DATA_PATH = "Telco_customer_churn.xlsx"

df = pd.read_excel(DATA_PATH)

print("=== All columns ===")
for col in df.columns:
    print(repr(col))

# Normalize names (lowercase, strip spaces/underscores) so "Churn Score",
# "churn_score", "ChurnScore", etc. all match the same leakage column.
def normalize(name: str) -> str:
    return re.sub(r"[\s_]+", "", name).lower()

LEAKAGE_COLUMNS = ["Churn Score", "Churn Reason", "Churn Label"]
leakage_normalized = {normalize(c): c for c in LEAKAGE_COLUMNS}

normalized_to_actual = {normalize(c): c for c in df.columns}

dropped = []
for norm_name, canonical_name in leakage_normalized.items():
    actual_col = normalized_to_actual.get(norm_name)
    if actual_col is not None:
        df = df.drop(columns=[actual_col])
        dropped.append(actual_col)

# --- Flag CLTV for review (not dropped automatically) ---
cltv_col = normalized_to_actual.get(normalize("CLTV"))
churn_value_col = normalized_to_actual.get(normalize("Churn Value"))

if cltv_col and churn_value_col:
    print("\n=== CLTV vs Churn check ===")
    cltv_by_churn = df.groupby(churn_value_col)[cltv_col].agg(["count", "mean", "median", "std"])
    print(cltv_by_churn)

    mean_churned = cltv_by_churn.loc[1, "mean"] if 1 in cltv_by_churn.index else None
    mean_retained = cltv_by_churn.loc[0, "mean"] if 0 in cltv_by_churn.index else None

    if mean_churned is not None and mean_retained is not None:
        diff_pct = (mean_retained - mean_churned) / mean_retained * 100
        print(f"\nMean CLTV - retained: {mean_retained:.1f}, churned: {mean_churned:.1f}")
        print(f"Churned customers' mean CLTV is {diff_pct:.1f}% {'lower' if diff_pct > 0 else 'higher'} than retained.")
        if diff_pct > 5:
            print(
                "NOTE: CLTV is systematically lower for churned customers. "
                "This is consistent with (but not proof of) CLTV being computed "
                "after/because of churn rather than as a pre-churn forward-looking "
                "prediction. Inspect IBM's CLTV methodology before deciding whether "
                "to keep this column for modeling."
            )
        else:
            print("No strong systematic gap detected between churned/retained CLTV.")
else:
    print("\nCLTV or Churn Value column not found — skipping CLTV leakage check.")

print("\n=== Summary ===")
print(f"Final column count: {df.shape[1]}")
print(f"Dropped columns: {dropped}")
