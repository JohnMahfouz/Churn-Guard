import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "Telco_customer_churn.xlsx"
OUTPUT_PATH = "telco_features.parquet"
RANDOM_STATE = 42
TARGET = "Churn Value"
SMOOTHING = 10  # shrinkage strength toward the global rate for small groups

df = pd.read_excel(DATA_PATH)

LEAKAGE_COLUMNS = ["Churn Score", "Churn Reason", "Churn Label"]
dropped_leakage = [c for c in LEAKAGE_COLUMNS if c in df.columns]
df = df.drop(columns=dropped_leakage)

# ============================================================
# 1. Geographic rollup: region_churn_rate
# ============================================================
# City and Zip Code are too sparse to average directly (median ~4 rows per
# group, Zip tops out at 5), and State is a dead end -- every row is
# California, so a state-level fallback would just be a constant column
# equal to the global rate. Smoothed leave-one-out target encoding at the
# city level fixes this by shrinking small cities toward the training
# global churn rate in proportion to how little data they have, instead of
# trusting a noisy small-group mean or falling back to a signal-free column.
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df[TARGET], random_state=RANDOM_STATE
)
# This split must be reused (not redone) for model training: region_churn_rate
# is only leak-free relative to this exact train/test boundary, and a different
# split would let test-set churn outcomes leak into the training-derived city rates.
df["split"] = np.where(df.index.isin(train_df.index), "train", "test")

global_train_rate = train_df[TARGET].mean()
group_stats = train_df.groupby("City")[TARGET].agg(["sum", "count"])

def loo_encode(row):
    s, n = group_stats.loc[row["City"], ["sum", "count"]]
    loo_sum = s - row[TARGET]
    loo_n = n - 1
    return (loo_sum + SMOOTHING * global_train_rate) / (loo_n + SMOOTHING)

def plain_encode(city):
    if city in group_stats.index:
        s, n = group_stats.loc[city, ["sum", "count"]]
        return (s + SMOOTHING * global_train_rate) / (n + SMOOTHING)
    return global_train_rate

df.loc[train_df.index, "region_churn_rate"] = train_df.apply(loo_encode, axis=1)
df.loc[test_df.index, "region_churn_rate"] = test_df["City"].map(plain_encode)

# ============================================================
# 2. Raw geographic columns
# ============================================================
# State (constant), City and Zip Code (median ~4 rows/category -- effectively
# identity columns to a tree model) are redundant with region_churn_rate, which
# already extracts their useful signal in a regularized form. Latitude/Longitude
# are continuous rather than categorical, but at near-unique-per-customer
# precision they act as a location fingerprint a tree model can split on as an
# identity shortcut -- the same overfitting risk as raw Zip Code. CustomerID is
# a unique identifier and Count/Country are both constant (always 1 / "United
# States"); none of the three carry signal.
REDUNDANT_COLUMNS = ["State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude",
                     "CustomerID", "Count", "Country"]
dropped_redundant = [c for c in REDUNDANT_COLUMNS if c in df.columns]
df = df.drop(columns=dropped_redundant)

# ============================================================
# 3. Satisfaction Score and CLTV
# ============================================================
if "Satisfaction Score" in df.columns:
    print(f"Satisfaction Score dtype: {df['Satisfaction Score'].dtype}, "
          f"missing: {df['Satisfaction Score'].isna().sum()}")
else:
    # Not present in this IBM export (Telco_customer_churn.xlsx); it only
    # appears in IBM's separate "Telco_customer_churn_status" export.
    print("Satisfaction Score: not present in this dataset export, skipped.")
print(f"CLTV dtype: {df['CLTV'].dtype}, missing: {df['CLTV'].isna().sum()}")

# ============================================================
# 4. Total Charges
# ============================================================
total_charges_numeric = pd.to_numeric(df["Total Charges"], errors="coerce")
blank_mask = total_charges_numeric.isna()

if blank_mask.sum() and (df.loc[blank_mask, "Tenure Months"] == 0).all():
    # All blanks are zero-tenure customers who haven't completed a billing
    # cycle yet, so 0 is the factually correct value -- a mean/median fill
    # would misrepresent them as having an established billing history.
    df["Total Charges"] = total_charges_numeric.fillna(0)
    print(f"Total Charges: filled {blank_mask.sum()} blank rows (all zero-tenure) with 0.")
else:
    df["Total Charges"] = total_charges_numeric
    print(f"Total Charges: {blank_mask.sum()} blank rows found but not all zero-tenure "
          "-- not auto-fixed, needs manual review.")

# ============================================================
# 5. Categorical encoding (cast to 'category' dtype, no one-hot)
# ============================================================
exclude = {TARGET, "split", "region_churn_rate",
           "Monthly Charges", "Total Charges", "Tenure Months"}
categorical_cols = [c for c in df.columns if c not in exclude and df[c].dtype == object]

for c in categorical_cols:
    df[c] = df[c].astype("category")

print(f"Cast {len(categorical_cols)} columns to 'category' dtype: {categorical_cols}")

# ============================================================
# Summary
# ============================================================
print(f"\nFinal shape: {df.shape}")
print(f"Dropped leakage columns: {dropped_leakage}")
print(f"Dropped redundant/no-signal columns: {dropped_redundant}")
print("\nDtypes:")
print(df.dtypes)
print("\nFeature list:")
print(list(df.columns))

df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved cleaned features to {OUTPUT_PATH}")
