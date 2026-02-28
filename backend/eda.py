import re
import unicodedata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)

def parse_mixed_excel_date(x):
    """Robust parser for Excel columns that may contain datetime, strings, or Excel serial numbers."""
    if pd.isna(x):
        return pd.NaT

    if isinstance(x, pd.Timestamp):
        return x

    # Excel serial dates are typically ~ 30000-60000 for modern years.
    if isinstance(x, (int, float)) and not np.isnan(x):
        xv = float(x)
        if 30000 <= xv <= 60000:
            return pd.to_datetime(xv, unit="D", origin="1899-12-30", errors="coerce")
        return pd.NaT

    s = str(x).strip()
    if not s:
        return pd.NaT

    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt


def normalize_key(val: str) -> str:
    """Normalize organization/entity keys for feasibility matching (punctuation/accents/spaces)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""

    s = str(val).strip()
    if not s or s.lower() == "nan":
        return ""

    # Remove accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)  # punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()  # collapse spaces
    return s

df_r = pd.read_excel("data/readiness.xlsx")
df_e = pd.read_excel("data/entities.xlsx")

print("Readiness Shape:", df_r.shape)
print("Entities Shape:", df_e.shape)

print("\nReadiness Columns:", df_r.columns.tolist())
print("Entities Columns:", df_e.columns.tolist())

print("\nReadiness Missing Values (top 20):")
print(df_r.isnull().sum().sort_values(ascending=False).head(20))

print("\nEntities Missing Values (top 20):")
print(df_e.isnull().sum().sort_values(ascending=False).head(20))

print("\nReadiness dtypes:\n", df_r.dtypes)
print("\nEntities dtypes:\n", df_e.dtypes)

dup_r = df_r.duplicated().sum()
dup_e = df_e.duplicated().sum()
print(f"\nDuplicate rows - Readiness: {dup_r}, Entities: {dup_e}")

approved_parsed = df_r["Approved Date"].apply(parse_mixed_excel_date)
bad_dates = approved_parsed.isna().sum()

print(f"\nApproved Date parse failures: {bad_dates} / {len(df_r)}")
print("Approved Date range:", approved_parsed.min(), "to", approved_parsed.max())

if bad_dates > 0:
    bad_examples = df_r.loc[approved_parsed.isna(), "Approved Date"].astype(str).value_counts().head(15)
    print("\nTop unparsed Approved Date raw values (sample):")
    print(bad_examples)

fin_numeric = pd.to_numeric(df_r["Financing"], errors="coerce")
bad_fin = fin_numeric.isna().sum()
print(f"\nFinancing numeric parse failures: {bad_fin} / {len(df_r)}")

print("\n================ STATUS DISTRIBUTION ================\n")
print(df_r["Status"].value_counts(dropna=False))

print("\n================ FINANCING DIAGNOSTICS ================\n")
print(fin_numeric.describe())
print("Skewness (numeric financing):", fin_numeric.skew())

plt.figure()
fin_numeric.dropna().hist(bins=40)
plt.title("Financing Distribution (Raw)")
plt.xlabel("Financing")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

log_fin = np.log1p(fin_numeric.dropna())
plt.figure()
log_fin.hist(bins=40)
plt.title("Financing Distribution (log1p)")
plt.xlabel("log1p(Financing)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

q1, q3 = fin_numeric.quantile(0.25), fin_numeric.quantile(0.75)
iqr = q3 - q1
outlier_mask = (fin_numeric < q1 - 1.5 * iqr) | (fin_numeric > q3 + 1.5 * iqr)
print(f"\nIQR outliers count: {int(outlier_mask.sum())}")

print("\n================ TEMPORAL DIAGNOSTICS ================\n")

df_time = pd.DataFrame(
    {
        "Approved_Parsed": approved_parsed,
        "Financing_Num": fin_numeric,
    }
).dropna(subset=["Approved_Parsed"])

df_time["Year"] = df_time["Approved_Parsed"].dt.year

projects_per_year = df_time["Year"].value_counts().sort_index()
print("Projects per year:\n", projects_per_year)

plt.figure()
projects_per_year.plot(kind="bar")
plt.title("Projects Approved Per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

financing_per_year = df_time.groupby("Year")["Financing_Num"].sum().sort_index()
print("\nTotal Financing per year:\n", financing_per_year)

plt.figure()
financing_per_year.plot(kind="bar")
plt.title("Total Financing Approved Per Year")
plt.xlabel("Year")
plt.ylabel("Total Financing")
plt.tight_layout()
plt.show()

today = pd.Timestamp.today()
duration_days = (today - approved_parsed).dt.days
print("\nDuration (days) proxy summary:")
print(duration_days.dropna().describe())

print("\n================ MERGE FEASIBILITY ================\n")

dp_norm = df_r["Delivery Partner"].apply(normalize_key)
entity_norm = df_e["Entity"].apply(normalize_key)

dp_norm = dp_norm.replace("", np.nan).dropna()
entity_norm = entity_norm.replace("", np.nan).dropna()

overlap = set(dp_norm.unique()) & set(entity_norm.unique())

print(f"Unique Delivery Partner keys: {dp_norm.nunique()}")
print(f"Unique Entity keys: {entity_norm.nunique()}")
print(f"Overlapping keys: {len(overlap)}")

unmatched_dp = dp_norm[~dp_norm.isin(entity_norm)].nunique()
print(f"Unmatched Delivery Partner keys (will not merge): {unmatched_dp}")

dp_norm_full = df_r["Delivery Partner"].apply(normalize_key)
unmatched_mask = ~dp_norm_full.isin(entity_norm.unique())

unmatched_examples = (
    df_r.loc[unmatched_mask, "Delivery Partner"]
    .dropna()
    .value_counts()
    .head(15)
)

print("\nTop unmatched Delivery Partners (sample):")
print(unmatched_examples)

print("\n================ EDA DONE ================\n")