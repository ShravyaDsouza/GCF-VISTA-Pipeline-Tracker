from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

READINESS_PATH = DATA_DIR / "readiness.xlsx"
ENTITIES_PATH = DATA_DIR / "entities.xlsx"

CORE_OUT_PATH = PROCESSED_DIR / "processed_core.csv"
ENTITY_OUT_PATH = PROCESSED_DIR / "processed_entity_subset.csv"

STATE_MAP = {
    "Cancelled": 0,
    "In Legal Processing": 1,
    "Legal Agreement Effective": 2,
    "Disbursed": 3,
    "Closed": 4,
}

@dataclass
class ProcessorConfig:
    stagnation_legal_days_threshold: int = 365
    drop_duplicates: bool = True
    # Avoid unstable velocity when duration is very small
    min_days_for_velocity: int = 30
    # Fill missing categories so encoders don't break later
    fill_unknown_region_country: bool = True
    # Add a simple outlier flag (useful for weighting/anomaly reporting)
    add_financing_outlier_flag: bool = True


def parse_mixed_excel_date(x) -> pd.Timestamp:
    """Robust date parser for Excel strings or serial numbers."""
    if pd.isna(x):
        return pd.NaT

    if isinstance(x, pd.Timestamp):
        return x

    # Excel serial date numbers (Windows origin)
    if isinstance(x, (int, float)) and not np.isnan(x):
        xv = float(x)
        # typical modern excel serial range
        if 30000 <= xv <= 60000:
            return pd.to_datetime(xv, unit="D", origin="1899-12-30", errors="coerce")
        return pd.NaT

    s = str(x).strip()
    if not s or s.lower() == "nan":
        return pd.NaT

    # Most of your values look like "1 May 2015"
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_r = pd.read_excel(READINESS_PATH)
    df_e = pd.read_excel(ENTITIES_PATH)
    return df_r, df_e


def clean_readiness(df_r: pd.DataFrame, cfg: ProcessorConfig) -> pd.DataFrame:
    df = df_r.copy()

    required = {"Approved Date", "Financing", "Status", "Delivery Partner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in readiness.xlsx: {sorted(missing)}")

    if cfg.drop_duplicates:
        df = df.drop_duplicates()

    # Normalize key categoricals early (and guard missing)
    df["Delivery Partner"] = df["Delivery Partner"].fillna("Unknown").astype(str).str.strip()
    if cfg.fill_unknown_region_country:
        if "Region" in df.columns:
            df["Region"] = df["Region"].fillna("Unknown").astype(str).str.strip()
        if "Country" in df.columns:
            df["Country"] = df["Country"].fillna("Unknown").astype(str).str.strip()

    # Parse approved date
    df["approved_date"] = df["Approved Date"].apply(parse_mixed_excel_date)

    # Numeric financing
    df["financing"] = pd.to_numeric(df["Financing"], errors="coerce")

    # Basic sanity: these should be 0 per your EDA, but keep safety.
    if df["financing"].isna().any():
        # If there are missing financing values, keep them; models can dropna on demand.
        pass

    return df


def engineer_features(df: pd.DataFrame, cfg: ProcessorConfig) -> pd.DataFrame:
    out = df.copy()

    today = pd.Timestamp.today().normalize()

    # Temporal
    out["year"] = out["approved_date"].dt.year

    # Duration proxy (today - approved_date)
    out["duration_days"] = (today - out["approved_date"]).dt.days
    # If there are future dates, mark duration as NaN (prevents nonsense velocity)
    out.loc[out["duration_days"] < 0, "duration_days"] = np.nan

    # Survival encodings:
    # - time-to-closure (Closed only)
    # - time-to-resolution (Closed or Cancelled)
    out["event_closed"] = (out["Status"] == "Closed").astype(int)
    out["event_resolved"] = out["Status"].isin(["Closed", "Cancelled"]).astype(int)

    # Markov state id
    out["state_id"] = out["Status"].map(STATE_MAP)
    if out["state_id"].isna().any():
        unknown_statuses = out.loc[out["state_id"].isna(), "Status"].unique().tolist()
        raise ValueError(f"Unknown Status values found (update STATE_MAP): {unknown_statuses}")

    # Log transform
    out["log_financing"] = np.log1p(out["financing"])

    # Velocity proxy
    out["velocity"] = np.where(
        out["duration_days"] >= cfg.min_days_for_velocity,
        out["financing"] / out["duration_days"],
        np.nan,
    )

    # Stagnation: legal-processing too long
    out["stagnation_flag"] = (
        (out["Status"] == "In Legal Processing")
        & (out["duration_days"] >= cfg.stagnation_legal_days_threshold)
    ).astype(int)

    # Bottleneck: legal states
    out["bottleneck_flag"] = out["Status"].isin(
        ["In Legal Processing", "Legal Agreement Effective"]
    ).astype(int)

    # financing outlier flag (IQR rule)
    if cfg.add_financing_outlier_flag:
        fin = out["financing"]
        q1, q3 = fin.quantile(0.25), fin.quantile(0.75)
        iqr = q3 - q1
        out["financing_outlier_flag"] = (
            (fin < (q1 - 1.5 * iqr)) | (fin > (q3 + 1.5 * iqr))
        ).astype(int)

    return out


def enrich_with_entities(df_core: pd.DataFrame, df_entities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_core.copy()
    df_e = df_entities.copy()

    required_e = {"Entity"}
    missing_e = required_e - set(df_e.columns)
    if missing_e:
        raise ValueError(f"Missing required columns in entities.xlsx: {sorted(missing_e)}")

    # Rename to avoid clashes
    df_e = df_e.rename(
        columns={
            "Entity": "entity_code",
            "Name": "entity_name",
            "Country": "entity_country",
            "Type": "entity_type",
            "Stage": "entity_stage",
            "BM": "entity_bm",
            "Size": "entity_size",
            "Sector": "entity_sector",
            "DAE": "entity_dae",
            "# Approved": "entity_approved_count",
            "FA Financing": "entity_fa_financing",
        }
    )

    df["Delivery Partner"] = df["Delivery Partner"].astype(str).str.strip()
    df_e["entity_code"] = df_e["entity_code"].astype(str).str.strip()

    merged = df.merge(
        df_e,
        left_on="Delivery Partner",
        right_on="entity_code",
        how="left",
        indicator=True,
    )
    merged["entity_match"] = (merged["_merge"] == "both")
    merged = merged.drop(columns=["_merge"])

    subset = merged[merged["entity_match"]].copy()
    return merged, subset


def build_datasets(cfg: ProcessorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_r, df_e = load_raw()
    df_clean = clean_readiness(df_r, cfg)
    df_core = engineer_features(df_clean, cfg)
    df_master, df_entity_subset = enrich_with_entities(df_core, df_e)
    return df_master, df_entity_subset


def save_outputs(df_core: pd.DataFrame, df_entity_subset: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_core.to_csv(CORE_OUT_PATH, index=False)
    df_entity_subset.to_csv(ENTITY_OUT_PATH, index=False)


def print_summary(df_core: pd.DataFrame, df_entity_subset: pd.DataFrame) -> None:
    print("\n================ PROCESSOR SUMMARY ================\n")
    print("Total readiness rows (core):", len(df_core))
    print("Matched entity rows (subset):", len(df_entity_subset))

    if "entity_match" in df_core.columns:
        print(f"Entity match rate: {df_core['entity_match'].mean():.2%}")

        total_fin = df_core["financing"].sum(skipna=True)
        matched_fin = df_core.loc[df_core["entity_match"], "financing"].sum(skipna=True)
        fin_share = (matched_fin / total_fin) if total_fin else float("nan")
        print(f"Financing share covered by Entities: {fin_share:.2%}")

    print("\nStatus distribution:")
    print(df_core["Status"].value_counts(dropna=False))

    print("\nProjects per year (parsed):")
    print(df_core["year"].value_counts(dropna=False).sort_index())

    print("\nSaved outputs:")
    print(f" - {CORE_OUT_PATH}")
    print(f" - {ENTITY_OUT_PATH}")
    print("\n===================================================\n")


def load_processed_data() -> pd.DataFrame:
    return pd.read_csv(CORE_OUT_PATH, parse_dates=["approved_date"])


if __name__ == "__main__":
    cfg = ProcessorConfig(
        stagnation_legal_days_threshold=365,
        drop_duplicates=True,
        min_days_for_velocity=30,
        fill_unknown_region_country=True,
        add_financing_outlier_flag=True,
    )

    df_core, df_entity_subset = build_datasets(cfg)
    save_outputs(df_core, df_entity_subset)
    print_summary(df_core, df_entity_subset)