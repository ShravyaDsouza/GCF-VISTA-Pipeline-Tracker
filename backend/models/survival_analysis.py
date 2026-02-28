from __future__ import annotations

import pandas as pd
from lifelines import KaplanMeierFitter

from .common import CoreCols, require_columns


def build_survival_frame(df_core: pd.DataFrame, *, event_col: str = "event_closed") -> pd.DataFrame:
    """
    Uses processor-engineered columns:
      - duration_days
      - event_closed OR event_resolved
    """
    cols = CoreCols()
    require_columns(df_core, [cols.duration_days, event_col], where="processed_core")

    out = df_core[[cols.duration_days, event_col]].copy()
    out[cols.duration_days] = pd.to_numeric(out[cols.duration_days], errors="coerce")
    out[event_col] = pd.to_numeric(out[event_col], errors="coerce").fillna(0).astype(int)

    out = out.dropna(subset=[cols.duration_days])
    out = out[out[cols.duration_days] >= 0]

    return out


def fit_kaplan_meier(df_surv: pd.DataFrame, *, duration_col: str = "duration_days", event_col: str = "event_closed"):
    require_columns(df_surv, [duration_col, event_col], where="survival frame")

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=df_surv[duration_col],
        event_observed=df_surv[event_col],
        label="Time-to-Event",
    )
    return kmf


def km_curve_points(kmf: KaplanMeierFitter) -> pd.DataFrame:
    """
    Return survival function as (timeline, survival_prob)
    for frontend plotting without embedding lifelines objects.
    """
    sf = kmf.survival_function_.reset_index()
    sf.columns = ["timeline", "survival_prob"]
    return sf