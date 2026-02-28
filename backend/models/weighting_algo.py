from __future__ import annotations

import pandas as pd

from .common import CoreCols, require_columns


def segment_stats(
    df_core: pd.DataFrame,
    *,
    segment_cols: list[str],
    status_col: str | None = None,
    financing_col: str | None = None,
) -> pd.DataFrame:
    """
    Empirical priors per segment.
    Assumes processor already normalized missing categories where possible.
    """
    cols = CoreCols()
    status_col = status_col or cols.status
    financing_col = financing_col or cols.financing

    require_columns(df_core, [status_col, financing_col], where="processed_core")
    require_columns(df_core, segment_cols, where="processed_core segments")

    tmp = df_core.copy()
    tmp[status_col] = tmp[status_col].astype(str).str.strip()
    tmp[financing_col] = pd.to_numeric(tmp[financing_col], errors="coerce")

    tmp["is_closed"] = (tmp[status_col] == "Closed").astype(int)
    tmp["is_cancelled"] = (tmp[status_col] == "Cancelled").astype(int)
    tmp["is_legal"] = tmp[status_col].isin(["In Legal Processing", "Legal Agreement Effective"]).astype(int)

    out = (
        tmp.groupby(segment_cols)
        .agg(
            n=("is_closed", "size"),
            closure_rate=("is_closed", "mean"),
            cancel_rate=("is_cancelled", "mean"),
            legal_rate=("is_legal", "mean"),
            financing_mean=(financing_col, "mean"),
            financing_median=(financing_col, "median"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    return out


def vista_score(
    df_core: pd.DataFrame,
    *,
    readiness_col: str | None = None,
) -> pd.Series:
    """
    Pure scoring: uses processor-engineered fields only.
    No transformations here.
    """
    cols = CoreCols()
    status = df_core[cols.status].astype(str).str.strip()

    base = status.map({
        "Cancelled": -1.0,
        "In Legal Processing": -0.3,
        "Legal Agreement Effective": 0.2,
        "Disbursed": 0.6,
        "Closed": 1.0,
    }).fillna(0.0)

    log_fin = pd.to_numeric(df_core.get(cols.log_financing, 0), errors="coerce").fillna(0)
    dur = pd.to_numeric(df_core.get(cols.duration_days, 0), errors="coerce").fillna(0).clip(lower=0)

    rcol = readiness_col if readiness_col and readiness_col in df_core.columns else None
    if rcol:
        r = pd.to_numeric(df_core[rcol], errors="coerce").fillna(0)
        if r.max() > 1.5:
            r = (r / 100.0).clip(0, 1)
    else:
        r = 0

    denom = (dur.median() + 1e-9)
    stagn = (dur / denom).clip(0, 3) / 3.0

    fin_scaled = (log_fin / (log_fin.max() + 1e-9)).clip(0, 1)

    score = (0.55 * base) + (0.15 * fin_scaled) + (0.25 * r) - (0.20 * stagn)
    return score.rename("vista_score")