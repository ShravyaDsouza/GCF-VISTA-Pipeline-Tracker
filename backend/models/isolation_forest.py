from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from backend.models.common import CoreCols, require_columns

def _infer_status_col(df: pd.DataFrame) -> str:
    cols = CoreCols()
    for candidate in [
        getattr(cols, "status", None),
        getattr(cols, "current_status", None),
        getattr(cols, "pipeline_status", None),
        "status",
        "current_status",
        "pipeline_status",
        "Stage",
        "Status",
    ]:
        if isinstance(candidate, str) and candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not infer status column. Pass status_col explicitly or add it to CoreCols."
    )


def _robust_z_within_group(
    s: pd.Series,
    *,
    eps: float = 1e-9,
) -> pd.Series:
    """
    Robust z-score using median and MAD:
      z = (x - median) / (1.4826 * MAD)
    MAD=median(|x-median|). 1.4826 makes it comparable to std for normal data.
    """
    med = s.median(skipna=True)
    mad = (s - med).abs().median(skipna=True)
    denom = (1.4826 * mad) if (mad and mad > 0) else eps
    return (s - med) / denom


def _safe_numeric(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col, errors="coerce")


def prepare_features(
    df_core: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    status_col: str | None = None,
    add_contextual_z: bool = True,
    add_missing_flags: bool = True,
) -> pd.DataFrame:
    """
    Builds the model matrix X for IsolationForest.

    Adds (optional):
      1) Contextual normalization: within-status robust z-scores for numeric drivers
      2) Missing-value flags: <col>__missing (0/1)

    Returns a scaled dataframe (RobustScaler) with stable indices for scoring alignment.
    """
    cols = CoreCols()
    default = [
        cols.log_financing,
        cols.duration_days,
        cols.velocity,
        cols.stagnation_flag,
        cols.bottleneck_flag,
        cols.financing_outlier_flag,
    ]
    feature_cols = feature_cols or [c for c in default if c in df_core.columns]
    if not feature_cols:
        raise ValueError("No IsoForest features available in processed_core.")

    require_columns(df_core, feature_cols, where="processed_core")

    status_col = status_col or _infer_status_col(df_core)
    require_columns(df_core, [status_col], where="processed_core")

    X = df_core[feature_cols].copy()

    for c in feature_cols:
        X[c] = _safe_numeric(X[c])

    if add_missing_flags:
        for c in feature_cols:
            X[f"{c}__missing"] = X[c].isna().astype(int)

    contextual_cols: list[str] = []
    if add_contextual_z:
        candidates = [cols.log_financing, cols.duration_days, cols.velocity]
        candidates = [c for c in candidates if c in X.columns]

        grp = df_core[status_col]
        for c in candidates:
            zname = f"{c}__z_in_status"
            X[zname] = X[c].groupby(grp).transform(_robust_z_within_group)
            contextual_cols.append(zname)

    med = X.median(numeric_only=True)
    X = X.fillna(med)

    scaler = RobustScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return Xs

def fit_and_score(
    X: pd.DataFrame,
    *,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[IsolationForest, pd.Series, pd.Series]:
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    score = pd.Series(model.decision_function(X), index=X.index, name="anomaly_score")
    pred = pd.Series(model.predict(X), index=X.index, name="anomaly_pred")  # -1 anomaly
    return model, score, pred


def contamination_sensitivity(
    X: pd.DataFrame,
    *,
    rates: tuple[float, ...] = (0.03, 0.05, 0.08),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Validate how stable anomaly sets are across contamination rates.

    Returns a dataframe with:
      - contamination
      - n_anomalies
      - jaccard_vs_0.05  (if 0.05 in rates)
    """
    if not rates:
        raise ValueError("Provide at least one contamination rate.")

    base_rate = 0.05 if 0.05 in rates else rates[0]
    base_model, base_score, base_pred = fit_and_score(
        X, contamination=base_rate, random_state=random_state
    )
    base_set = set(X.index[base_pred == -1])

    rows: list[dict[str, Any]] = []
    for r in rates:
        _, _, pred = fit_and_score(X, contamination=r, random_state=random_state)
        s = set(X.index[pred == -1])
        inter = len(s & base_set)
        union = len(s | base_set) if (s | base_set) else 1
        rows.append(
            {
                "contamination": r,
                "n_anomalies": len(s),
                f"jaccard_vs_{base_rate:.2f}": inter / union,
            }
        )
    return pd.DataFrame(rows).sort_values("contamination")


def anomalies_summary(
    df_with_scores: pd.DataFrame,
    *,
    status_col: str,
    top_n: int,
) -> dict[str, Any]:
    """
    Small summary block for API response:
      - total returned
      - status distribution
      - anomaly score stats (min/median/max)
    """
    # status distribution
    dist = (
        df_with_scores[status_col]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    score_col = "anomaly_score" if "anomaly_score" in df_with_scores.columns else None
    score_stats = {}
    if score_col:
        s = df_with_scores[score_col].dropna()
        if len(s):
            score_stats = {
                "min": float(s.min()),
                "median": float(s.median()),
                "max": float(s.max()),
            }

    return {
        "top_n": top_n,
        "returned": int(len(df_with_scores)),
        "status_distribution": dist,
        "score_stats": score_stats,
    }


def interpretability_narrative(
    df_core: pd.DataFrame,
    df_top: pd.DataFrame,
    *,
    status_col: str,
    base_feature_cols: list[str],
    top_k_features: int = 3,
) -> list[dict[str, Any]]:
    """
    Lightweight interpretability:
    For each returned anomaly, report which base features are most extreme
    relative to the same status group (using robust z within status).

    Returns list of per-row explanations suitable for showing in dashboard/tooltips.
    """
    explanations: list[dict[str, Any]] = []

    # Build robust z columns for base continuous drivers (computed on raw core, not scaled)
    cont_cols = [c for c in base_feature_cols if c in df_core.columns]
    cont_cols = [c for c in cont_cols if c != status_col]

    tmp = df_core[[status_col] + cont_cols].copy()
    for c in cont_cols:
        tmp[c] = _safe_numeric(tmp[c])

    for c in cont_cols:
        tmp[f"{c}__rz"] = tmp[c].groupby(tmp[status_col]).transform(_robust_z_within_group)

    rz_cols = [f"{c}__rz" for c in cont_cols if f"{c}__rz" in tmp.columns]

    for idx, row in df_top.iterrows():
        status = str(df_core.loc[idx, status_col]) if idx in df_core.index else "Unknown"

        contrib = {}
        if idx in tmp.index:
            for c in cont_cols:
                rz = tmp.at[idx, f"{c}__rz"] if f"{c}__rz" in tmp.columns else np.nan
                contrib[c] = float(rz) if pd.notna(rz) else 0.0

        top_feats = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k_features]

        explanations.append(
            {
                "index": str(idx),
                "status": status,
                "top_drivers": [
                    {"feature": f, "robust_z_in_status": z} for f, z in top_feats
                ],
                "note": "Drivers reflect deviation vs. peers in the same status (robust z).",
            }
        )

    return explanations


def top_anomalies(
    df_core: pd.DataFrame,
    score: pd.Series,
    pred: pd.Series,
    *,
    top_n: int = 10,
    anomalies_only: bool = True,
    status_col: str | None = None,
    include_summary: bool = True,
    include_explanations: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """
    Returns the most anomalous records (project-level).
    - Filters to true anomalies by default (pred == -1)
    - Adds a summary distribution block for API
    - Adds lightweight interpretability explanations for dashboard
    """
    status_col = status_col or _infer_status_col(df_core)

    if not score.index.equals(df_core.index):
        score = score.reindex(df_core.index)
    if not pred.index.equals(df_core.index):
        pred = pred.reindex(df_core.index)

    out = df_core.copy()
    out["anomaly_score"] = score
    out["anomaly_pred"] = pred

    out = out.sort_values("anomaly_score", ascending=True)

    if anomalies_only:
        out = out[out["anomaly_pred"] == -1]

    top = out.head(top_n)

    meta: dict[str, Any] | None = None
    if include_summary or include_explanations:
        meta = {}

        if include_summary:
            meta["summary"] = anomalies_summary(top, status_col=status_col, top_n=top_n)

        if include_explanations:
            cols = CoreCols()
            # explain using the raw “drivers” that matter for stakeholders
            base_cols = [
                cols.log_financing if getattr(cols, "log_financing", None) in df_core.columns else None,
                cols.duration_days if getattr(cols, "duration_days", None) in df_core.columns else None,
                cols.velocity if getattr(cols, "velocity", None) in df_core.columns else None,
            ]
            base_cols = [c for c in base_cols if c]
            meta["explanations"] = interpretability_narrative(
                df_core,
                top,
                status_col=status_col,
                base_feature_cols=base_cols,
                top_k_features=3,
            )

    return top, meta