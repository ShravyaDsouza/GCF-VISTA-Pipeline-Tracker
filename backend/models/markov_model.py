from __future__ import annotations

import numpy as np
import pandas as pd

from .common import require_columns

ABSORBING = {0, 4}  # Cancelled, Closed


def validate_event_log(df_events: pd.DataFrame, *, n_states: int = 5) -> None:
    """
    Markov requires *event-level* data:
      project_id, state_id, status_date
    """
    require_columns(df_events, ["project_id", "state_id", "status_date"], where="Markov event log")

    # state_id sanity
    bad = df_events.loc[~df_events["state_id"].isin(range(n_states)), "state_id"].unique()
    if len(bad) > 0:
        raise ValueError(f"Invalid state_id(s) found: {bad}. Expected 0..{n_states-1}")


def _prep_events(df_events: pd.DataFrame, *, n_states: int = 5) -> pd.DataFrame:
    """
    - Parses dates
    - Sorts
    - Drops invalids
    - Removes consecutive duplicate states per project
    """
    validate_event_log(df_events, n_states=n_states)

    ev = df_events.copy()
    ev["status_date"] = pd.to_datetime(ev["status_date"], errors="coerce")
    ev = ev.dropna(subset=["project_id", "state_id", "status_date"])
    ev["state_id"] = ev["state_id"].astype(int)

    ev = ev.sort_values(["project_id", "status_date"])

    prev_state = ev.groupby("project_id")["state_id"].shift(1)
    ev = ev[(prev_state.isna()) | (ev["state_id"] != prev_state)].copy()

    return ev


def compute_transitions(df_events: pd.DataFrame, *, n_states: int = 5) -> pd.DataFrame:
    """
    Returns transition counts and probabilities:
      state_id, next_state_id, count, prob
    """
    ev = _prep_events(df_events, n_states=n_states)

    ev["next_state_id"] = ev.groupby("project_id")["state_id"].shift(-1)
    trans = ev.dropna(subset=["next_state_id"]).copy()
    trans["next_state_id"] = trans["next_state_id"].astype(int)

    counts = (
        trans.groupby(["state_id", "next_state_id"])
        .size()
        .reset_index(name="count")
    )

    counts["prob"] = counts["count"] / counts.groupby("state_id")["count"].transform("sum")
    return counts.sort_values(["state_id", "prob"], ascending=[True, False])


def transition_matrix(transitions: pd.DataFrame, *, n_states: int = 5) -> np.ndarray:
    """
    transitions: output of compute_transitions
    """
    require_columns(transitions, ["state_id", "next_state_id", "prob"], where="Transitions")

    P = np.zeros((n_states, n_states), dtype=float)
    for _, r in transitions.iterrows():
        i = int(r["state_id"])
        j = int(r["next_state_id"])
        if 0 <= i < n_states and 0 <= j < n_states:
            P[i, j] = float(r["prob"])

    for a in ABSORBING:
        if P[a].sum() == 0:
            P[a, a] = 1.0

    for i in range(n_states):
        s = P[i].sum()
        if s == 0:
            P[i, i] = 1.0
        else:
            P[i] = P[i] / s
    return P


def dwell_time_stats(df_events: pd.DataFrame, *, n_states: int = 5) -> pd.DataFrame:
    """
    Avg time spent (days) in each state before transition.
    """
    ev = _prep_events(df_events, n_states=n_states)

    ev["next_date"] = ev.groupby("project_id")["status_date"].shift(-1)
    ev["dwell_days"] = (ev["next_date"] - ev["status_date"]).dt.days

    tmp = ev.dropna(subset=["dwell_days"]).copy()
    tmp["dwell_days"] = tmp["dwell_days"].clip(lower=0)

    return (
        tmp.groupby("state_id")["dwell_days"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .sort_values("state_id")
    )


def absorption_analysis(P: np.ndarray, *, absorbing: set[int] = ABSORBING) -> dict:
    """
    Absorbing Markov chain analytics:
    - absorption probabilities into each absorbing state
    - expected steps until absorption
    """
    n = P.shape[0]
    absorbing = sorted(absorbing)
    transient = [i for i in range(n) if i not in absorbing]

    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]

    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)          # fundamental matrix
    B = N @ R                         # absorption probabilities
    t = N @ np.ones((Q.shape[0], 1))  # expected steps to absorption

    result = {
        "transient_states": transient,
        "absorbing_states": absorbing,
        "absorption_probabilities": {},
        "expected_steps_to_absorption": {},
    }

    for idx, s in enumerate(transient):
        result["absorption_probabilities"][str(s)] = {
            str(absorbing[j]): float(B[idx, j]) for j in range(len(absorbing))
        }
        result["expected_steps_to_absorption"][str(s)] = float(t[idx, 0])

    return result