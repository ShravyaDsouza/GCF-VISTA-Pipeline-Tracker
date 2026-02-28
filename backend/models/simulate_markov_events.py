from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from models.common import load_processed_core, ENTITY_OUT_PATH, require_columns

ABSORBING = {0, 4}

FORWARD_PATH = [1, 2, 3, 4]

DEFAULT_OUT = Path("data/processed/simulated_events.csv")


def _rng(seed: int = 42):
    return np.random.default_rng(seed)


def _sample_gap_days(r: np.random.Generator, state_from: int) -> int:
    """
    Synthetic dwell time by state (tunable).
    """
    if state_from == 1:   # In Legal Processing
        return int(r.gamma(shape=2.0, scale=120)) + 15   # ~ months
    if state_from == 2:   # Legal Agreement Effective
        return int(r.gamma(shape=2.0, scale=90)) + 10
    if state_from == 3:   # Disbursed
        return int(r.gamma(shape=2.0, scale=180)) + 30   # execution can be long
    return int(r.gamma(shape=2.0, scale=60)) + 7


def _build_path_to_target(r: np.random.Generator, target_state: int) -> list[int]:
    """
    Build a plausible path ending at target_state.
    - If target is Cancelled (0): usually cancels from state 1 or 2.
    - If target is Closed (4): usually 1→2→3→4 with small skip chances.
    - If target is 2 or 3: truncated forward path.
    """
    if target_state == 0:
        # Cancelled: either [1,0] or [1,2,0]
        return [1, 0] if r.random() < 0.75 else [1, 2, 0]

    if target_state in (1, 2, 3, 4):
        path = [1]
        if target_state == 1:
            return path

        # chance to skip 2 (legal effective) and jump to 3
        if target_state >= 3 and r.random() < 0.12:
            path.append(3)
        else:
            path.append(2)
            if target_state >= 3:
                path.append(3)

        if target_state == 4:
            path.append(4)

        # ensure ends at target exactly (safety)
        if path[-1] != target_state:
            # truncate or append
            if target_state in path:
                path = path[: path.index(target_state) + 1]
            else:
                path.append(target_state)
        return path

    # fallback
    return [1]


def generate_synthetic_event_log(
    df_snapshot: pd.DataFrame,
    *,
    out_path: Path = DEFAULT_OUT,
    seed: int = 42,
    use_entity_subset: bool = True,
) -> pd.DataFrame:
    """
    df_snapshot must contain: approved_date, state_id
    project_id is synthesized if missing.
    """
    r = _rng(seed)

    require_columns(df_snapshot, ["approved_date", "state_id"], where="snapshot for Markov simulation")

    df = df_snapshot.copy()
    df["approved_date"] = pd.to_datetime(df["approved_date"], errors="coerce")
    df = df.dropna(subset=["approved_date", "state_id"])
    df["state_id"] = df["state_id"].astype(int)

    if "project_id" not in df.columns:
        df["project_id"] = [f"SIM-{i:05d}" for i in range(len(df))]

    events = []
    for _, row in df.iterrows():
        pid = row["project_id"]
        start_date = row["approved_date"].normalize()
        target = int(row["state_id"])

        path = _build_path_to_target(r, target)

        current_date = start_date
        for i, s in enumerate(path):
            events.append(
                {"project_id": pid, "status_date": current_date, "state_id": s}
            )
            if s in ABSORBING or i == len(path) - 1:
                break
            current_date = current_date + pd.Timedelta(days=_sample_gap_days(r, s))

    df_events = pd.DataFrame(events).sort_values(["project_id", "status_date"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_events.to_csv(out_path, index=False)
    return df_events


if __name__ == "__main__":
    snapshot_path = ENTITY_OUT_PATH if (ENTITY_OUT_PATH.exists()) else None

    if snapshot_path:
        df_snapshot = pd.read_csv(snapshot_path, parse_dates=["approved_date"])
    else:
        df_snapshot = load_processed_core()

    df_events = generate_synthetic_event_log(df_snapshot, out_path=DEFAULT_OUT, seed=42)
    print("Saved:", DEFAULT_OUT)
    print(df_events.head(10))
    print("Rows:", len(df_events))