from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

CORE_OUT_PATH = PROCESSED_DIR / "processed_core.csv"
ENTITY_OUT_PATH = PROCESSED_DIR / "processed_entity_subset.csv"


def load_processed_core(path: Path = CORE_OUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed core file not found at {path}. Run processor.py first."
        )
    return pd.read_csv(path, parse_dates=["approved_date"])


def require_columns(df: pd.DataFrame, cols: Iterable[str], *, where: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        loc = f" in {where}" if where else ""
        raise ValueError(f"Missing required columns{loc}: {missing}")


@dataclass(frozen=True)
class CoreCols:
    approved_date: str = "approved_date"
    status: str = "Status"
    state_id: str = "state_id"
    financing: str = "financing"
    log_financing: str = "log_financing"
    duration_days: str = "duration_days"
    velocity: str = "velocity"
    stagnation_flag: str = "stagnation_flag"
    bottleneck_flag: str = "bottleneck_flag"
    financing_outlier_flag: str = "financing_outlier_flag"