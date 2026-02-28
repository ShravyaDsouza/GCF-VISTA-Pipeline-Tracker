import json
from pathlib import Path
import pandas as pd
import numpy as np

from models.markov_model import (
    compute_transitions,
    transition_matrix,
    dwell_time_stats,
    absorption_analysis,
)

EVENTS_PATH = Path("data/processed/simulated_events.csv")
OUT_DIR = Path("data/processed/model_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_events = pd.read_csv(EVENTS_PATH)

trans = compute_transitions(df_events)
P = transition_matrix(trans)
most_likely_next = {
    str(i): int(np.argmax(P[i])) for i in range(len(P))
}
dwell = dwell_time_stats(df_events)
absorb = absorption_analysis(P)

STATE_LABELS = {
    0: "Cancelled",
    1: "In Legal Processing",
    2: "Legal Agreement Effective",
    3: "Disbursed",
    4: "Closed",
}

trans["state"] = trans["state_id"].map(STATE_LABELS)
trans["next_state"] = trans["next_state_id"].map(STATE_LABELS)

dwell["state"] = dwell["state_id"].map(STATE_LABELS)
# Save outputs for frontend
trans.to_json(OUT_DIR / "markov_transitions.json", orient="records")
dwell.to_json(OUT_DIR / "markov_dwell.json", orient="records")

with open(OUT_DIR / "markov_matrix.json", "w") as f:
    json.dump(P.tolist(), f)

with open(OUT_DIR / "markov_absorption.json", "w") as f:
    json.dump(absorb, f)

print("Saved Markov outputs to:", OUT_DIR)
print("\nTop transitions:")
print(trans.head(12))
print("\nDwell stats:")
print(dwell)
print("\nAbsorption:")
print(absorb)