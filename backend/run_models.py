from __future__ import annotations

import json
from pathlib import Path

from models.common import load_processed_core
from models.isolation_forest import fit_and_score, prepare_features, top_anomalies
from models.survival_analysis import build_survival_frame, fit_kaplan_meier, km_curve_points
from models.weighting_algo import segment_stats, vista_score


OUT_DIR = Path("data/processed/model_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_COL = "event_closed"


def main() -> None:
    df = load_processed_core()

    # --- Survival (Kaplan–Meier) ---
    df_surv = build_survival_frame(df, event_col=EVENT_COL)
    kmf = fit_kaplan_meier(df_surv, duration_col="duration_days", event_col=EVENT_COL)
    km_points = km_curve_points(kmf)

    print("\n[KM] head:")
    print(km_points.head())

    km_points.to_json(OUT_DIR / "km_curve.json", orient="records")

    # --- Isolation Forest (Anomaly Detection) ---
    X = prepare_features(df)
    _, score, pred = fit_and_score(X, contamination=0.05)
    anomalies_df, meta = top_anomalies(df, score, pred, top_n=10)

    ID_COL = "Ref #"
    if ID_COL in anomalies_df.columns:
        anomalies_df["id"] = anomalies_df[ID_COL].fillna("—")
    else:
        anomalies_df["id"] = "—"

    cols = ["id"] + [c for c in [
        "Status",
        "financing",
        "duration_days",
        "velocity",
        "stagnation_flag",
        "bottleneck_flag",
        "financing_outlier_flag",
        "anomaly_score",
        "anomaly_pred",
    ] if c in anomalies_df.columns]

    print("\n[IsoForest] Top anomalies:")
    print(anomalies_df[cols].head(10))

    anomalies_df[cols].to_json(OUT_DIR / "anomalies.json", orient="records")

    # --- Weighting / VISTA Score ---
    df["vista_score"] = vista_score(df)

    print("\n[VISTA] score summary:")
    print(df["vista_score"].describe())

    vista_summary = df["vista_score"].describe().to_dict()
    with open(OUT_DIR / "vista_summary.json", "w") as f:
        json.dump(vista_summary, f)

    df[["vista_score"]].to_json(OUT_DIR / "vista_scores.json", orient="records")

    # --- Segment Stats ---
    seg_candidates = ["Region", "Country", "entity_type", "entity_size"]
    seg_cols = [c for c in seg_candidates if c in df.columns]

    if seg_cols:
        seg = seg_cols[0]
        stats = segment_stats(df, segment_cols=[seg])

        print(f"\n[Segment Stats by {seg}] head:")
        print(stats.head())

        stats.to_json(OUT_DIR / f"segment_{seg}.json", orient="records")
    else:
        print("\n[Segment Stats] No segment columns found in processed_core.")

    print(f"\nSaved model outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()