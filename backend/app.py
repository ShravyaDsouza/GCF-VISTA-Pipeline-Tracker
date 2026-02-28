from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
OUT_DIR = Path("data/processed/model_outputs")


def _json_file(path: Path):
    if not path.exists():
        return jsonify({
            "error": f"{path.name} not found.",
            "hint": "Run: python processor.py && python run_models.py && python run_markov.py"
        }), 404
    return jsonify(json.loads(path.read_text())), 200


def _list_outputs():
    if not OUT_DIR.exists():
        return []
    return sorted([p.name for p in OUT_DIR.glob("*.json")])


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "outputs_dir": str(OUT_DIR), "files": _list_outputs()}), 200


@app.get("/api/survival/km")
def survival_km():
    return _json_file(OUT_DIR / "km_curve.json")


@app.route("/api/anomalies", methods=["GET"])
def anomalies():
    path = OUT_DIR / "anomalies.json"
    if not path.exists():
        return _json_file(path)

    data = json.loads(path.read_text())

    status = request.args.get("status")
    if status:
        data = [r for r in data if str(r.get("Status", "")).strip() == status.strip()]

    top_n = request.args.get("top_n", default=5, type=int)
    data = data[: max(1, top_n)]

    return jsonify(data), 200


@app.get("/api/vista/summary")
def vista_summary():
    return _json_file(OUT_DIR / "vista_summary.json")


@app.get("/api/vista/scores")
def vista_scores():
    path = OUT_DIR / "vista_scores.json"
    if not path.exists():
        return _json_file(path)

    data = json.loads(path.read_text())
    limit = request.args.get("limit", type=int) or 500
    return jsonify(data[: max(1, limit)]), 200


@app.get("/api/segments/<segment_name>")
def segment_stats(segment_name: str):
    return _json_file(OUT_DIR / f"segment_{segment_name}.json")


@app.get("/api/markov/matrix")
def markov_matrix():
    return _json_file(OUT_DIR / "markov_matrix.json")


@app.get("/api/markov/transitions")
def markov_transitions():
    path = OUT_DIR / "markov_transitions.json"
    if not path.exists():
        return _json_file(path)

    data = json.loads(path.read_text())
    from_state = request.args.get("from_state_id", type=int)
    if from_state is not None:
        data = [r for r in data if int(r.get("state_id", -999)) == from_state]
    return jsonify(data), 200


@app.get("/api/markov/dwell")
def markov_dwell():
    return _json_file(OUT_DIR / "markov_dwell.json")


@app.get("/api/markov/absorption")
def markov_absorption():
    return _json_file(OUT_DIR / "markov_absorption.json")


@app.get("/api/outputs")
def outputs():
    return jsonify({"files": _list_outputs()}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)