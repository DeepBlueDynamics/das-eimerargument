"""Minimal HTTP wrapper for Weber electrodynamics simulation on Cloud Run."""

import subprocess
import json
import os
import tempfile
import time
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

RESULTS_DIR = "/tmp/weber_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "weber-electrodynamics"})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "weber-electrodynamics",
        "description": "Weber bracket deviation in spinning toroidal geometry",
        "endpoints": {
            "GET /health": "Health check",
            "GET /reference": "Reference CSV data (pre-computed)",
            "POST /run": "Run simulation (accepts JSON config)",
            "GET /results": "List completed runs",
            "GET /results/<run_id>": "Download CSV for a run",
        },
    })


@app.route("/reference", methods=["GET"])
def reference():
    csv_path = os.path.join("/app/data", "brake_recoil_gpu_single.csv")
    if os.path.exists(csv_path):
        return send_file(csv_path, mimetype="text/csv",
                         download_name="brake_recoil_gpu_single.csv")
    return jsonify({"error": "reference data not found"}), 404


@app.route("/run", methods=["POST"])
def run_simulation():
    """Run the brake-recoil simulation. Accepts optional JSON config."""
    body = request.get_json(silent=True) or {}
    binary = body.get("binary", "brake-recoil")
    args = body.get("args", [])

    allowed = ["brake-recoil", "weber-anomaly", "debug-weber",
               "field-residual", "shake-test"]
    if binary not in allowed:
        return jsonify({"error": f"unknown binary: {binary}",
                        "allowed": allowed}), 400

    run_id = f"{int(time.time())}_{binary}"
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # If config JSON provided, write it to a temp file
    config_path = None
    cmd_args = list(args)
    if "config" in body:
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(body["config"], f)
        cmd_args = ["--config", config_path] + cmd_args

    cmd = [f"/usr/local/bin/{binary}"] + cmd_args

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=3500, cwd=run_dir
        )

        # Collect any CSV files produced
        csvs = [f for f in os.listdir(run_dir) if f.endswith(".csv")]

        return jsonify({
            "run_id": run_id,
            "binary": binary,
            "args": cmd_args,
            "returncode": result.returncode,
            "stderr": result.stderr[-5000:] if result.stderr else "",
            "csv_files": csvs,
        })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "simulation timed out (3500s limit)",
                        "run_id": run_id}), 504
    except Exception as e:
        return jsonify({"error": str(e), "run_id": run_id}), 500


@app.route("/results", methods=["GET"])
def list_results():
    if not os.path.exists(RESULTS_DIR):
        return jsonify({"runs": []})
    runs = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        run_dir = os.path.join(RESULTS_DIR, d)
        if os.path.isdir(run_dir):
            csvs = [f for f in os.listdir(run_dir) if f.endswith(".csv")]
            runs.append({"run_id": d, "csv_files": csvs})
    return jsonify({"runs": runs})


@app.route("/results/<run_id>", methods=["GET"])
def get_result(run_id):
    run_dir = os.path.join(RESULTS_DIR, run_id)
    if not os.path.isdir(run_dir):
        return jsonify({"error": "run not found"}), 404

    csvs = [f for f in os.listdir(run_dir) if f.endswith(".csv")]
    if not csvs:
        return jsonify({"error": "no CSV output", "run_id": run_id}), 404

    # Return first CSV
    return send_file(os.path.join(run_dir, csvs[0]), mimetype="text/csv",
                     download_name=csvs[0])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
