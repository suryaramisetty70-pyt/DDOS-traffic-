"""
Web Application: DDoS Traffic Classification & Threat Alert System (SOC)
Framework: Flask (Standalone, High-Performance, Interactive Web Server)
Features:
- REST API for traffic batch inspection
- Interactive Cyber SOC Dashboard
- Real-time packet streaming simulator
- 5-Model benchmark hub
- Instant CSV export
"""

import os
import sys

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import io
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024 # 32 MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Load artifacts
print("[App Startup] Loading trained ML models, scaler, and schemas...")
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))

with open(os.path.join(MODELS_DIR, "feature_columns.json"), "r") as f:
    feature_columns = json.load(f)

with open(os.path.join(MODELS_DIR, "model_metrics.json"), "r") as f:
    model_metrics = json.load(f)

# Ensure CPU safety
if hasattr(best_model, "n_jobs"):
    best_model.n_jobs = 1
for m in all_models.values():
    if hasattr(m, "n_jobs"):
        m.n_jobs = 1

last_analyzed_df = None

def preprocess_and_predict(df_input: pd.DataFrame, model_name="Random Forest"):
    model = all_models.get(model_name, best_model)
    
    clean_df = df_input.copy()
    drop_candidates = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "Label"]
    clean_df = clean_df.drop(columns=[c for c in drop_candidates if c in clean_df.columns], errors="ignore")
    
    # Fill any missing features
    for col in feature_columns:
        if col not in clean_df.columns:
            clean_df[col] = 0.0
            
    clean_df = clean_df[feature_columns].copy()
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    X_scaled = scaler.transform(clean_df.to_numpy(dtype=np.float64))
    preds = model.predict(X_scaled)
    
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_scaled)
        confidences = [round(float(probas[i, preds[i]]) * 100, 1) for i in range(len(preds))]
    else:
        confidences = [100.0] * len(preds)
        
    labels = ["Normal" if p == 0 else "DDoS Attack" for p in preds]
    
    total = len(preds)
    ddos_count = int(np.sum(preds == 1))
    normal_count = total - ddos_count
    attack_rate = round((ddos_count / total) * 100, 1) if total > 0 else 0.0
    
    # Threat level
    if attack_rate > 30.0:
        threat_level = "CRITICAL"
        threat_color = "#ef4444"
    elif attack_rate >= 10.0:
        threat_level = "HIGH"
        threat_color = "#f97316"
    elif ddos_count > 0:
        threat_level = "ELEVATED"
        threat_color = "#eab308"
    else:
        threat_level = "NORMAL"
        threat_color = "#22c55e"
        
    # Scatter data (first 100 points)
    scatter_data = []
    for i in range(min(120, total)):
        scatter_data.append({
            "x": round(float(clean_df.iloc[i].get("Flow Duration", 0)) / 1000.0, 2), # ms
            "y": round(float(clean_df.iloc[i].get("Flow Packets/s", 0)), 1),
            "label": labels[i],
            "conf": confidences[i]
        })
        
    # Table rows (first 100 rows)
    table_rows = []
    for i in range(min(100, total)):
        table_rows.append({
            "id": i + 1,
            "verdict": labels[i],
            "confidence": confidences[i],
            "duration": round(float(clean_df.iloc[i].get("Flow Duration", 0)), 1),
            "packets_s": round(float(clean_df.iloc[i].get("Flow Packets/s", 0)), 1),
            "bytes_s": round(float(clean_df.iloc[i].get("Flow Bytes/s", 0)), 1),
            "syn_flags": int(clean_df.iloc[i].get("SYN Flag Count", 0)),
            "ack_flags": int(clean_df.iloc[i].get("ACK Flag Count", 0))
        })
        
    result_df = clean_df.copy()
    result_df.insert(0, "Prediction", labels)
    result_df.insert(1, "Confidence (%)", confidences)
    
    return {
        "total": total,
        "ddos_count": ddos_count,
        "normal_count": normal_count,
        "attack_rate": attack_rate,
        "threat_level": threat_level,
        "threat_color": threat_color,
        "scatter_data": scatter_data,
        "table_rows": table_rows,
        "result_df": result_df
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    global last_analyzed_df
    model_name = request.form.get("model", "Random Forest")
    sample_type = request.form.get("sample", "")
    
    df = None
    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Failed to parse CSV file: {str(e)}"}), 400
    elif sample_type:
        sample_map = {
            "attack": "sample_traffic_attack.csv",
            "mixed": "sample_traffic_mixed.csv",
            "normal": "sample_traffic_normal.csv"
        }
        filename = sample_map.get(sample_type)
        if filename:
            path = os.path.join(DATA_DIR, filename)
            if os.path.exists(path):
                df = pd.read_csv(path)
                
    if df is None:
        return jsonify({"error": "No valid data or sample provided."}), 400
        
    try:
        results = preprocess_and_predict(df, model_name=model_name)
        last_analyzed_df = results.pop("result_df")
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Inference pipeline error: {str(e)}"}), 500

@app.route("/api/stream/sample", methods=["GET"])
def stream_sample():
    sample_path = os.path.join(DATA_DIR, "sample_traffic_mixed.csv")
    if not os.path.exists(sample_path):
        return jsonify({"error": "Sample data not found"}), 404
        
    df = pd.read_csv(sample_path)
    res = preprocess_and_predict(df, model_name="Random Forest")
    return jsonify({
        "packets": res["table_rows"]
    })

@app.route("/api/benchmarks", methods=["GET"])
def benchmarks():
    return jsonify({
        "metrics": model_metrics,
        "feature_count": len(feature_columns)
    })

@app.route("/api/charts/<filename>", methods=["GET"])
def get_chart(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

@app.route("/api/export", methods=["GET"])
def export():
    global last_analyzed_df
    if last_analyzed_df is None:
        return "No analysis data available to export.", 400
        
    output = io.BytesIO()
    last_analyzed_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="ddos_inspection_report.csv"
    )

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  [+] DDOS TRAFFIC CLASSIFIER & SOC DASHBOARD (FLASK SERVER) [+]")
    print("  URL: http://localhost:5000")
    print("="*70 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
