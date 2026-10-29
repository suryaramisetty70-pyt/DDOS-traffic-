"""
Unit test to verify models, scaler, data files, and inference logic.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np

def test_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")
    
    assert os.path.exists(os.path.join(models_dir, "best_model.pkl")), "best_model.pkl missing"
    assert os.path.exists(os.path.join(models_dir, "scaler.pkl")), "scaler.pkl missing"
    assert os.path.exists(os.path.join(models_dir, "feature_columns.json")), "feature_columns.json missing"
    assert os.path.exists(os.path.join(models_dir, "all_models.pkl")), "all_models.pkl missing"
    assert os.path.exists(os.path.join(outputs_dir, "model_comparison_table.csv")), "comparison table missing"
    assert os.path.exists(os.path.join(outputs_dir, "confusion_matrices.png")), "confusion matrices image missing"
    assert os.path.exists(os.path.join(outputs_dir, "roc_curves.png")), "roc curves image missing"
    assert os.path.exists(os.path.join(outputs_dir, "top_features.png")), "top features image missing"
    
    # Load model & test inference
    model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_columns.json"), "r") as f:
        features = json.load(f)
        
    df_sample = pd.read_csv(os.path.join(data_dir, "sample_traffic_mixed.csv"))
    clean_features = df_sample[[c for c in features if c in df_sample.columns]]
    X_scaled = scaler.transform(clean_features.to_numpy(dtype=np.float64))
    preds = model.predict(X_scaled)
    
    print(f"[TEST PASSED] Evaluated {len(preds)} test rows.")
    print(f"[TEST PASSED] Predicted Normal: {sum(preds == 0)}, Predicted DDoS: {sum(preds == 1)}")

if __name__ == "__main__":
    test_pipeline()
