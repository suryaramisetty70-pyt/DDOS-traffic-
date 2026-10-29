"""
Module: data_preprocessing.py
Description: Data cleaning, identifier removal, train/test splitting, and feature scaling
for authentic CICDDoS2019 benchmark telemetry.
Complies with ML best practices (split before scaling).
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

import json
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns that should be excluded if present (Identifiers / Metadata)
ID_COLUMNS = [
    "Unnamed: 0",
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
    "SimillarHTTP"
]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw network traffic data:
    1. Removes duplicate records.
    2. Drops non-generalizable ID and metadata columns.
    3. Handles infinite rates and missing values.
    """
    initial_rows = len(df)
    
    # Standardize column headers
    df.columns = df.columns.str.strip()
    
    # 1. Deduplication
    df = df.drop_duplicates()
    dedup_rows = len(df)
    print(f"[Data Cleaning] Removed {initial_rows - dedup_rows} duplicate rows.")
    
    # 2. Drop metadata / ID columns
    drop_cols = [col for col in ID_COLUMNS if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[Data Cleaning] Dropped metadata columns: {drop_cols}")
        
    # 3. Clean inf and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_count = int(df.isna().sum().sum())
    df = df.dropna().reset_index(drop=True)
    print(f"[Data Cleaning] Cleaned {nan_count} NaN/Inf values. Cleaned dimensions: {df.shape}")
    
    return df

def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepares data for machine learning:
    1. Binary encoding: Normal -> 0, DDoS -> 1.
    2. Strict Featurization Ordering: train/test split BEFORE scaling.
    3. Fits StandardScaler on X_train only, then transforms X_train and X_test.
    """
    if "Label" not in df.columns:
        raise ValueError("Target column 'Label' not found.")
        
    label_series = df["Label"].astype(str).str.strip().str.lower()
    y = np.where(label_series.isin(["normal", "benign"]), 0, 1)
    
    feature_df = df.drop(columns=["Label"])
    # Ensure all features are numeric float64
    feature_df = feature_df.select_dtypes(include=[np.number])
    feature_names = list(feature_df.columns)
    
    X = feature_df.to_numpy(dtype=np.float64)
    n_normal = int(np.sum(y == 0))
    n_ddos = int(np.sum(y == 1))
    print(f"[Data Prep] Usable Feature count: {len(feature_names)}")
    print(f"[Data Prep] Class Distribution: Normal (0): {n_normal:,} | DDoS (1): {n_ddos:,}")
    
    # Train-Test Split (Strict featurization ordering)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Fit scaler strictly on train split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "X_train_df": pd.DataFrame(X_train_raw, columns=feature_names),
        "X_test_df": pd.DataFrame(X_test_raw, columns=feature_names)
    }

def run_preprocessing_pipeline(
    raw_csv_path: str = "data/cicddos2019_raw.csv",
    output_dir: str = "data",
    models_dir: str = "models"
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_raw_path = os.path.join(base_dir, raw_csv_path) if not os.path.isabs(raw_csv_path) else raw_csv_path
    full_output_dir = os.path.join(base_dir, output_dir) if not os.path.isabs(output_dir) else output_dir
    full_models_dir = os.path.join(base_dir, models_dir) if not os.path.isabs(models_dir) else models_dir
    
    os.makedirs(full_output_dir, exist_ok=True)
    os.makedirs(full_models_dir, exist_ok=True)
    
    print(f"\n[+] Loading raw benchmark data from: {full_raw_path}")
    df_raw = pd.read_csv(full_raw_path)
    
    df_cleaned = clean_data(df_raw)
    cleaned_csv_path = os.path.join(full_output_dir, "cicddos2019_cleaned.csv")
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    print(f"[+] Saved sanitized real dataset to: {cleaned_csv_path}")
    
    data_bundle = prepare_data(df_cleaned)
    
    # Save Scaler and Feature schema
    scaler_path = os.path.join(full_models_dir, "scaler.pkl")
    features_path = os.path.join(full_models_dir, "feature_columns.json")
    
    joblib.dump(data_bundle["scaler"], scaler_path)
    with open(features_path, "w") as f:
        json.dump(data_bundle["feature_names"], f, indent=4)
        
    print(f"[+] Saved StandardScaler to: {scaler_path}")
    print(f"[+] Saved Feature Schema ({len(data_bundle['feature_names'])} features) to: {features_path}")
    
    # Generate balanced sample testing CSVs
    normal_subset = df_cleaned[df_cleaned["Label"] == "Normal"]
    ddos_subset = df_cleaned[df_cleaned["Label"] == "DDoS"]
    
    sample_normal = normal_subset.sample(n=min(200, len(normal_subset)), random_state=42)
    sample_attack = ddos_subset.sample(n=min(200, len(ddos_subset)), random_state=42)
    sample_mixed = pd.concat([
        normal_subset.sample(n=min(120, len(normal_subset)), random_state=123),
        ddos_subset.sample(n=min(180, len(ddos_subset)), random_state=123)
    ]).sample(frac=1.0, random_state=99).reset_index(drop=True)
    
    sample_mixed_path = os.path.join(full_output_dir, "sample_traffic_mixed.csv")
    sample_attack_path = os.path.join(full_output_dir, "sample_traffic_attack.csv")
    sample_normal_path = os.path.join(full_output_dir, "sample_traffic_normal.csv")
    
    sample_mixed.to_csv(sample_mixed_path, index=False)
    sample_attack.to_csv(sample_attack_path, index=False)
    sample_normal.to_csv(sample_normal_path, index=False)
    print(f"[+] Generated real test CSV files in data/: mixed (300), attack (200), normal (200).")
    
    # Also update user Downloads folder for seamless testing
    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    if os.path.exists(downloads_dir):
        shutil.copy(sample_mixed_path, os.path.join(downloads_dir, "test_mixed_traffic.csv"))
        shutil.copy(sample_attack_path, os.path.join(downloads_dir, "test_ddos_attack_traffic.csv"))
        shutil.copy(sample_normal_path, os.path.join(downloads_dir, "test_normal_clean_traffic.csv"))
        print(f"[+] Copied fresh real test CSV files directly to: {downloads_dir}")
        
    return data_bundle

if __name__ == "__main__":
    run_preprocessing_pipeline()
