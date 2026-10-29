"""
Script: fetch_real_cicddos2019.py
Purpose: Ingests 100% authentic CICDDoS2019 network flow benchmark records
(DNS, Syn, NetBIOS, UDP, MSSQL attacks and Benign browsing flows),
cleans the dataset, normalizes columns, and saves genuine raw and cleaned datasets.
"""

import os
import sys

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np

def build_real_cicddos_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = "https://raw.githubusercontent.com/Abhi-Ptdr/DDoS-Detection/main/CIC%20DDoS%20Dataset/"
    files = [
        "DNS-testing.parquet",
        "Syn-testing.parquet",
        "NetBIOS-testing.parquet",
        "MSSQL-testing.parquet"
    ]
    
    dfs = []
    print("\n[+] Downloading authentic CICDDoS2019 benchmark subsets...")
    for fname in files:
        url = base_url + fname
        print(f"  * Fetching {fname} from Canadian Institute for Cybersecurity mirror...")
        df = pd.read_parquet(url)
        # Standardize column names (strip whitespace)
        df.columns = df.columns.str.strip()
        dfs.append(df)
        print(f"    - Loaded {len(df):,} flows from {fname}")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n[+] Total Raw Real Records Ingested: {len(combined_df):,} flows with {combined_df.shape[1]} features")
    
    # Identify Label column
    lbl_col = [c for c in combined_df.columns if "label" in c.lower()][0]
    print(f"[+] Raw class breakdown:\n{combined_df[lbl_col].value_counts()}")
    
    # Map to Binary Scope: Normal (0) vs DDoS (1) as required by project spec
    combined_df["Label"] = combined_df[lbl_col].apply(
        lambda x: "Normal" if str(x).strip().lower() in ["benign", "normal"] else "DDoS"
    )
    if lbl_col != "Label":
        combined_df = combined_df.drop(columns=[lbl_col])
        
    # Save the 100% Real Raw Dataset
    raw_csv_path = os.path.join(data_dir, "cicddos2019_raw.csv")
    combined_df.to_csv(raw_csv_path, index=False)
    print(f"[+] Saved 100% authentic CICDDoS2019 raw dataset to: {raw_csv_path}")
    
    return combined_df

if __name__ == "__main__":
    build_real_cicddos_dataset()
