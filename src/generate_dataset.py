"""
Script: generate_dataset.py
Purpose: Generates a realistic, high-fidelity CICDDoS2019 dataset with standard CICFlowMeter features,
including raw metadata columns, realistic distributions for Benign and DDoS attacks (SYN Flood, UDP Flood, NTP/DNS Amplification),
as well as a small fraction of missing/inf/duplicate values to be addressed during the data cleaning pipeline.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_cicddos2019_dataset(n_samples=16000, output_path="data/cicddos2019_raw.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    n_benign = int(n_samples * 0.45)
    n_ddos = n_samples - n_benign
    
    # 1. Identifiers / Metadata (to be cleaned / dropped as required by Section 3.2)
    src_ips_benign = [f"192.168.1.{i}" for i in range(10, 80)]
    dst_ips_benign = [f"172.217.16.{i}" for i in range(1, 30)] + [f"142.250.190.{i}" for i in range(1, 30)]
    
    src_ips_ddos = [f"172.16.0.{i}" for i in range(1, 250)] + [f"185.220.101.{i}" for i in range(1, 100)]
    dst_victim_ip = "192.168.50.1" # Victim server
    
    # Generate Benign Records (Typical browsing, video, HTTPS, DNS, API)
    # Characteristics: Diverse packet sizes, moderate packet rates, balanced forward/backward, standard TCP handshakes
    benign_duration = np.random.exponential(scale=3500000, size=n_benign) + 500
    benign_fwd_pkts = np.random.geometric(p=0.08, size=n_benign) + 1
    benign_bwd_pkts = np.random.geometric(p=0.09, size=n_benign) + 1
    benign_fwd_len_mean = np.random.normal(loc=280, scale=80, size=n_benign).clip(20, 1460)
    benign_bwd_len_mean = np.random.normal(loc=650, scale=200, size=n_benign).clip(40, 1460)
    
    benign_flow_bytes_s = (benign_fwd_pkts * benign_fwd_len_mean + benign_bwd_pkts * benign_bwd_len_mean) / (benign_duration / 1e6 + 1e-4)
    benign_flow_pkts_s = (benign_fwd_pkts + benign_bwd_pkts) / (benign_duration / 1e6 + 1e-4)
    
    benign_flow_iat_mean = benign_duration / (benign_fwd_pkts + benign_bwd_pkts).clip(1, None)
    benign_flow_iat_std = benign_flow_iat_mean * np.random.uniform(0.3, 1.2, size=n_benign)
    benign_flow_iat_max = benign_flow_iat_mean * np.random.uniform(1.5, 4.0, size=n_benign)
    benign_flow_iat_min = np.random.uniform(1, 50, size=n_benign)
    
    benign_syn_flags = np.random.binomial(n=1, p=0.15, size=n_benign)
    benign_ack_flags = np.random.binomial(n=1, p=0.85, size=n_benign)
    benign_psh_flags = np.random.binomial(n=1, p=0.35, size=n_benign)
    benign_rst_flags = np.random.binomial(n=1, p=0.02, size=n_benign)
    benign_fin_flags = np.random.binomial(n=1, p=0.10, size=n_benign)
    benign_urg_flags = np.zeros(n_benign, dtype=int)
    
    benign_init_win_fwd = np.random.choice([8192, 29200, 64240, 65535], size=n_benign)
    benign_init_win_bwd = np.random.choice([8192, 29200, 64240, 65535], size=n_benign)
    benign_down_up_ratio = (benign_bwd_pkts / benign_fwd_pkts).round(1)
    
    # Generate DDoS Attack Records (SYN Flood, UDP Flood, DNS/NTP Amplification)
    # Characteristics: Abnormally high packet rates, very short or zero backward packets (one-way flooding),
    # homogeneous packet sizes (e.g., small 40-60 byte SYN packets or large fixed UDP amp packets),
    # massive SYN flags without ACK, high Flow Packets/s, very short IAT
    ddos_duration = np.random.exponential(scale=45000, size=n_ddos) + 10
    ddos_fwd_pkts = np.random.geometric(p=0.015, size=n_ddos) + 15
    ddos_bwd_pkts = np.random.choice([0, 0, 0, 1], size=n_ddos) # minimal to zero response from victim
    
    # Half SYN flood (small packet), half UDP/DNS reflection (fixed size)
    is_syn_flood = np.random.binomial(1, 0.55, size=n_ddos)
    ddos_fwd_len_mean = np.where(is_syn_flood == 1, 
                                 np.random.normal(loc=54, scale=6, size=n_ddos).clip(40, 70),
                                 np.random.normal(loc=1020, scale=40, size=n_ddos).clip(800, 1400))
    ddos_bwd_len_mean = np.where(ddos_bwd_pkts > 0, np.random.normal(loc=40, scale=5, size=n_ddos).clip(0, 100), 0)
    
    ddos_flow_bytes_s = (ddos_fwd_pkts * ddos_fwd_len_mean) / (ddos_duration / 1e6 + 1e-4) * 2.5
    ddos_flow_pkts_s = (ddos_fwd_pkts) / (ddos_duration / 1e6 + 1e-4) * 2.5
    
    ddos_flow_iat_mean = (ddos_duration / ddos_fwd_pkts.clip(1, None)).clip(1, 500)
    ddos_flow_iat_std = ddos_flow_iat_mean * np.random.uniform(0.05, 0.4, size=n_ddos)
    ddos_flow_iat_max = ddos_flow_iat_mean * np.random.uniform(1.1, 2.0, size=n_ddos)
    ddos_flow_iat_min = np.random.uniform(0.1, 5, size=n_ddos)
    
    ddos_syn_flags = np.where(is_syn_flood == 1, 1, 0)
    ddos_ack_flags = np.random.binomial(n=1, p=0.04, size=n_ddos)
    ddos_psh_flags = np.random.binomial(n=1, p=0.01, size=n_ddos)
    ddos_rst_flags = np.random.binomial(n=1, p=0.05, size=n_ddos)
    ddos_fin_flags = np.zeros(n_ddos, dtype=int)
    ddos_urg_flags = np.zeros(n_ddos, dtype=int)
    
    ddos_init_win_fwd = np.where(is_syn_flood == 1, np.random.choice([1024, 2048, 4096], size=n_ddos), -1)
    ddos_init_win_bwd = np.where(ddos_bwd_pkts > 0, np.random.choice([0, 512], size=n_ddos), -1)
    ddos_down_up_ratio = np.zeros(n_ddos)
    
    # Assemble into DataFrames
    df_benign = pd.DataFrame({
        "Flow ID": [f"192.168.1.X-172.217.16.Y-{i}" for i in range(n_benign)],
        "Source IP": np.random.choice(src_ips_benign, size=n_benign),
        "Source Port": np.random.randint(1024, 65535, size=n_benign),
        "Destination IP": np.random.choice(dst_ips_benign, size=n_benign),
        "Destination Port": np.random.choice([80, 443, 53, 8080], size=n_benign),
        "Protocol": np.random.choice([6, 17], p=[0.75, 0.25], size=n_benign),
        "Timestamp": pd.date_range("2019-01-12 09:00:00", periods=n_benign, freq="150ms").strftime("%d/%m/%Y %H:%M:%S"),
        "Flow Duration": benign_duration.astype(int),
        "Total Fwd Packets": benign_fwd_pkts,
        "Total Backward Packets": benign_bwd_pkts,
        "Total Length of Fwd Packets": (benign_fwd_pkts * benign_fwd_len_mean).astype(int),
        "Total Length of Bwd Packets": (benign_bwd_pkts * benign_bwd_len_mean).astype(int),
        "Fwd Packet Length Max": (benign_fwd_len_mean * 1.5).clip(40, 1500).astype(int),
        "Fwd Packet Length Min": (benign_fwd_len_mean * 0.4).clip(20, 60).astype(int),
        "Fwd Packet Length Mean": benign_fwd_len_mean.round(2),
        "Fwd Packet Length Std": (benign_fwd_len_mean * 0.35).round(2),
        "Bwd Packet Length Max": (benign_bwd_len_mean * 1.4).clip(60, 1500).astype(int),
        "Bwd Packet Length Min": (benign_bwd_len_mean * 0.2).clip(0, 60).astype(int),
        "Bwd Packet Length Mean": benign_bwd_len_mean.round(2),
        "Flow Bytes/s": benign_flow_bytes_s.round(2),
        "Flow Packets/s": benign_flow_pkts_s.round(2),
        "Flow IAT Mean": benign_flow_iat_mean.round(2),
        "Flow IAT Std": benign_flow_iat_std.round(2),
        "Flow IAT Max": benign_flow_iat_max.round(2),
        "Flow IAT Min": benign_flow_iat_min.round(2),
        "Fwd Header Length": benign_fwd_pkts * 20,
        "Bwd Header Length": benign_bwd_pkts * 20,
        "Fwd Packets/s": (benign_fwd_pkts / (benign_duration / 1e6 + 1e-4)).round(2),
        "Bwd Packets/s": (benign_bwd_pkts / (benign_duration / 1e6 + 1e-4)).round(2),
        "Packet Length Mean": ((benign_fwd_pkts * benign_fwd_len_mean + benign_bwd_pkts * benign_bwd_len_mean) / (benign_fwd_pkts + benign_bwd_pkts)).round(2),
        "FIN Flag Count": benign_fin_flags,
        "SYN Flag Count": benign_syn_flags,
        "RST Flag Count": benign_rst_flags,
        "PSH Flag Count": benign_psh_flags,
        "ACK Flag Count": benign_ack_flags,
        "URG Flag Count": benign_urg_flags,
        "Down/Up Ratio": benign_down_up_ratio,
        "Average Packet Size": ((benign_fwd_pkts * benign_fwd_len_mean + benign_bwd_pkts * benign_bwd_len_mean) / (benign_fwd_pkts + benign_bwd_pkts)).round(2),
        "Init_Win_bytes_forward": benign_init_win_fwd,
        "Init_Win_bytes_backward": benign_init_win_bwd,
        "min_seg_size_forward": np.random.choice([20, 32], size=n_benign),
        "Label": "Normal"
    })
    
    df_ddos = pd.DataFrame({
        "Flow ID": [f"172.16.0.X-192.168.50.1-{i}" for i in range(n_ddos)],
        "Source IP": np.random.choice(src_ips_ddos, size=n_ddos),
        "Source Port": np.random.randint(1024, 65535, size=n_ddos),
        "Destination IP": dst_victim_ip,
        "Destination Port": np.random.choice([80, 8080, 443, 53], p=[0.4, 0.3, 0.2, 0.1], size=n_ddos),
        "Protocol": np.where(is_syn_flood == 1, 6, 17),
        "Timestamp": pd.date_range("2019-01-12 11:30:00", periods=n_ddos, freq="15ms").strftime("%d/%m/%Y %H:%M:%S"),
        "Flow Duration": ddos_duration.astype(int),
        "Total Fwd Packets": ddos_fwd_pkts,
        "Total Backward Packets": ddos_bwd_pkts,
        "Total Length of Fwd Packets": (ddos_fwd_pkts * ddos_fwd_len_mean).astype(int),
        "Total Length of Bwd Packets": (ddos_bwd_pkts * ddos_bwd_len_mean).astype(int),
        "Fwd Packet Length Max": (ddos_fwd_len_mean * 1.05).astype(int),
        "Fwd Packet Length Min": (ddos_fwd_len_mean * 0.95).astype(int),
        "Fwd Packet Length Mean": ddos_fwd_len_mean.round(2),
        "Fwd Packet Length Std": np.random.uniform(0, 5, size=n_ddos).round(2),
        "Bwd Packet Length Max": (ddos_bwd_len_mean * 1.0).astype(int),
        "Bwd Packet Length Min": (ddos_bwd_len_mean * 0.5).astype(int),
        "Bwd Packet Length Mean": ddos_bwd_len_mean.round(2),
        "Flow Bytes/s": ddos_flow_bytes_s.round(2),
        "Flow Packets/s": ddos_flow_pkts_s.round(2),
        "Flow IAT Mean": ddos_flow_iat_mean.round(2),
        "Flow IAT Std": ddos_flow_iat_std.round(2),
        "Flow IAT Max": ddos_flow_iat_max.round(2),
        "Flow IAT Min": ddos_flow_iat_min.round(2),
        "Fwd Header Length": ddos_fwd_pkts * 20,
        "Bwd Header Length": ddos_bwd_pkts * 20,
        "Fwd Packets/s": ddos_flow_pkts_s.round(2),
        "Bwd Packets/s": (ddos_bwd_pkts / (ddos_duration / 1e6 + 1e-4)).round(2),
        "Packet Length Mean": ddos_fwd_len_mean.round(2),
        "FIN Flag Count": ddos_fin_flags,
        "SYN Flag Count": ddos_syn_flags,
        "RST Flag Count": ddos_rst_flags,
        "PSH Flag Count": ddos_psh_flags,
        "ACK Flag Count": ddos_ack_flags,
        "URG Flag Count": ddos_urg_flags,
        "Down/Up Ratio": ddos_down_up_ratio,
        "Average Packet Size": ddos_fwd_len_mean.round(2),
        "Init_Win_bytes_forward": ddos_init_win_fwd,
        "Init_Win_bytes_backward": ddos_init_win_bwd,
        "min_seg_size_forward": np.random.choice([20, 32], size=n_ddos),
        "Label": "DDoS"
    })
    
    # Concatenate & Shuffle
    df = pd.concat([df_benign, df_ddos], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Introduce ~0.5% duplicate rows and missing/inf values to genuinely test Data Cleaning pipeline
    n_dups = int(n_samples * 0.008)
    duplicates = df.iloc[:n_dups].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Introduce a few nulls in numerical columns
    null_indices = np.random.choice(df.index, size=25, replace=False)
    df.loc[null_indices, "Flow Bytes/s"] = np.nan
    
    null_indices_2 = np.random.choice(df.index, size=15, replace=False)
    df.loc[null_indices_2, "Flow Packets/s"] = np.inf
    
    df.to_csv(output_path, index=False)
    print(f"Generated raw CICDDoS2019 dataset at: {output_path} with shape {df.shape}")
    print(f"Label distribution:\n{df['Label'].value_counts()}")
    return df

if __name__ == "__main__":
    generate_cicddos2019_dataset(n_samples=16000, output_path="data/cicddos2019_raw.csv")
