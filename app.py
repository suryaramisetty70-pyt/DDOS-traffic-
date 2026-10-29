"""
File: app.py
Project: DDoS Traffic Classification & Real-Time Alert System
Description: Interactive Streamlit Web Application featuring CSV traffic inspection,
real-time packet stream simulation, model benchmark comparison, and threat alerting.
Note: Strictly operates as an Intrusion Detection System (IDS) — alerts only, does not block.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="DDoS Traffic Classifier & Alert System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM STYLING -----------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #f8fafc;
        border-left: 5px solid #0284c7;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.92rem;
        color: #334155;
        margin-bottom: 1.5rem;
    }
    .alert-critical {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    .alert-safe {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- PATHS & ASSET LOADERS -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

@st.cache_resource
def load_models_and_artifacts():
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    all_models_path = os.path.join(MODELS_DIR, "all_models.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_columns.json")
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    best_model = joblib.load(best_model_path) if os.path.exists(best_model_path) else None
    all_models = joblib.load(all_models_path) if os.path.exists(all_models_path) else {}
    
    feature_cols = []
    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            feature_cols = json.load(f)
            
    metrics = []
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    return scaler, best_model, all_models, feature_cols, metrics

scaler, best_model, all_models, feature_cols, model_metrics = load_models_and_artifacts()

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("IDS Control Center")
    st.caption("CICDDoS2019 Traffic Classifier")
    
    st.markdown("---")
    navigation = st.radio(
        "Navigation",
        [
            "🔍 Traffic Analysis",
            "⚡ Live Traffic Simulator",
            "📊 Model Benchmarks",
            "ℹ️ System Architecture"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Model Configuration")
    
    model_choices = list(all_models.keys()) if all_models else ["Random Forest"]
    selected_model_name = st.selectbox(
        "Active Classifier",
        model_choices,
        index=0 if "Random Forest" in model_choices else 0,
        help="Select which trained ML model performs inference."
    )
    active_model = all_models.get(selected_model_name, best_model)
    
    st.markdown("---")
    st.subheader("Threat Policy")
    alert_threshold_pct = st.slider(
        "Alert Trigger Threshold (%)",
        min_value=1,
        max_value=50,
        value=10,
        help="Triggers critical red alert banner if DDoS attack flows exceed this percentage of total traffic."
    )
    
    st.markdown("---")
    st.caption("**Scope**: Intrusion Detection System (IDS)")
    st.caption("Detects and alerts only. No inline packet blocking.")

# ----------------- HEADER & IDS DISCLAIMER -----------------
col_title, col_status = st.columns([4, 1.2])
with col_title:
    st.markdown('<div class="main-title">🛡️ DDoS Traffic Classification & Threat Alert System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Machine Learning-Driven Intrusion Detection System (CICDDoS2019 Benchmark)</div>', unsafe_allow_html=True)

with col_status:
    st.markdown("""
    <div style="background:#e0f2fe; padding:8px 12px; border-radius:6px; border:1px solid #7dd3fc; text-align:center;">
        <span style="font-size:0.8rem; color:#0369a1; font-weight:600;">SYSTEM STATUS</span><br>
        <span style="font-size:0.95rem; color:#0284c7; font-weight:700;">🟢 ACTIVE MONITORING</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    📌 <strong>Operational Notice:</strong> This project is architected as an <strong>Intrusion Detection System (IDS)</strong>.
    It inspects network flow statistics, predicts potential flood anomalies, and raises visual/diagnostic alerts.
    In compliance with zero-cost demo safety constraints, it <em>does not alter firewall rules or execute active packet drops</em>.
</div>
""", unsafe_allow_html=True)

# ----------------- HELPER PREDICTION FUNCTION -----------------
def run_inference(df_input: pd.DataFrame, model, scaler, feature_cols):
    # Strip any ID/metadata columns if user uploaded raw format
    clean_df = df_input.copy()
    drop_candidates = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "Label"]
    clean_df = clean_df.drop(columns=[c for c in drop_candidates if c in clean_df.columns], errors="ignore")
    
    # Handle missing features if any
    for col in feature_cols:
        if col not in clean_df.columns:
            clean_df[col] = 0.0
            
    # Reorder exactly to model feature columns
    clean_df = clean_df[feature_cols].copy()
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Scale
    X_scaled = scaler.transform(clean_df.to_numpy(dtype=np.float64))
    
    # Predict
    preds = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_scaled)
        confidences = [round(float(probas[i, preds[i]]) * 100, 1) for i in range(len(preds))]
    else:
        confidences = [100.0] * len(preds)
        
    labels = ["Normal" if p == 0 else "DDoS" for p in preds]
    return labels, confidences, clean_df

# =====================================================================
# TAB 1: TRAFFIC ANALYSIS (CSV UPLOAD & INSPECTION)
# =====================================================================
if navigation == "🔍 Traffic Analysis":
    st.subheader("1. Network Traffic Ingestion & Batch Inspection")
    
    st.write("Upload a CSV file of captured network flow telemetry, or load one of the pre-configured sample datasets below:")
    
    # Sample Load Buttons
    c1, c2, c3 = st.columns(3)
    load_choice = None
    with c1:
        if st.button("📂 Load Mixed Traffic (Normal + DDoS)", use_container_width=True):
            load_choice = "sample_traffic_mixed.csv"
    with c2:
        if st.button("🚨 Load High Attack Traffic (DDoS Spike)", use_container_width=True):
            load_choice = "sample_traffic_attack.csv"
    with c3:
        if st.button("🟢 Load Normal Traffic (Clean Flow)", use_container_width=True):
            load_choice = "sample_traffic_normal.csv"
            
    uploaded_file = st.file_uploader("Or upload your own traffic CSV file:", type=["csv"])
    
    df_to_analyze = None
    source_name = ""
    
    if uploaded_file is not None:
        try:
            df_to_analyze = pd.read_csv(uploaded_file)
            source_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    elif load_choice:
        sample_path = os.path.join(DATA_DIR, load_choice)
        if os.path.exists(sample_path):
            df_to_analyze = pd.read_csv(sample_path)
            source_name = load_choice
            st.session_state["loaded_sample"] = load_choice
            st.session_state["loaded_df"] = df_to_analyze
        else:
            st.warning(f"Sample file {load_choice} not found. Please train models first.")
    elif "loaded_df" in st.session_state:
        df_to_analyze = st.session_state["loaded_df"]
        source_name = st.session_state.get("loaded_sample", "Cached Traffic")

    if df_to_analyze is not None:
        st.success(f"Loaded **{len(df_to_analyze)}** network traffic flow records from `{source_name}`.")
        
        with st.expander("👁️ Preview Raw Ingested Data (First 5 Rows)", expanded=False):
            st.dataframe(df_to_analyze.head(5), use_container_width=True)
            
        with st.spinner(f"Classifying network flows using {selected_model_name}..."):
            labels, confidences, clean_df = run_inference(df_to_analyze, active_model, scaler, feature_cols)
            
        # Analysis calculations
        total_flows = len(labels)
        ddos_count = sum(1 for l in labels if l == "DDoS")
        normal_count = total_flows - ddos_count
        attack_pct = round((ddos_count / total_flows) * 100, 1)
        
        st.markdown("---")
        st.subheader("2. Detection Results & Threat Assessment")
        
        # Threat Alert Evaluation
        is_critical = attack_pct >= alert_threshold_pct
        if is_critical:
            st.markdown(f"""
            <div class="alert-critical">
                <h3 style="color:#b91c1c; margin:0 0 8px 0; font-weight:800;">🚨 CRITICAL THREAT ALERT: High-Volume DDoS Attack Detected!</h3>
                <p style="color:#7f1d1d; margin:0; font-size:1.02rem;">
                    <strong>Alert Trigger:</strong> <strong>{ddos_count:,}</strong> flows classified as malicious DDoS attacks 
                    (<strong>{attack_pct}%</strong> of total traffic, exceeding safety threshold of {alert_threshold_pct}%).<br>
                    <strong>Recommended Action:</strong> Notify Network Security Admin, inspect victim IP, engage upstream ISP rate-limiting.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif ddos_count > 0:
            st.warning(f"⚠️ **Moderate Threat Alert:** Detected {ddos_count} DDoS attack flow(s) ({attack_pct}%). Below threshold of {alert_threshold_pct}%.")
        else:
            st.markdown(f"""
            <div class="alert-safe">
                <h3 style="color:#15803d; margin:0 0 8px 0; font-weight:700;">🟢 All Clear: Network Traffic Appears Normal</h3>
                <p style="color:#166534; margin:0; font-size:1.02rem;">
                    All {total_flows:,} inspected network flows conform to standard protocol behaviors. No abnormal flood patterns detected.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Flows Inspected", f"{total_flows:,}")
        m2.metric("Normal Traffic Flows", f"{normal_count:,}", delta=f"{100 - attack_pct:.1f}%")
        m3.metric("DDoS Attack Flows", f"{ddos_count:,}", delta=f"{attack_pct:.1f}%", delta_color="inverse")
        
        threat_level = "CRITICAL" if attack_pct > 25 else "ELEVATED" if attack_pct >= alert_threshold_pct else "LOW"
        m4.metric("Threat Level", threat_level)
        
        # Visual Charts
        st.markdown("#### 3. Flow Analytics & Visual Distribution")
        g1, g2 = st.columns([1, 1.3])
        
        with g1:
            # Donut chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Normal", "DDoS Attack"],
                values=[normal_count, ddos_count],
                hole=0.55,
                marker=dict(colors=["#22c55e", "#ef4444"])
            )])
            fig_pie.update_layout(
                title_text="Traffic Composition Breakdown",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with g2:
            # Packet Rate vs Flow Duration
            plot_df = pd.DataFrame({
                "Flow Duration (ms)": clean_df["Flow Duration"] / 1000.0 if "Flow Duration" in clean_df else np.random.rand(total_flows),
                "Flow Packets/s": clean_df["Flow Packets/s"] if "Flow Packets/s" in clean_df else np.random.rand(total_flows),
                "Classification": labels,
                "Confidence (%)": confidences
            })
            fig_scatter = px.scatter(
                plot_df,
                x="Flow Duration (ms)",
                y="Flow Packets/s",
                color="Classification",
                color_discrete_map={"Normal": "#22c55e", "DDoS": "#ef4444"},
                hover_data=["Confidence (%)"],
                title="Flow Duration vs. Packet Transmission Rate"
            )
            fig_scatter.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        # Detailed Prediction Table
        st.markdown("#### 4. Individual Flow Inspection Records")
        results_table = clean_df.copy()
        results_table.insert(0, "Prediction", labels)
        results_table.insert(1, "Confidence", [f"{c:.1f}%" for c in confidences])
        
        def highlight_attack(val):
            if val == "DDoS":
                return "background-color: #fecaca; color: #991b1b; font-weight: bold;"
            elif val == "Normal":
                return "background-color: #bbf7d0; color: #166534; font-weight: bold;"
            return ""

        styled_table = results_table.head(100).style.map(highlight_attack, subset=["Prediction"])
        st.dataframe(styled_table, use_container_width=True)
        if len(results_table) > 100:
            st.caption(f"Showing first 100 of {len(results_table)} rows.")
            
        # Download Report Section
        st.markdown("#### 5. Export Inspection Telemetry")
        csv_download = results_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Classification Report (CSV)",
            data=csv_download,
            file_name=f"ddos_inspection_report_{int(time.time())}.csv",
            mime="text/csv"
        )
    else:
        st.info("👆 Please upload a CSV traffic capture or click one of the sample buttons above to begin analysis.")

# =====================================================================
# TAB 2: LIVE TRAFFIC SIMULATOR
# =====================================================================
elif navigation == "⚡ Live Traffic Simulator":
    st.subheader("Real-Time Network Stream Simulation")
    st.write("""
    This simulator models real-time packet stream inspection. It replays network flows line-by-line,
    performing sub-millisecond inference and updating live counters and alert indicators.
    """)
    
    sim_col1, sim_col2 = st.columns([2, 1])
    with sim_col1:
        stream_speed = st.slider("Stream Interval (seconds per packet):", 0.05, 0.8, 0.2, step=0.05)
    with sim_col2:
        total_stream_packets = st.number_input("Number of Packets to Stream:", min_value=10, max_value=200, value=50, step=10)
        
    start_sim = st.button("🚀 Start Live Traffic Stream", type="primary")
    
    if start_sim:
        sample_path = os.path.join(DATA_DIR, "sample_traffic_mixed.csv")
        if not os.path.exists(sample_path):
            st.error("Sample dataset not found. Please train models first.")
        else:
            df_sim_pool = pd.read_csv(sample_path)
            df_sim_stream = df_sim_pool.sample(n=min(total_stream_packets, len(df_sim_pool)), replace=True).reset_index(drop=True)
            
            # Streaming UI Placeholders
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
            p_metric1 = kpi_c1.empty()
            p_metric2 = kpi_c2.empty()
            p_metric3 = kpi_c3.empty()
            
            alert_placeholder = st.empty()
            table_placeholder = st.empty()
            
            sim_records = []
            live_ddos = 0
            live_normal = 0
            
            for i, row in df_sim_stream.iterrows():
                row_df = pd.DataFrame([row])
                lbl, conf, _ = run_inference(row_df, active_model, scaler, feature_cols)
                verdict = lbl[0]
                conf_val = conf[0]
                
                if verdict == "DDoS":
                    live_ddos += 1
                else:
                    live_normal += 1
                    
                sim_records.append({
                    "Packet #": i + 1,
                    "Verdict": "🚨 DDoS Attack" if verdict == "DDoS" else "🟢 Normal",
                    "Confidence": f"{conf_val}%",
                    "Flow Duration": f"{row.get('Flow Duration', 0)} us",
                    "Packets/s": f"{row.get('Flow Packets/s', 0)}"
                })
                
                # Update metrics
                curr_pct = round((live_ddos / (i + 1)) * 100, 1)
                p_metric1.metric("Packets Processed", f"{i + 1} / {total_stream_packets}")
                p_metric2.metric("Normal Packets", f"{live_normal}")
                p_metric3.metric("DDoS Detected", f"{live_ddos}", delta=f"{curr_pct}% Attack Rate", delta_color="inverse")
                
                progress_bar.progress((i + 1) / total_stream_packets)
                status_text.text(f"Inspecting packet #{i + 1}... Status: {verdict}")
                
                if curr_pct >= alert_threshold_pct and live_ddos > 1:
                    alert_placeholder.error(f"🚨 LIVE ALERT: High DDoS rate detected ({curr_pct}% of stream)!")
                else:
                    alert_placeholder.empty()
                    
                # Update live log table with recent 8 packets
                recent_df = pd.DataFrame(sim_records[::-1][:8])
                table_placeholder.dataframe(recent_df, use_container_width=True)
                
                time.sleep(stream_speed)
                
            status_text.success("✅ Real-Time Stream Simulation Completed.")

# =====================================================================
# TAB 3: MODEL BENCHMARKS & EVALUATION
# =====================================================================
elif navigation == "📊 Model Benchmarks":
    st.subheader("Model Evaluation & Algorithm Comparison")
    st.write("All 5 machine learning models were rigorously evaluated on the identical test split using 5-fold standard metrics:")
    
    # Metrics Table
    comp_table_path = os.path.join(OUTPUTS_DIR, "model_comparison_table.csv")
    if os.path.exists(comp_table_path):
        df_bench = pd.read_csv(comp_table_path)
        
        # Display Styled Metrics Table
        st.dataframe(
            df_bench.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#bbf7d0")
                          .highlight_min(subset=["Training Time (s)"], color="#dbeafe"),
            use_container_width=True
        )
    elif model_metrics:
        st.dataframe(pd.DataFrame(model_metrics), use_container_width=True)
    else:
        st.info("Benchmarks will appear after running `train_models.py`.")
        
    st.markdown("---")
    st.markdown("#### Evaluation Artifacts & Diagnostic Charts")
    
    tab_cm, tab_roc, tab_comp, tab_feat = st.tabs([
        "Confusion Matrices",
        "ROC Curves",
        "Comparative Performance",
        "Top Features Importance"
    ])
    
    with tab_cm:
        cm_img = os.path.join(OUTPUTS_DIR, "confusion_matrices.png")
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix Grid across all 5 Classifiers", use_container_width=True)
            
    with tab_roc:
        roc_img = os.path.join(OUTPUTS_DIR, "roc_curves.png")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="Multi-Model ROC Curves and Area Under Curve (AUC)", use_container_width=True)
            
    with tab_comp:
        comp_img = os.path.join(OUTPUTS_DIR, "model_comparison.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Side-by-Side Metric Comparison Bar Chart", use_container_width=True)
            
    with tab_feat:
        feat_img = os.path.join(OUTPUTS_DIR, "top_features.png")
        if os.path.exists(feat_img):
            st.image(feat_img, caption="Top 15 Most Discriminative Features (Gini Importance)", use_container_width=True)

# =====================================================================
# TAB 4: SYSTEM ARCHITECTURE & REPORT
# =====================================================================
elif navigation == "ℹ️ System Architecture":
    st.subheader("System Architecture & Project Specification")
    
    st.markdown("""
    ### 1. Intrusion Detection Architecture (Alert Only)
    ```
    +-------------------------+
    | Network Traffic Capture |
    |    (PCAP / CSV Log)     |
    +------------+------------+
                 |
                 v
    +-------------------------+
    | Feature Extraction &    |
    | Cleaning (Drop IDs, IPs)|
    +------------+------------+
                 |
                 v
    +-------------------------+
    | ML Inference Engine     |
    | (Random Forest / 5 Mdl) |
    +------------+------------+
                 |
                 v
    +-------------------------+
    | Alerting Engine         |
    |  * Red Banner Alert     |  <=== Strictly Alert-Only (IDS)
    |  * Dashboard Visualizer |       (No packet drop / blocking)
    |  * Diagnostic Export    |
    +-------------------------+
    ```
    
    ### 2. Binary Classification Scope
    * **Normal (Class 0):** Standard HTTP, HTTPS, DNS, and benign web traffic.
    * **DDoS Attack (Class 1):** Volumetric anomalies including SYN Floods, UDP Floods, and amplification attacks.
    * *Note:* Specific sub-attack classification is deferred to future work as per project plan.
    
    ### 3. Key Differences: IDS (This Project) vs IPS
    | Parameter | Intrusion Detection System (IDS) | Intrusion Prevention System (IPS) |
    | :--- | :--- | :--- |
    | **Functionality** | Passively detects & raises alerts | Actively blocks & drops malicious packets |
    | **Network Placement** | Out-of-band / TAP / Log replay | In-line with live traffic flow |
    | **Risk Profile** | Zero risk of dropping legitimate packets | Risk of false-positive service disruption |
    | **Implementation** | Pure ML + Alert Dashboard (Streamlit) | Kernel hooks / iptables / eBPF rules |
    
    ### 4. Zero-Cost Technology Stack
    * **Algorithms:** Random Forest, Decision Tree, Logistic Regression, SVM, KNN (Scikit-Learn)
    * **Language & Runtime:** Python 3.14, NumPy, Pandas, Joblib
    * **Visualization & App:** Streamlit, Plotly, Matplotlib
    * **Cloud / Hosting:** Streamlit Community Cloud & GitHub (100% Free)
    """)

# ----------------- FOOTER -----------------
st.markdown("---")
st.caption("🛡️ DDoS Traffic Classification Project | Zero-Cost Machine Learning Demonstration | Developed with Python & Streamlit")
