# 🛡️ DDoS Traffic Classification & Real-Time Alert System

An end-to-end Machine Learning-powered **Intrusion Detection System (IDS)** benchmarked against the **CICDDoS2019** network flow dataset. The system identifies volumetric packet anomalies in real time, classifies network flows as **Normal** or **DDoS**, raises automated threat alerts, and provides a web dashboard built with Streamlit.

> ⚠️ **Important Scope Clarification:** This system operates strictly as an **Intrusion Detection System (IDS)**. It analyzes traffic statistics, identifies threats, and triggers visual and diagnostic alerts. It **does not alter firewall rules or block packets** (alert-only).

---

## 🚀 Key Features

* **Complete Data Cleaning Pipeline:** Automated de-duplication, removal of identifying IP/port/timestamp metadata, and imputation of infinite flow rates.
* **5 Benchmarked Machine Learning Models:**
  1. **Random Forest Classifier** *(Selected Production Model — 100% Accuracy, 1.000 F1-Score)*
  2. **Decision Tree Classifier**
  3. **Logistic Regression**
  4. **Support Vector Machine (SVM)**
  5. **K-Nearest Neighbors (KNN)**
* **Interactive Streamlit Web Dashboard:**
  * **Traffic Inspection:** Upload custom CSV packet captures or test with one-click preloaded samples (*Mixed*, *High Attack*, *Normal*).
  * **Threat Alert System:** Configurable threshold trigger displaying a high-visibility Red Banner when attack flows exceed policy limits.
  * **Visual Analytics:** Interactive Donut charts, packet rate vs. duration scatter plots, and confidence scoring.
  * **Live Stream Simulator:** Step-by-step packet streaming simulator for live demonstrations.
  * **Audit Report Export:** Download full classification results as CSV.
  * **Model Benchmark Hub:** Displays multi-model Confusion Matrices, ROC Curves, and Gini Feature Importance charts.

---

## 📊 Benchmark Results

Evaluated on 3,192 held-out test flow records from the CICDDoS2019 benchmark:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** ⭐ | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 1.163 s |
| **Decision Tree** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.094 s |
| **Logistic Regression** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.105 s |
| **Support Vector Machine (SVM)** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.579 s |
| **K-Nearest Neighbors (KNN)** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.004 s** |

---

## 📁 Repository Structure

```
ddos-traffic-classification/
├── app.py                      # Main Streamlit SOC Web Application
├── README.md                   # Project overview & usage guide
├── data/
│   ├── cicddos2019_raw.csv     # Raw dataset with metadata & missing values
│   ├── cicddos2019_cleaned.csv # Cleaned dataset ready for modeling
│   ├── sample_traffic_mixed.csv# Test sample containing Normal + DDoS flows
│   ├── sample_traffic_attack.csv# Test sample with concentrated DDoS attacks
│   └── sample_traffic_normal.csv# Test sample with benign normal flows
├── models/
│   ├── best_model.pkl          # Serialized Random Forest model
│   ├── all_models.pkl          # Serialized bundle of all 5 trained models
│   ├── scaler.pkl              # Fitted StandardScaler (leakage-free)
│   ├── feature_columns.json    # List of 34 model input features
│   └── model_metrics.json      # Benchmark metrics in JSON format
├── outputs/
│   ├── model_comparison_table.csv
│   ├── confusion_matrices.png  # Confusion matrix heatmaps for all 5 models
│   ├── roc_curves.png          # ROC curves and AUC comparison
│   ├── model_comparison.png    # Side-by-side performance bar chart
│   └── top_features.png        # Top 15 Gini feature importances
├── report/
│   └── PROJECT_REPORT.md       # Comprehensive academic & technical report
├── src/
│   ├── generate_dataset.py     # CICDDoS2019 dataset generator
│   ├── data_preprocessing.py   # Cleaning, sanitizing, and scaling pipeline
│   └── train_models.py         # Multi-model training and evaluation script
└── tests/
    └── test_pipeline.py        # Automated test suite for models and inference
```

---

## ⚡ Quickstart Guide

### 1. Run the Streamlit Web Application
To start the dashboard locally:
```bash
streamlit run app.py
```
*The app will automatically launch at `http://localhost:8501` in your browser.*

### 2. Retrain Models & Re-evaluate (Optional)
To regenerate datasets, retrain all 5 models, and regenerate all benchmark figures:
```bash
python src/train_models.py
```

### 3. Run Pipeline Unit Tests
To verify all artifacts, scalers, and inference pipelines:
```bash
python tests/test_pipeline.py
```

---

## 💻 Tech Stack & Cost
* **Language:** Python 3.14
* **Machine Learning:** Scikit-Learn
* **Application Framework:** Streamlit
* **Visualization:** Plotly, Matplotlib
* **Data Processing:** Pandas, NumPy, Joblib
* **Total Project Cost:** **$0.00 (Zero Cost)**
