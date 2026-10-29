# DDoS Traffic Classification and Real-Time Alert System
## Comprehensive Technical Project Report

**Author:** R. Jayasurya  
**Date:** October 2026  
**Scope:** Intrusion Detection System (IDS) — Passive Detection & Alerting  
**Dataset:** Official Canadian Institute for Cybersecurity (CICDDoS2019)  
**Repository:** `ddos-traffic-classification`  

---

## Executive Summary
Distributed Denial of Service (DDoS) attacks present an acute hazard to enterprise networks, web applications, and critical internet services. Conventional perimeter defense mechanisms predominantly rely on static threshold rules (e.g., volume rate triggers), which are notoriously brittle against dynamic, heterogeneous packet floods and prone to unacceptable false-positive rates. 

This project delivers a production-grade machine learning **Intrusion Detection System (IDS)** trained and evaluated on **17,918 authentic network flow records** from the university benchmark **CICDDoS2019** dataset (encompassing DNS amplification, Syn floods, NetBIOS reflection, MSSQL attacks, and legitimate Benign browsing flows). The system performs binary traffic classification (**Normal** vs. **DDoS**) across 77 discriminative network flow features. Five machine learning classifiers—**Random Forest, Decision Tree, Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN)**—were benchmarked under strict train-test separation (split before scaling). The **Random Forest** classifier achieved optimal classification performance (**99.86% accuracy, 0.9988 F1-score, 1.0000 ROC-AUC**) with millisecond inference latency. 

A high-performance **Cyber Security Operations Center (SOC) web application** running on Flask provides batch CSV analysis, interactive visual analytics (Chart.js), live stream packet simulation, and instant threat alerting.

---

## 1. Introduction & Problem Statement
A DDoS attack orchestrates a distributed network of compromised machines (botnets) to flood a targeted server with fake requests or volumetric packet streams, exhausting bandwidth, memory, or CPU resources until legitimate users are locked out.

### 1.1 Limitations of Rule-Based Intrusion Detection
Legacy intrusion detection relies on fixed heuristics, such as *"trigger alert if incoming requests exceed 5,000 packets per second"*. Attackers evade these thresholds by:
* Distributing malicious streams across thousands of spoofed IPs.
* Launching multi-vector attacks blending small-packet SYN floods with reflection amplification.
* Varying transmission timing and flow inter-arrival times (IAT).

Static threshold rules frequently miss distributed low-rate attacks or trigger costly false alarms during legitimate traffic spikes (e.g., promotional sales or breaking news events). Machine learning overcomes these shortcomings by learning complex, multidimensional statistical signatures directly from network flow telemetry.

### 1.2 System Scope: Detection & Alerting (IDS) vs. Mitigation (IPS)
It is critical to establish the architectural boundary of this project:
* **Project Role (IDS):** The developed system functions strictly as a **passive Intrusion Detection System**. It ingests flow logs, extracts statistical features, identifies anomalies, and generates diagnostic visual and system alerts.
* **Non-Blocking Guarantee:** The system does **not** alter packet routing tables, drop packets in-line, or inject firewall/iptables rules. Passive monitoring guarantees zero risk of inadvertent service denial to legitimate users caused by false-positive drops.

---

## 2. System Architecture & Flow
The system follows a modular, decoupled four-stage architecture:

```
+-------------------------------------------------------------+
|                 1. Network Ingestion Layer                  |
|    - Authentic CICDDoS2019 Benchmark Flow Telemetry         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                2. Preprocessing & Cleaning                  |
|    - De-duplication (597 duplicate records stripped)        |
|    - Drop IDs & Metadata (Source/Dest IP, Ports, Timestamps)|
|    - Sanitize NaN and Infinite Flow Rates                   |
|    - Standard Scaling (Fitted on Train Set Only)            |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                   3. ML Inference Engine                    |
|    - Active Model: Random Forest (or DT, LR, SVM, KNN)      |
|    - Binary Classification: Normal (0) vs. DDoS (1)         |
|    - Probability Estimation & Confidence Scoring            |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               4. Alert & Presentation Layer                 |
|    - Cyber SOC Dashboard (Flask & Tailwind CSS)             |
|    - Critical Red Banner Alert (Threshold-Triggered >10%)   |
|    - Interactive Distribution Donut & Scatter Plots         |
|    - Live Packet Stream Simulation                          |
|    - Downloadable Threat Inspection Report (CSV)            |
+------------------------------+------------------------------+
```

---

## 3. Class Diagram & Binary Scope

### 3.1 Binary Focus
In alignment with the initial project scope, the system executes **Binary Classification**:
* **Class 0 — Normal:** Legitimate HTTP, HTTPS, API calls, and standard DNS queries.
* **Class 1 — DDoS:** Volumetric flood traffic (SYN floods, UDP floods, amplification attacks).

*(Decomposition into multi-class sub-attacks—such as distinguishing NetBIOS from LDAP reflection—is formally cataloged as future work).*

---

## 4. Dataset & Data Preprocessing

### 4.1 Dataset Benchmark (CICDDoS2019)
The project utilizes 100% genuine network telemetry records from the Canadian Institute for Cybersecurity **CICDDoS2019** benchmark dataset, generated using CICFlowMeter network traffic profilers. Subsets include:
* `DNS-testing.parquet` (DrDoS_DNS + Benign)
* `Syn-testing.parquet` (Syn flood + Benign)
* `NetBIOS-testing.parquet` (DrDoS_NetBIOS + Benign)
* `MSSQL-testing.parquet` (DrDoS_MSSQL + Benign)

### 4.2 Step-by-Step Data Cleaning Pipeline
1. **Deduplication:** 597 duplicate flow records were identified and dropped to eliminate sampling bias.
2. **Identifier & Metadata Stripping:** All non-generalizable network metadata columns were stripped:
   * `Flow ID`
   * `Source IP` & `Source Port`
   * `Destination IP` & `Destination Port`
   * `Timestamp`
3. **Handling Anomalies (NaN & Infinite values):** Zero corrupted rows remained after sanitization. Cleaned dataset contains **17,321 valid flows** across **77 features**.
4. **Strict Featurization Ordering:** Stratified train-test splitting (80% training = 13,856 samples; 20% testing = 3,465 samples) was performed **prior** to feature scaling to completely eliminate data leakage.

---

## 5. Model Training & Comparative Results

### 5.1 Evaluated Algorithms
Five machine learning algorithms were trained and benchmarked under identical conditions:
1. **Random Forest Classifier:** Ensemble of 100 decision trees with feature sub-sampling.
2. **Decision Tree Classifier:** Single CART tree optimized with maximum depth regularization.
3. **Logistic Regression:** Linear probabilistic classifier with L2 regularization ($C=1.0$).
4. **Support Vector Machine (SVM):** LinearSVC with Calibrated probability estimator.
5. **K-Nearest Neighbors (KNN):** Instance-based metric classifier with Euclidean distance ($k=5$).

### 5.2 Performance Comparison Table
Evaluation was performed on the held-out test split of 3,465 records:

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** ⭐ *(Best)* | **99.86%** | **0.9991** | **0.9986** | **0.9988** | **1.0000** | 2.103 s |
| **Decision Tree** | 99.83% | 0.9991 | 0.9981 | 0.9986 | 0.9983 | 0.237 s |
| **Logistic Regression** | 99.60% | 0.9939 | 0.9995 | 0.9967 | 0.9996 | 0.285 s |
| **Support Vector Machine (SVM)** | 99.57% | 0.9934 | 0.9995 | 0.9965 | 0.9997 | 7.869 s |
| **K-Nearest Neighbors (KNN)** | 99.57% | 0.9943 | 0.9986 | 0.9965 | 0.9985 | **0.004 s** |

### 5.3 Model Selection Rationale
**Random Forest** was selected as the **Primary Production Classifier** because:
* **Ensemble Robustness:** Achieves **99.86% accuracy** with near-zero false alarms (0.9991 precision).
* **Inference Speed:** Average batch prediction takes under 15 milliseconds for hundreds of flows.
* **Feature Importance Transparency:** Explains which network flow characteristics drive each verdict.

---

## 6. Cyber SOC Web Application Overview
The project features a dedicated **Security Operations Center (SOC)** web dashboard running on port 5000:
1. **Traffic Analysis Dashboard:** CSV drag & drop zone, one-click test sample loaders, dynamic Threat Red Alert Banner, donut distribution charts, and packet rate scatter plots.
2. **Live Traffic Stream Simulator:** Step-by-step packet streaming simulator with adjustable interval speed (200ms - 1000ms), live attack counters, and dynamic status verdicts.
3. **Model Benchmarks Hub:** Performance matrix for all 5 algorithms with embedded Confusion Matrices, ROC curves, and Top-15 Feature Importances.
4. **Architecture & Scope Guide:** Explains the zero-cost architecture, IDS non-blocking boundaries, and feature definitions.

---

## 7. Cost Breakdown ($0.00 Budget)
* **Language & Frameworks:** Python 3.14, Scikit-Learn, Flask
* **Visuals & UI:** Tailwind CSS, Chart.js, FontAwesome
* **Hardware/Hosting:** Localhost & GitHub ($0.00)

---

## 8. References
1. **Sharafaldin, I., Lashkari, A. H., Hakak, S., & Ghorbani, A. A.** (2019). *Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy.* IEEE 53rd International Carnahan Conference on Security Technology (ICCST), Chennai, India, pp. 1-8.
2. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825-2830.
3. **Canadian Institute for Cybersecurity (CIC).** (2019). *CICDDoS2019 Dataset Documentation and Feature Definitions.* University of New Brunswick.
4. **Mirkovic, J., & Reiher, P.** (2004). *A taxonomy of DDoS attack and DDoS defense mechanisms.* ACM SIGCOMM Computer Communication Review, 34(2), pp. 39-53.
