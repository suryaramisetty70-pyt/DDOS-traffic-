"""
Module: train_models.py
Description: Trains and benchmarks all 5 Machine Learning algorithms on genuine CICDDoS2019 data:
1. Random Forest Classifier
2. Decision Tree Classifier
3. Logistic Regression
4. Support Vector Machine (LinearSVC with Probability Calibration)
5. K-Nearest Neighbors (KNN)
Generates evaluation metrics, comparison tables, and diagnostic plots.
"""

import os
import sys

# Ensure UTF-8 and CPU safety on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from data_preprocessing import run_preprocessing_pipeline

def train_and_evaluate_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Run Preprocessing Pipeline
    data_bundle = run_preprocessing_pipeline()
    
    X_train = data_bundle["X_train"]
    X_test = data_bundle["X_test"]
    y_train = data_bundle["y_train"]
    y_test = data_bundle["y_test"]
    feature_names = data_bundle["feature_names"]
    
    print(f"\n================ MODEL TRAINING & BENCHMARKING ================")
    print(f"Training partition: {X_train.shape[0]:,} flows | Testing partition: {X_test.shape[0]:,} flows")
    print(f"Input Features: {len(feature_names)}")
    
    # 5 Models specified in Section 3.4 of project plan
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": CalibratedClassifierCV(LinearSVC(dual=False, random_state=42), ensemble=False),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    }
    
    results = []
    trained_models = {}
    roc_data = {}
    confusion_matrices = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - start_time, 3)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Predict Proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = 0.5
            
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)
        
        metrics = {
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "Training Time (s)": train_time
        }
        results.append(metrics)
        trained_models[name] = model
        
        print(f"Results for {name}:")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall:    {rec * 100:.2f}%")
        print(f"  F1-Score:  {f1 * 100:.2f}%")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Time:      {train_time}s")
        
    df_results = pd.DataFrame(results)
    
    # Save comparison table
    results_csv_path = os.path.join(outputs_dir, "model_comparison_table.csv")
    df_results.to_csv(results_csv_path, index=False)
    
    results_json_path = os.path.join(models_dir, "model_metrics.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n================ FINAL MODEL COMPARISON ================")
    print(df_results.to_string(index=False))
    
    # Select Best Model
    best_row = df_results.sort_values(by=["F1-Score", "Accuracy"], ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n>> Best Performing Model Selected: {best_model_name} (F1: {best_row['F1-Score']}, Acc: {best_row['Accuracy']*100:.2f}%)")
    
    # Save Models
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    joblib.dump(trained_models, os.path.join(models_dir, "all_models.pkl"))
    print(f"Saved Best Model to: {best_model_path}")
    
    # ---------------- PLOTTING DIAGNOSTICS (Pure Matplotlib) ----------------
    print("\nGenerating charts and evaluation plots...")
    
    # 1. Confusion Matrix Grid Plot
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{name}\nAcc: {results[idx]['Accuracy']*100:.2f}%", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "DDoS"])
        ax.set_yticklabels(["Normal", "DDoS"] if idx == 0 else ["", ""])
        ax.set_xlabel("Predicted")
        if idx == 0:
            ax.set_ylabel("Actual")
            
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(outputs_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices to: {cm_path}")
    
    # 2. ROC Curves Comparison Plot
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for idx, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        plt.plot(fpr, tpr, lw=2, color=colors[idx % len(colors)], label=f"{name} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Baseline (AUC = 0.50)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(outputs_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves to: {roc_path}")
    
    # 3. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(df_results))
    width = 0.2
    plt.bar(x - 1.5 * width, df_results["Accuracy"], width, label="Accuracy", color="#1f77b4")
    plt.bar(x - 0.5 * width, df_results["Precision"], width, label="Precision", color="#2ca02c")
    plt.bar(x + 0.5 * width, df_results["Recall"], width, label="Recall", color="#ff7f0e")
    plt.bar(x + 1.5 * width, df_results["F1-Score"], width, label="F1-Score", color="#d62728")
    plt.xlabel("Machine Learning Algorithm", fontsize=11, fontweight="bold")
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title("Performance Metric Comparison Across All 5 ML Models", fontsize=13, fontweight="bold")
    plt.xticks(x, df_results["Model"], fontsize=10)
    plt.ylim(0.80, 1.02)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    comp_path = os.path.join(outputs_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved model comparison chart to: {comp_path}")
    
    # 4. Feature Importance (Random Forest)
    rf_model = trained_models.get("Random Forest", best_model)
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15] # Top 15
        top_feats = [feature_names[i] for i in indices]
        top_vals = importances[indices]
        
        plt.figure(figsize=(10, 6.5))
        y_pos = np.arange(len(top_feats))
        plt.barh(y_pos, top_vals[::-1], color="#0284c7", align="center")
        plt.yticks(y_pos, top_feats[::-1], fontsize=9)
        plt.xlabel("Gini Feature Importance Score", fontsize=11, fontweight="bold")
        plt.title("Top 15 Most Discriminative Flow Features (Random Forest)", fontsize=13, fontweight="bold")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        feat_path = os.path.join(outputs_dir, "top_features.png")
        plt.savefig(feat_path, dpi=300)
        plt.close()
        print(f"Saved top features chart to: {feat_path}")
        
    return df_results

if __name__ == "__main__":
    train_and_evaluate_all()
