# 🧠 CFPB Complaint Intelligence System

## 📌 Overview

This project builds an **end-to-end NLP pipeline** for analyzing CFPB consumer complaint narratives and transforming them into **actionable risk signals**.

The system performs:
- Complaint classification
- Risk-based prioritization
- Explainable predictions

👉 Goal: help financial institutions and regulators **identify high-risk complaints early and allocate resources efficiently**

---

## 🎯 Business Problem

Financial institutions receive thousands of complaints daily.

Challenges:
- Large volume of unstructured text
- Hard to identify urgent/high-risk complaints
- Limited explainability for decision-making

This project solves:

> **Which complaints should be prioritized for immediate attention and why?**

---

## 🧠 System Design

Pipeline:

1. Data ingestion (CFPB dataset)
2. Text preprocessing
3. Feature extraction (TF-IDF / embeddings)
4. Model training
5. Risk scoring (calibrated probabilities)
6. Explainability layer
7. Triage output generation

---

## 📊 Dataset

- Source: CFPB Consumer Complaint Database
- Type: Unstructured complaint narratives
- Target Variables:
  - Complaint category (multiclass)
  - Complaint severity (binary: high vs low risk)

---

## 🧪 Modeling Approach

### 🔹 Baseline Model
- TF-IDF + Logistic Regression

### 🔹 Advanced Model
- Sentence embeddings + Gradient Boosting / XGBoost

### 🔹 Outputs
- Classification label
- Probability score
- Calibrated risk score

---

## 🎯 Target Definition

### Multiclass Task
- Predict complaint category / issue

### Binary Task (Critical)
- High Risk (urgent regulatory concern)
- Low Risk (standard complaint)

---

## 📈 Evaluation Metrics

- Accuracy
- Macro F1 Score
- Precision / Recall (High-Risk class)
- ROC-AUC
- PR-AUC (for imbalance)
- Calibration curves

---

## 📊 Results (Example)

| Model | ROC-AUC | F1 Score | Precision (High Risk) |
|------|--------|----------|-----------------------|
| Logistic Regression | ~0.78 | 0.72 | 0.70 |
| Advanced Model | 🔥 ~0.85 | 🔥 0.79 | 🔥 0.77 |

---

## ⚡ Risk Scoring

The system outputs a **priority score (0–1)**:

- 0.0 → Low priority
- 1.0 → High priority

This is based on:
- predicted probability
- calibrated confidence

---

## 🔍 Explainability

We use:
- SHAP values
- Top contributing words/phrases

Example:

Complaint:
> "Bank charged unauthorized fees repeatedly"

Prediction:
- Risk: High (0.87)
- Key drivers:
  - "unauthorized"
  - "fees"
  - "charged"

---

## 📊 Outputs

### Figures
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Feature importance
- Complaint distribution

### Metrics
- Model evaluation CSVs
- Risk score distributions

---

## 📦 Triage Output (FINAL PRODUCT)

Example output:

| Complaint ID | Category | Risk Score | Priority |
|-------------|---------|------------|---------|
| 12345 | Credit Card | 0.91 | 🔴 High |
| 12346 | Mortgage | 0.42 | 🟡 Medium |
| 12347 | Loan | 0.12 | 🟢 Low |

---

## 📁 Project Structure
