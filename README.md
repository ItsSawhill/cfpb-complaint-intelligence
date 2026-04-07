# CFPB Complaint Intelligence System

This repository is an end-to-end complaint intelligence workflow for CFPB complaint narratives. It supports regulatory triage by predicting complaint category, estimating whether a complaint is high risk, generating a calibrated risk score from 0 to 1, and surfacing explainable reasons for prioritization.

## Business Problem

Regulatory operations teams need a repeatable way to triage large volumes of complaint narratives. This system helps answer four questions:

1. What complaint category is this?
2. Is this complaint high risk or low risk?
3. What calibrated probability should we use as the risk score?
4. Why was this complaint marked as high priority?

## Current Training Pipeline

The pipeline lives in these modules:

- `src/complaint_intel/modeling/train_classifier.py`: multiclass product classification
- `src/complaint_intel/risk/high_risk.py`: explicit binary target definition
- `src/complaint_intel/risk/train_risk_model.py`: binary-risk training, calibration, and evaluation
- `src/complaint_intel/risk/score.py`: full-dataset scoring and explainability
- `src/complaint_intel/app/build_dashboard_data.py`: dashboard-ready merge step
- `src/complaint_intel/pipeline/train_system.py`: end-to-end orchestration

## Target Definition

The binary target is explicitly named `high_risk`.

`high_risk = 1` when at least one of these observable regulatory triage signals is present:

- `Consumer disputed? = Yes`
- `Timely response? = No`
- `Tags` contains a vulnerable consumer segment such as `Servicemember` or `Older American`

The existing multiclass task remains in place and still predicts the complaint `product` label from the cleaned CFPB dataset.

## Models

### Multiclass Task

- Baseline: TF-IDF + Logistic Regression
- Output: predicted complaint product plus top-1 confidence

### Binary High-Risk Task

- Baseline: word TF-IDF + Logistic Regression
- Advanced: word and character TF-IDF + calibrated Linear SVM
- Champion selection prefers better calibration while retaining strong discrimination

### Model Comparison

| Task | Model | Features | Output | Current role |
| --- | --- | --- | --- | --- |
| Multiclass category | Logistic Regression | Word TF-IDF | `predicted_product`, `prediction_confidence` | Baseline classifier kept in production path |
| Binary risk | Logistic Regression | Word TF-IDF | `baseline_risk_probability` | Baseline, strongest discrimination in current sampled run |
| Binary risk | Calibrated Linear SVM | Word + character TF-IDF | `advanced_risk_probability` | Advanced calibrated model, current champion by Brier score |

## Results

Each training run writes machine-readable metrics so the reported results stay attached to the exact run that produced them.

Saved multiclass outputs:

- `outputs/metrics/multiclass_metrics.json`
- `outputs/metrics/multiclass_metrics.txt`
- `outputs/metrics/multiclass_predictions.parquet`
- `outputs/figures/multiclass_confusion_matrix.png`

Saved binary-risk outputs:

- `outputs/metrics/binary_risk_metrics.json`
- `outputs/metrics/binary_risk_metrics.csv`
- `outputs/metrics/binary_risk_predictions.parquet`
- `outputs/metrics/binary_risk_top_terms.csv`
- `outputs/metrics/top_priority_complaints.csv`
- `outputs/figures/binary_risk_roc_curve.png`
- `outputs/figures/binary_risk_pr_curve.png`
- `outputs/figures/binary_risk_calibration_curve.png`
- `outputs/figures/binary_risk_top_terms.png`

### Key Metrics From The Current Polished Run

| Task | Model | Metric | Value |
| --- | --- | --- | --- |
| Multiclass category | TF-IDF + Logistic Regression | Accuracy | 0.8996 |
| Multiclass category | TF-IDF + Logistic Regression | Macro F1 | 0.7829 |
| Binary risk | TF-IDF + Logistic Regression | ROC-AUC | 0.7275 |
| Binary risk | TF-IDF + Logistic Regression | Average Precision | 0.1917 |
| Binary risk | TF-IDF + Logistic Regression | Brier | 0.1146 |
| Binary risk | Calibrated Linear SVM | ROC-AUC | 0.6328 |
| Binary risk | Calibrated Linear SVM | Average Precision | 0.1482 |
| Binary risk | Calibrated Linear SVM | Brier | 0.0765 |

## Risk Scoring

The system produces three related risk outputs:

- `risk_score_rule`: rule-based score from structured signals and narrative keywords
- `risk_probability_ml` / `risk_score_ml`: calibrated binary-model probability from 0.0 to 1.0
- `risk_level_ml`: bucketed risk label derived from the calibrated score

The dashboard prefers the calibrated ML score when model artifacts are available and falls back to the rule-based score otherwise.

Practical interpretation:

- `0.00-0.39`: low triage priority
- `0.40-0.69`: medium triage priority
- `0.70-1.00`: high triage priority

The triage export in `outputs/metrics/top_priority_complaints.csv` ranks complaints by calibrated risk score first and model confidence second.

## Explainability

High-priority explanations combine three layers:

- target-definition reasons from observed CFPB fields
- rule-based reasons from keyword and structured-signal scoring
- model-term reasons from the linear baseline risk model

These are assembled into `risk_reasons_final` in the scored output parquet.

Example explanation string:

```text
target: vulnerable consumer tag | rules: keywords: denied | model terms: navy, navy federal, loan, federal, we
```

That format makes it easy to distinguish:

- what made the complaint high risk by definition
- which rule-based triggers fired
- which narrative terms pushed the model upward

## Running The End-to-End System

Full pipeline:

```bash
PYTHONPATH=src ./.venv/bin/python -m complaint_intel.pipeline.train_system \
  --data data/processed/complaints_clean.parquet \
  --split-date 2023-01-01
```

Faster smoke run:

```bash
PYTHONPATH=src ./.venv/bin/python -m complaint_intel.pipeline.train_system \
  --data data/processed/complaints_clean.parquet \
  --split-date 2023-01-01 \
  --max-train-rows 20000 \
  --max-test-rows 5000 \
  --max-score-rows 5000
```

Launch the dashboard after training:

```bash
PYTHONPATH=src ./.venv/bin/streamlit run src/complaint_intel/app/streamlit_app.py
```
