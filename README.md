# CFPB Complaint Intelligence System

This repository presents an end-to-end complaint intelligence workflow for CFPB complaint narratives. It is designed for regulatory triage: classify the complaint category, estimate whether the complaint is high risk, assign a calibrated risk score from 0 to 1, explain the drivers of prioritization, and export a ranked triage list for analyst review.

## Business Problem

Consumer-finance complaint teams work with large volumes of unstructured narratives. A practical triage system needs to do more than label complaints: it should identify which complaints deserve immediate attention, provide a stable risk score, and explain why the complaint rose to the top of the queue.

This project addresses that workflow by combining:

- multiclass complaint categorization
- binary high-risk prediction
- calibrated probability scoring
- explainable prioritization outputs
- triage-ready artifact generation

## System Purpose

The system is built to support a reviewer or regulatory analyst through the following sequence:

1. ingest and clean CFPB complaint narratives
2. classify complaint category
3. estimate calibrated high-risk probability
4. combine rule-based and model-based explanations
5. export ranked complaints for triage and dashboard inspection

## Targets

### Multiclass Target

The multiclass task predicts complaint `product` from the cleaned CFPB dataset. In the current aligned run, the classifier operates on five retained product classes after label normalization and minimum-support filtering.

### Binary `high_risk` Target

The binary target is explicitly defined in [high_risk.py](src/complaint_intel/risk/high_risk.py):

`high_risk = 1` when any of the following hold:

- `Consumer disputed? = Yes`
- `Timely response? = No`
- `Tags` contains a vulnerable-consumer marker such as `Servicemember` or `Older American`

This gives the risk model a concrete operational target tied to observable complaint metadata rather than a vague notion of severity.

## Models

### Multiclass Complaint Classifier

- Model: TF-IDF + Logistic Regression
- Output artifacts: `predicted_product`, `prediction_confidence`
- Code path: [src/complaint_intel/modeling/train_classifier.py](src/complaint_intel/modeling/train_classifier.py)

### Binary High-Risk Models

- Baseline: word-level TF-IDF + Logistic Regression
- Advanced: word + character TF-IDF + calibrated Linear SVM
- Champion selection: best binary model chosen from the current run using calibration quality, with Brier score as the primary criterion
- Code path: [src/complaint_intel/risk/train_risk_model.py](src/complaint_intel/risk/train_risk_model.py)

## Calibrated Risk Score

The final risk score is a calibrated probability in `[0, 1]`.

- `risk_probability_ml` / `risk_score_ml`: final calibrated binary-model probability
- `risk_score_rule`: rule-based score from structured signals and narrative keywords
- `risk_level_ml`: bucketed view of the calibrated probability

Interpretation used in the project:

- `0.00-0.39`: low priority
- `0.40-0.69`: medium priority
- `0.70-1.00`: high priority

The dashboard and triage outputs prefer the calibrated ML score when model artifacts are available.

## Explainability Approach

The system builds `risk_reasons_final` by combining three layers:

- target-definition reasons from complaint metadata
- rule-based reasons from keywords and structured triggers
- model-term reasons from the linear baseline binary model

Example explanation string from the current outputs:

```text
target: vulnerable consumer tag | rules: keywords: denied | model terms: navy, navy federal, loan, federal, we
```

This makes it possible to separate:

- why the complaint qualifies as high-risk by definition
- which deterministic rules fired
- which narrative terms drove the model score upward

## Key Results

These values come directly from the aligned artifacts currently stored in `outputs/metrics/`.

- Multiclass accuracy: `0.8996`
- Multiclass macro F1: `0.7829`
- Binary champion model: `advanced_calibrated_svm`
- Aligned artifact row counts:
  - `multiclass_predictions.parquet`: `1882`
  - `risk_scored.parquet`: `1882`
  - `dashboard.parquet`: `1882`

### Model Comparison

| Task | Model | Main metric snapshot | Notes |
| --- | --- | --- | --- |
| Multiclass category | TF-IDF + Logistic Regression | Accuracy `0.8996`, Macro F1 `0.7829` | Current complaint-category model |
| Binary high risk | TF-IDF + Logistic Regression | ROC-AUC `0.7275`, AP `0.1917`, Brier `0.1146` | Better ranking metrics in current run |
| Binary high risk | Calibrated Linear SVM | ROC-AUC `0.6328`, AP `0.1482`, Brier `0.0765` | Current champion by calibration quality |

The binary result is worth reading carefully: the advanced model wins on calibration, while the baseline logistic model is stronger on ROC-AUC and average precision in the current sampled run.

## System Pipeline

The project pipeline is:

1. `Ingest`
   Read raw CFPB complaints from CSV or API into intermediate parquet.

2. `Preprocess`
   Clean narrative text, standardize schema, normalize date fields, and retain downstream modeling columns.

3. `Classify`
   Train the multiclass TF-IDF + Logistic Regression model and write complaint-category predictions.

4. `Risk Score`
   Train binary high-risk models, calibrate the final probability, and select the champion model.

5. `Explain`
   Generate rule-based reasons, target-definition reasons, and model-term explanations.

6. `Triage Output`
   Merge aligned artifacts into dashboard data and export ranked complaints for analyst review.

## Output Artifacts

The primary generated outputs live in:

- `outputs/metrics/`
- `outputs/figures/`

Important metric and table artifacts:

- `outputs/metrics/multiclass_metrics.json`
- `outputs/metrics/multiclass_metrics.txt`
- `outputs/metrics/multiclass_predictions.parquet`
- `outputs/metrics/binary_risk_metrics.json`
- `outputs/metrics/binary_risk_metrics.csv`
- `outputs/metrics/binary_risk_predictions.parquet`
- `outputs/metrics/binary_risk_top_terms.csv`
- `outputs/metrics/risk_scored.parquet`
- `outputs/metrics/top_priority_complaints.csv`

Important figure artifacts:

- `outputs/figures/multiclass_confusion_matrix.png`
- `outputs/figures/binary_risk_roc_curve.png`
- `outputs/figures/binary_risk_pr_curve.png`
- `outputs/figures/binary_risk_calibration_curve.png`
- `outputs/figures/binary_risk_top_terms.png`

Legacy files still exist under `reports/metrics/`, but the polished presentation artifacts for this version of the project are the files under `outputs/`.

## Triage Workflow

A practical triage workflow for this repository is:

1. run the pipeline and generate aligned prediction artifacts
2. inspect `outputs/metrics/top_priority_complaints.csv`
3. review the corresponding rows in `data/processed/dashboard.parquet`
4. use `risk_score_final`, `predicted_high_risk`, and `risk_reasons_final` to prioritize analyst review

The triage CSV is sorted by:

1. highest calibrated risk score
2. highest classifier confidence

## Triage Output Example

The current generated triage file is [outputs/metrics/top_priority_complaints.csv](outputs/metrics/top_priority_complaints.csv). The first few rows show the intended output shape:

| complaint_id | predicted_product | risk_score_final | risk_level_final | example reason |
| --- | --- | --- | --- | --- |
| `9904025` | `Credit card` | `0.6368` | `Medium` | `target: vulnerable consumer tag | rules: keywords: refused | model terms: military, wife, xxxx, close the, the account` |
| `8000818` | `Mortgage` | `0.5681` | `Medium` | `target: vulnerable consumer tag | rules: keywords: denied | model terms: navy, navy federal, loan, federal, we` |
| `6550762` | `Credit card` | `0.5005` | `Medium` | `target: no target risk signals | rules: no strong signals | model terms: plus, 00, barclay, xxxx, interest` |

These examples are drawn directly from the generated output and intentionally use the redacted complaint text already present in the CFPB-derived data.

## Repository Map

Core code paths:

- [src/complaint_intel/data/](src/complaint_intel/data/)
- [src/complaint_intel/modeling/](src/complaint_intel/modeling/)
- [src/complaint_intel/risk/](src/complaint_intel/risk/)
- [src/complaint_intel/app/](src/complaint_intel/app/)
- [src/complaint_intel/pipeline/](src/complaint_intel/pipeline/)

Presentation and outputs:

- [outputs/metrics/](outputs/metrics/)
- [outputs/figures/](outputs/figures/)
- [reports/](reports/)

## How To Run

Run the end-to-end workflow:

```bash
PYTHONPATH=src ./.venv/bin/python -m complaint_intel.pipeline.train_system \
  --data data/processed/complaints_clean.parquet \
  --split-date 2023-01-01
```

Run a faster smoke version:

```bash
PYTHONPATH=src ./.venv/bin/python -m complaint_intel.pipeline.train_system \
  --data data/processed/complaints_clean.parquet \
  --split-date 2023-01-01 \
  --max-train-rows 20000 \
  --max-test-rows 5000 \
  --max-score-rows 5000
```

Launch the Streamlit dashboard:

```bash
PYTHONPATH=src ./.venv/bin/streamlit run src/complaint_intel/app/streamlit_app.py
```

## Notes For Reviewers

- All headline metrics in this README are taken from the current generated files under `outputs/metrics/`.
- The aligned triage bundle is intentionally evaluation-focused: `multiclass_predictions.parquet`, `risk_scored.parquet`, and `dashboard.parquet` all contain `1882` rows from the same aligned set.
- If a recruiter or reviewer wants one file to inspect first, start with `outputs/metrics/top_priority_complaints.csv`.
