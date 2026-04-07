from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from complaint_intel.risk.rule_score import compute_rule_risk, RiskRuleConfig
from complaint_intel.risk.high_risk import define_high_risk


def risk_level_from_score(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def _extract_top_terms(texts: pd.Series, vectorizer, model, top_n: int = 5) -> list[str]:
    X = vectorizer.transform(texts.astype("string").fillna(""))
    if not hasattr(model, "coef_"):
        return ["model explanation unavailable"] * X.shape[0]

    coef = model.coef_[0]
    feature_names = np.array(vectorizer.get_feature_names_out())
    explanations: list[str] = []
    for row_idx in range(X.shape[0]):
        row = X.getrow(row_idx)
        if row.nnz == 0:
            explanations.append("no strong model terms")
            continue
        contributions = row.data * coef[row.indices]
        positive_mask = contributions > 0
        if not positive_mask.any():
            explanations.append("no strong model terms")
            continue
        positive_indices = row.indices[positive_mask]
        positive_contrib = contributions[positive_mask]
        order = np.argsort(positive_contrib)[-top_n:][::-1]
        explanations.append(", ".join(feature_names[positive_indices[order]]))
    return explanations


def main(
    data_path: Path,
    out_path: Path,
    risk_model_dir: Path,
    max_rows: int | None = None,
) -> None:
    df = pd.read_parquet(data_path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    # Rule-based risk
    df = compute_rule_risk(df, cfg=RiskRuleConfig())
    df = define_high_risk(df)

    metadata_path = risk_model_dir / "binary_risk_metadata.json"
    baseline_vec_path = risk_model_dir / "baseline_risk_vectorizer.joblib"
    baseline_model_path = risk_model_dir / "baseline_risk_model.joblib"
    advanced_vec_path = risk_model_dir / "advanced_risk_vectorizer.joblib"
    advanced_model_path = risk_model_dir / "advanced_risk_model.joblib"

    if baseline_vec_path.exists() and baseline_model_path.exists():
        baseline_vec = joblib.load(baseline_vec_path)
        baseline_model = joblib.load(baseline_model_path)
        X_baseline = baseline_vec.transform(df["narrative"].astype("string").fillna(""))
        df["baseline_risk_probability"] = baseline_model.predict_proba(X_baseline)[:, 1]
        df["risk_reasons_model"] = _extract_top_terms(df["narrative"], baseline_vec, baseline_model)
    else:
        df["baseline_risk_probability"] = pd.NA
        df["risk_reasons_model"] = "model explanation unavailable"

    if advanced_vec_path.exists() and advanced_model_path.exists():
        advanced_vec = joblib.load(advanced_vec_path)
        advanced_model = joblib.load(advanced_model_path)
        X_advanced = advanced_vec.transform(df["narrative"].astype("string").fillna(""))
        df["advanced_risk_probability"] = advanced_model.predict_proba(X_advanced)[:, 1]
    else:
        df["advanced_risk_probability"] = pd.NA

    champion_name = "baseline_logreg"
    if metadata_path.exists():
        champion_name = json.loads(metadata_path.read_text(encoding="utf-8")).get("champion_model", champion_name)

    if champion_name == "advanced_calibrated_svm" and df["advanced_risk_probability"].notna().any():
        final_prob = df["advanced_risk_probability"].astype(float)
    elif df["baseline_risk_probability"].notna().any():
        final_prob = df["baseline_risk_probability"].astype(float)
    else:
        final_prob = df["risk_score_rule"].astype(float) / 100.0

    df["risk_probability_ml"] = final_prob
    df["risk_score_ml"] = final_prob
    df["predicted_high_risk"] = (df["risk_probability_ml"] >= 0.5).astype(int)
    df["risk_level_ml"] = df["risk_score_ml"].apply(risk_level_from_score)
    df["risk_reasons_final"] = (
        "target: "
        + df["high_risk_definition_reason"].astype("string")
        + " | rules: "
        + df["risk_reasons_rule"].astype("string")
        + " | model terms: "
        + df["risk_reasons_model"].astype("string")
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[OK] Saved scored table -> {out_path}")
    print(f"     shape={df.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/complaints_clean.parquet")
    p.add_argument("--out", default="outputs/metrics/risk_scored.parquet")
    p.add_argument("--risk-model-dir", default="models/risk")
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    main(Path(args.data), Path(args.out), Path(args.risk_model_dir), args.max_rows)
