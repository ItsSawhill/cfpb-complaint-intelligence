from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from complaint_intel.risk.rule_score import compute_rule_risk, RiskRuleConfig
from complaint_intel.risk.train_risk_model import make_bad_outcome


def risk_level_from_score(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def main(
    data_path: Path,
    out_path: Path,
    risk_model_dir: Path,
) -> None:
    df = pd.read_parquet(data_path)

    # Rule-based risk
    df = compute_rule_risk(df, cfg=RiskRuleConfig())

    # Proxy label (for analysis)
    if "Consumer disputed?" in df.columns and "Timely response?" in df.columns:
        df["bad_outcome"] = make_bad_outcome(df, "Consumer disputed?", "Timely response?")
    else:
        df["bad_outcome"] = pd.NA

    # Learned risk (optional if model exists)
    vec_path = risk_model_dir / "risk_vectorizer.joblib"
    model_path = risk_model_dir / "risk_model.joblib"
    if vec_path.exists() and model_path.exists():
        vec = joblib.load(vec_path)
        clf = joblib.load(model_path)
        X = vec.transform(df["narrative"].astype("string").fillna(""))
        proba = clf.predict_proba(X)[:, 1]
        df["risk_probability_ml"] = proba
        df["risk_score_ml"] = (proba * 100).round(0).astype(int)
        df["risk_level_ml"] = df["risk_score_ml"].apply(risk_level_from_score)
    else:
        df["risk_probability_ml"] = pd.NA
        df["risk_score_ml"] = pd.NA
        df["risk_level_ml"] = pd.NA

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[OK] Saved scored table -> {out_path}")
    print(f"     shape={df.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/complaints_clean.parquet")
    p.add_argument("--out", default="reports/metrics/risk_scored.parquet")
    p.add_argument("--risk-model-dir", default="models/risk")
    args = p.parse_args()

    main(Path(args.data), Path(args.out), Path(args.risk_model_dir))
