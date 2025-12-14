from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(
    risk_path: Path,
    preds_path: Path,
    out_path: Path,
) -> None:
    risk = pd.read_parquet(risk_path)
    preds = pd.read_parquet(preds_path)

    # Ensure join keys exist
    for col in ["complaint_id"]:
        if col not in risk.columns:
            raise ValueError(f"Missing {col} in risk_scored table")
        if col not in preds.columns:
            raise ValueError(f"Missing {col} in predictions table")

    # Reduce risk table to dashboard-relevant columns
    keep_risk = [
        "complaint_id",
        "date_received",
        "product",
        "narrative",
        "Company",
        "State",
        "Tags",
        "Consumer disputed?",
        "Timely response?",
        "risk_score_rule",
        "risk_level_rule",
        "risk_reasons_rule",
        "risk_probability_ml",
        "risk_score_ml",
        "risk_level_ml",
    ]
    keep_risk = [c for c in keep_risk if c in risk.columns]
    risk_small = risk[keep_risk].copy()

    # Merge (left join on predictions so dashboard focuses on the evaluation/test period)
    # preds contains only TEST (>= split date) rows from the classifier script
    dash = preds.merge(risk_small, on="complaint_id", how="left", suffixes=("", "_risk"))

    # Prefer consistent date/product fields from risk table if present
    if "date_received_risk" in dash.columns and "date_received" in dash.columns:
        dash["date_received"] = dash["date_received"].fillna(dash["date_received_risk"])
        dash = dash.drop(columns=["date_received_risk"])

    # Add a single “final” risk score column for UI (prefer ML if available)
    if "risk_score_ml" in dash.columns:
        dash["risk_score_final"] = dash["risk_score_ml"]
        dash["risk_level_final"] = dash["risk_level_ml"]
        dash["risk_reasons_final"] = dash.get("risk_reasons_rule")
        # fallback if ML is NA
        mask = dash["risk_score_final"].isna()
        dash.loc[mask, "risk_score_final"] = dash.loc[mask, "risk_score_rule"]
        dash.loc[mask, "risk_level_final"] = dash.loc[mask, "risk_level_rule"]
    else:
        dash["risk_score_final"] = dash["risk_score_rule"]
        dash["risk_level_final"] = dash["risk_level_rule"]
        dash["risk_reasons_final"] = dash.get("risk_reasons_rule")

    # Basic cleanup for UI
    if "prediction_confidence" in dash.columns:
        dash["prediction_confidence"] = dash["prediction_confidence"].astype(float)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dash.to_parquet(out_path, index=False)

    print(f"[OK] Dashboard dataset saved -> {out_path}")
    print(f"     shape={dash.shape}")
    print("     columns:", ", ".join(list(dash.columns)[:25]), "...")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--risk", default="reports/metrics/risk_scored.parquet")
    p.add_argument("--preds", default="reports/metrics/test_predictions.parquet")
    p.add_argument("--out", default="data/processed/dashboard.parquet")
    args = p.parse_args()

    main(Path(args.risk), Path(args.preds), Path(args.out))
