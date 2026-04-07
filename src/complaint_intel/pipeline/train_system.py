from __future__ import annotations

import argparse
from pathlib import Path

from complaint_intel.app.build_dashboard_data import main as build_dashboard_main
from complaint_intel.modeling.train_classifier import TrainConfig, main as train_classifier_main
from complaint_intel.risk.score import main as score_risk_main
from complaint_intel.risk.train_risk_model import RiskTrainConfig, main as train_risk_main


def main(
    data_path: Path,
    split_date: str,
    classifier_model_dir: Path,
    risk_model_dir: Path,
    metrics_dir: Path,
    figures_dir: Path,
    dashboard_out: Path,
    max_train_rows: int | None = None,
    max_test_rows: int | None = None,
    max_score_rows: int | None = None,
) -> None:
    train_classifier_main(
        TrainConfig(
            data_path=data_path,
            models_dir=classifier_model_dir,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
            split_date=split_date,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
        )
    )
    train_risk_main(
        RiskTrainConfig(
            data_path=data_path,
            model_dir=risk_model_dir,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
            split_date=split_date,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
        )
    )
    score_risk_main(data_path, metrics_dir / "risk_scored.parquet", risk_model_dir, max_score_rows)
    build_dashboard_main(metrics_dir / "risk_scored.parquet", metrics_dir / "multiclass_predictions.parquet", dashboard_out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/complaints_clean.parquet")
    p.add_argument("--split-date", default="2023-01-01")
    p.add_argument("--classifier-model-dir", default="models/classifier")
    p.add_argument("--risk-model-dir", default="models/risk")
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--figures-dir", default="outputs/figures")
    p.add_argument("--dashboard-out", default="data/processed/dashboard.parquet")
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-test-rows", type=int, default=None)
    p.add_argument("--max-score-rows", type=int, default=None)
    args = p.parse_args()

    main(
        data_path=Path(args.data),
        split_date=args.split_date,
        classifier_model_dir=Path(args.classifier_model_dir),
        risk_model_dir=Path(args.risk_model_dir),
        metrics_dir=Path(args.metrics_dir),
        figures_dir=Path(args.figures_dir),
        dashboard_out=Path(args.dashboard_out),
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        max_score_rows=args.max_score_rows,
    )
