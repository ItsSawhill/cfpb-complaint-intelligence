from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from complaint_intel.risk.high_risk import define_high_risk


@dataclass
class RiskTrainConfig:
    data_path: Path
    model_dir: Path
    metrics_dir: Path
    figures_dir: Path
    split_date: str = "2023-01-01"
    text_col: str = "narrative"
    date_col: str = "date_received"
    min_df: int = 10
    max_features: int = 150_000
    C: float = 2.0
    max_iter: int = 2000
    max_train_rows: int | None = None
    max_test_rows: int | None = None


def time_split(df: pd.DataFrame, date_col: str, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_ts = pd.to_datetime(split_date)
    train_df = df[df[date_col] < split_ts].copy()
    test_df = df[df[date_col] >= split_ts].copy()
    return train_df, test_df


def _sample_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def _safe_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "precision_at_0_5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(np.mean(y_true)),
    }


def _plot_curve(
    xy_by_model: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig = plt.figure(figsize=(8, 6))
    for label, (x, y) in xy_by_model.items():
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _top_linear_features(vectorizer: TfidfVectorizer, model: LogisticRegression, top_n: int = 25) -> pd.DataFrame:
    coef = model.coef_[0]
    names = np.array(vectorizer.get_feature_names_out())
    top_idx = np.argsort(coef)[-top_n:][::-1]
    return pd.DataFrame({"feature": names[top_idx], "coefficient": coef[top_idx]})


def _build_advanced_vectorizer(max_features: int) -> FeatureUnion:
    return FeatureUnion(
        transformer_list=[
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    min_df=10,
                    ngram_range=(1, 2),
                    max_features=max_features,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=5,
                    max_features=max_features // 2,
                    strip_accents="unicode",
                ),
            ),
        ]
    )


def train_models(cfg: RiskTrainConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.data_path)
    required = {cfg.text_col, cfg.date_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for risk model: {missing}")

    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df[df[cfg.date_col].notna()].copy()
    df = define_high_risk(df)

    train_df, test_df = time_split(df, cfg.date_col, cfg.split_date)
    train_df = _sample_rows(train_df, cfg.max_train_rows)
    test_df = _sample_rows(test_df, cfg.max_test_rows)
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Time split produced empty set. train={len(train_df)} test={len(test_df)}")

    X_train = train_df[cfg.text_col].astype("string").fillna("")
    y_train = train_df["high_risk"].astype(int)
    X_test = test_df[cfg.text_col].astype("string").fillna("")
    y_test = test_df["high_risk"].astype(int)

    baseline_vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=cfg.min_df,
        ngram_range=(1, 2),
        max_features=cfg.max_features,
        strip_accents="unicode",
    )
    Xtr_baseline = baseline_vectorizer.fit_transform(X_train)
    Xte_baseline = baseline_vectorizer.transform(X_test)
    baseline_model = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        n_jobs=-1,
        class_weight="balanced",
    )
    baseline_model.fit(Xtr_baseline, y_train)
    baseline_prob = baseline_model.predict_proba(Xte_baseline)[:, 1]

    advanced_vectorizer = _build_advanced_vectorizer(cfg.max_features)
    Xtr_advanced = advanced_vectorizer.fit_transform(X_train)
    Xte_advanced = advanced_vectorizer.transform(X_test)
    advanced_model = CalibratedClassifierCV(
        LinearSVC(C=cfg.C, class_weight="balanced"),
        method="sigmoid",
        cv=3,
    )
    advanced_model.fit(Xtr_advanced, y_train)
    advanced_prob = advanced_model.predict_proba(Xte_advanced)[:, 1]

    metrics = {
        "baseline_logreg": _safe_metrics(y_test, baseline_prob),
        "advanced_calibrated_svm": _safe_metrics(y_test, advanced_prob),
    }
    champion_name = min(metrics, key=lambda name: (metrics[name]["brier"], -metrics[name]["roc_auc"]))
    champion_prob = baseline_prob if champion_name == "baseline_logreg" else advanced_prob

    predictions = test_df[["complaint_id", cfg.date_col, "high_risk", "high_risk_definition_reason"]].copy()
    predictions["baseline_risk_probability"] = baseline_prob
    predictions["advanced_risk_probability"] = advanced_prob
    predictions["risk_probability"] = champion_prob
    predictions["predicted_high_risk"] = (predictions["risk_probability"] >= 0.5).astype(int)
    predictions_out = cfg.metrics_dir / "binary_risk_predictions.parquet"
    predictions.to_parquet(predictions_out, index=False)

    metrics_payload = {
        "split_date": cfg.split_date,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "target_prevalence_train": float(y_train.mean()),
        "target_prevalence_test": float(y_test.mean()),
        "target_definition": "high_risk = disputed OR untimely response OR vulnerable consumer tag",
        "champion_model": champion_name,
        "models": metrics,
    }
    (cfg.metrics_dir / "binary_risk_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    pd.DataFrame(metrics).T.to_csv(cfg.metrics_dir / "binary_risk_metrics.csv", index_label="model")
    _top_linear_features(baseline_vectorizer, baseline_model).to_csv(cfg.metrics_dir / "binary_risk_top_terms.csv", index=False)

    roc_curves = {}
    pr_curves = {}
    for name, probs in {
        "Baseline Logistic Regression": baseline_prob,
        "Advanced Calibrated Linear SVM": advanced_prob,
    }.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        roc_curves[name] = (fpr, tpr)
        pr_curves[name] = (recall, precision)

    _plot_curve(roc_curves, cfg.figures_dir / "binary_risk_roc_curve.png", "Binary Risk ROC Curve", "False Positive Rate", "True Positive Rate")
    _plot_curve(pr_curves, cfg.figures_dir / "binary_risk_pr_curve.png", "Binary Risk Precision-Recall Curve", "Recall", "Precision")

    fig = plt.figure(figsize=(8, 6))
    for name, probs in {
        "Baseline Logistic Regression": baseline_prob,
        "Advanced Calibrated Linear SVM": advanced_prob,
    }.items():
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.title("Binary Risk Calibration Curve")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.legend()
    plt.tight_layout()
    fig.savefig(cfg.figures_dir / "binary_risk_calibration_curve.png", dpi=200)
    plt.close(fig)

    top_terms = pd.read_csv(cfg.metrics_dir / "binary_risk_top_terms.csv")
    fig = plt.figure(figsize=(10, 6))
    plot_df = top_terms.sort_values("coefficient")
    plt.barh(plot_df["feature"], plot_df["coefficient"])
    plt.title("Baseline Binary Risk Top Positive Terms")
    plt.xlabel("Logistic coefficient")
    plt.tight_layout()
    fig.savefig(cfg.figures_dir / "binary_risk_top_terms.png", dpi=200)
    plt.close(fig)

    joblib.dump(baseline_vectorizer, cfg.model_dir / "baseline_risk_vectorizer.joblib")
    joblib.dump(baseline_model, cfg.model_dir / "baseline_risk_model.joblib")
    joblib.dump(advanced_vectorizer, cfg.model_dir / "advanced_risk_vectorizer.joblib")
    joblib.dump(advanced_model, cfg.model_dir / "advanced_risk_model.joblib")
    (cfg.model_dir / "binary_risk_metadata.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("\n=== BINARY RISK MODEL RESULTS ===")
    print(f"Train size: {len(train_df):,}  Test size: {len(test_df):,}")
    print(f"Champion model: {champion_name}")
    for name, values in metrics.items():
        print(f"{name}: ROC-AUC={values['roc_auc']:.4f} AP={values['average_precision']:.4f} Brier={values['brier']:.4f}")
    print(f"[OK] Saved binary risk outputs -> {cfg.metrics_dir} and {cfg.figures_dir}")

    return {"metrics": metrics_payload, "predictions_path": predictions_out}


def main(cfg: RiskTrainConfig) -> None:
    train_models(cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/complaints_clean.parquet")
    p.add_argument("--model-dir", default="models/risk")
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--figures-dir", default="outputs/figures")
    p.add_argument("--split-date", default="2023-01-01")
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-test-rows", type=int, default=None)
    args = p.parse_args()

    cfg = RiskTrainConfig(
        data_path=Path(args.data),
        model_dir=Path(args.model_dir),
        metrics_dir=Path(args.metrics_dir),
        figures_dir=Path(args.figures_dir),
        split_date=args.split_date,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )
    main(cfg)
