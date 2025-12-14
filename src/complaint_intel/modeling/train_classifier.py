from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# -------------------------
# Label normalization (MERGE)
# -------------------------
PRODUCT_MAP: Dict[str, str] = {
    # credit reporting variants
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    # credit card variants
    "Credit card or prepaid card": "Credit card",
    # you can add more merges later if you discover taxonomy drift
    "Payday loan, title loan, personal loan, or advance loan": "Payday/personal loan",
    "Payday loan, title loan, or personal loan": "Payday/personal loan",
}


@dataclass
class TrainConfig:
    data_path: Path
    models_dir: Path
    reports_dir: Path
    split_date: str  # e.g., "2023-01-01"
    text_col: str = "narrative"
    label_col: str = "product"
    date_col: str = "date_received"
    min_df: int = 5
    ngram_max: int = 2
    max_features: int | None = 200_000
    C: float = 4.0
    max_iter: int = 2000


def normalize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = df[label_col].astype("string").str.strip()
    df[label_col] = df[label_col].replace(PRODUCT_MAP)
    return df


def time_split(df: pd.DataFrame, date_col: str, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_ts = pd.to_datetime(split_date)
    train_df = df[df[date_col] < split_ts].copy()
    test_df = df[df[date_col] >= split_ts].copy()
    return train_df, test_df


def plot_and_save_confusion_matrix(y_true, y_pred, labels, out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(12, 10))
    plt.imshow(cm)
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(cfg: TrainConfig) -> None:
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.data_path)

    # Basic checks
    required = {cfg.text_col, cfg.label_col, cfg.date_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {cfg.data_path}: {missing}")

    # Ensure date type
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df[df[cfg.date_col].notna()].copy()

    # Normalize/merge labels
    df = normalize_labels(df, cfg.label_col)

    # Time split
    train_df, test_df = time_split(df, cfg.date_col, cfg.split_date)
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Time split produced empty set. "
            f"Try a different --split-date. train={len(train_df)} test={len(test_df)}"
        )

    X_train = train_df[cfg.text_col].astype("string").fillna("")
    y_train = train_df[cfg.label_col].astype("string")
    X_test = test_df[cfg.text_col].astype("string").fillna("")
    y_test = test_df[cfg.label_col].astype("string")

    # Filter rare classes in TRAIN to avoid unstable classes
    min_train_samples = 2000  # tune: 500, 1000, 2000
    vc = y_train.value_counts()
    keep = vc[vc >= min_train_samples].index
    train_df = train_df[train_df[cfg.label_col].isin(keep)].copy()
    test_df = test_df[test_df[cfg.label_col].isin(keep)].copy()

    X_train = train_df[cfg.text_col].astype("string").fillna("")
    y_train = train_df[cfg.label_col].astype("string")
    X_test  = test_df[cfg.text_col].astype("string").fillna("")
    y_test  = test_df[cfg.label_col].astype("string")

    # Vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=cfg.min_df,
        ngram_range=(1, cfg.ngram_max),
        max_features=cfg.max_features,
        strip_accents="unicode",
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    # Model
    clf = LogisticRegression(
    C=cfg.C,
    max_iter=cfg.max_iter,
    n_jobs=-1,
    class_weight="balanced",
    )
    clf.fit(Xtr, y_train)

    # Predictions + confidence
    proba = clf.predict_proba(Xte)
    pred = clf.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)

    # Metrics
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    print("\n=== CLASSIFIER RESULTS (TF-IDF + LogisticRegression) ===")
    print(f"Train size: {len(train_df):,}  Test size: {len(test_df):,}")
    print(f"Split date: {cfg.split_date}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}\n")

    report_txt = classification_report(y_test, pred, digits=4, zero_division=0)
    print(report_txt)

    # Save metrics
    metrics_path = cfg.reports_dir / "classifier_metrics.txt"
    metrics_path.write_text(
        f"Accuracy: {acc:.6f}\nMacroF1: {macro_f1:.6f}\n\n{report_txt}",
        encoding="utf-8",
    )

    # Save confusion matrix (labels in sorted order)
    labels = sorted(y_test.unique().tolist())
    cm_path = cfg.reports_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(y_test, pred, labels, cm_path, title="Confusion Matrix (Test)")

    # Save artifacts
    joblib.dump(vectorizer, cfg.models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(clf, cfg.models_dir / "logreg_model.joblib")

    # Save predictions for dashboard integration
    pred_df = test_df[["complaint_id", cfg.date_col]].copy()
    pred_df["true_product"] = y_test.values
    pred_df["predicted_product"] = pred
    pred_df["prediction_confidence"] = conf
    pred_out = cfg.reports_dir / "test_predictions.parquet"
    pred_df.to_parquet(pred_out, index=False)

    print(f"\n[OK] Saved metrics -> {metrics_path}")
    print(f"[OK] Saved confusion matrix -> {cm_path}")
    print(f"[OK] Saved vectorizer -> {cfg.models_dir / 'tfidf_vectorizer.joblib'}")
    print(f"[OK] Saved model -> {cfg.models_dir / 'logreg_model.joblib'}")
    print(f"[OK] Saved test predictions -> {pred_out}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/complaints_clean.parquet")
    p.add_argument("--models-dir", default="models/classifier")
    p.add_argument("--reports-dir", default="reports/metrics")
    p.add_argument("--split-date", default="2023-01-01", help="Time split boundary (YYYY-MM-DD).")
    p.add_argument("--min-df", type=int, default=5)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--max-features", type=int, default=200_000)
    p.add_argument("--C", type=float, default=4.0)
    p.add_argument("--max-iter", type=int, default=2000)
    args = p.parse_args()

    cfg = TrainConfig(
        data_path=Path(args.data),
        models_dir=Path(args.models_dir),
        reports_dir=Path(args.reports_dir),
        split_date=args.split_date,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
        C=args.C,
        max_iter=args.max_iter,
    )
    main(cfg)
