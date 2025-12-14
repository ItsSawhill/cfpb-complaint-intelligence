from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class RiskTrainConfig:
    data_path: Path
    model_dir: Path
    split_date: str = "2023-01-01"
    text_col: str = "narrative"
    disputed_col: str = "Consumer disputed?"
    timely_col: str = "Timely response?"
    date_col: str = "date_received"
    min_df: int = 10
    max_features: int = 200_000
    C: float = 2.0
    max_iter: int = 2000


def make_bad_outcome(df: pd.DataFrame, disputed_col: str, timely_col: str) -> pd.Series:
    disputed = df[disputed_col].astype("string").str.lower().fillna("")
    timely = df[timely_col].astype("string").str.lower().fillna("")
    return (disputed.eq("yes") | timely.eq("no")).astype(int)


def time_split(df: pd.DataFrame, date_col: str, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_ts = pd.to_datetime(split_date)
    train_df = df[df[date_col] < split_ts].copy()
    test_df = df[df[date_col] >= split_ts].copy()
    return train_df, test_df


def main(cfg: RiskTrainConfig) -> None:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.data_path)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df[df[cfg.date_col].notna()].copy()

    # must have structured columns for proxy label
    needed = {cfg.disputed_col, cfg.timely_col, cfg.text_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for risk model: {missing}")

    df["bad_outcome"] = make_bad_outcome(df, cfg.disputed_col, cfg.timely_col)

    train_df, test_df = time_split(df, cfg.date_col, cfg.split_date)

    X_train = train_df[cfg.text_col].astype("string").fillna("")
    y_train = train_df["bad_outcome"].astype(int)

    X_test = test_df[cfg.text_col].astype("string").fillna("")
    y_test = test_df["bad_outcome"].astype(int)

    vec = TfidfVectorizer(
        lowercase=True,
        min_df=cfg.min_df,
        ngram_range=(1, 2),
        max_features=cfg.max_features,
        strip_accents="unicode",
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        class_weight="balanced",
    )
    clf.fit(Xtr, y_train)

    proba = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print("\n=== RISK MODEL RESULTS (Proxy bad_outcome) ===")
    print(f"Train size: {len(train_df):,}  Test size: {len(test_df):,}")
    print(f"ROC-AUC:    {auc:.4f}")
    print(f"AvgPrec:    {ap:.4f}\n")

    joblib.dump(vec, cfg.model_dir / "risk_vectorizer.joblib")
    joblib.dump(clf, cfg.model_dir / "risk_model.joblib")
    print(f"[OK] Saved risk vectorizer/model -> {cfg.model_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/complaints_clean.parquet")
    p.add_argument("--model-dir", default="models/risk")
    p.add_argument("--split-date", default="2023-01-01")
    args = p.parse_args()

    cfg = RiskTrainConfig(
        data_path=Path(args.data),
        model_dir=Path(args.model_dir),
        split_date=args.split_date,
    )
    main(cfg)
