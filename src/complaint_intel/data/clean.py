from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from complaint_intel.utils.io import load_yaml, resolve_paths


def pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def word_count(s: str) -> int:
    return len(str(s).split())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names lightly (trim spaces).
    Keep original names too; we just avoid accidental whitespace bugs.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_complaints(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    date_col: Optional[str],
    min_words: int,
    keep_top_n_products: Optional[int],
    drop_missing_narratives: bool = True,
) -> pd.DataFrame:
    df = df.copy()

    # Keep only needed columns first (and some useful metadata if present)
    keep_cols = [text_col, label_col]
    optional_cols = [
        date_col,
        "Company",
        "State",
        "ZIP code",
        "Tags",
        "Consumer disputed?",
        "Timely response?",
        "Complaint ID",
        "Issue",
        "Sub-issue",
        "Sub-product",
        "Submitted via",
        "Company response to consumer",
    ]
    for c in optional_cols:
        if c and c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    df = df[keep_cols]

    # Rename into canonical names used downstream
    rename_map = {
        text_col: "narrative",
        label_col: "product",
    }
    if date_col:
        rename_map[date_col] = "date_received"
    df = df.rename(columns=rename_map)

    # Drop missing narratives
    if drop_missing_narratives:
        df["narrative"] = df["narrative"].astype("string")
        df = df[df["narrative"].notna()]
        df = df[df["narrative"].str.strip().str.len() > 0]

    # Basic text cleanup
    df["narrative"] = (
        df["narrative"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Minimum length filter
    df["n_words"] = df["narrative"].apply(word_count)
    df = df[df["n_words"] >= int(min_words)]

    # Clean product labels
    df["product"] = df["product"].astype("string").str.strip()

    # Keep top N products (optional for a clean first model)
    if keep_top_n_products is not None:
        top_products = df["product"].value_counts().head(int(keep_top_n_products)).index
        df = df[df["product"].isin(top_products)].copy()

    # Parse dates
    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
        # Drop rows with invalid dates if present
        df = df[df["date_received"].notna()]

    # Add stable id column
    if "Complaint ID" in df.columns and "complaint_id" not in df.columns:
        df["complaint_id"] = df["Complaint ID"]
    elif "complaint_id" not in df.columns:
        df["complaint_id"] = pd.util.hash_pandas_object(df["narrative"], index=False).astype("int64")

    # Final column ordering (safe)
    preferred = [
        "complaint_id",
        "date_received",
        "product",
        "narrative",
        "n_words",
        "Company",
        "State",
        "ZIP code",
        "Tags",
        "Consumer disputed?",
        "Timely response?",
        "Issue",
        "Sub-issue",
        "Sub-product",
        "Submitted via",
        "Company response to consumer",
    ]
    final_cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[final_cols].reset_index(drop=True)

    return df


def main(config_path: str) -> None:
    cfg = resolve_paths(load_yaml(config_path))
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])

    # Detect which raw parquet exists
    raw_parquet_csv = interim_dir / "complaints_raw.parquet"
    raw_parquet_api = interim_dir / "complaints_raw_api.parquet"
    if raw_parquet_csv.exists():
        df = pd.read_parquet(raw_parquet_csv)
        src = raw_parquet_csv
    elif raw_parquet_api.exists():
        df = pd.read_parquet(raw_parquet_api)
        src = raw_parquet_api
    else:
        raise FileNotFoundError(
            "No ingested parquet found. Run ingest first:\n"
            "python -m complaint_intel.data.ingest --config configs/base.yaml"
        )

    df = standardize_columns(df)

    text_col = pick_first_existing_column(df, cfg["cleaning"]["text_col_candidates"])
    label_col = pick_first_existing_column(df, cfg["cleaning"]["label_col_candidates"])
    date_col = pick_first_existing_column(df, cfg["cleaning"]["date_col_candidates"])

    if not text_col:
        raise ValueError(f"Could not find narrative column. Candidates={cfg['cleaning']['text_col_candidates']}")
    if not label_col:
        raise ValueError(f"Could not find label column. Candidates={cfg['cleaning']['label_col_candidates']}")

    keep_top = cfg["cleaning"].get("keep_top_n_products", None)
    if keep_top in ("null", "None", ""):
        keep_top = None

    cleaned = clean_complaints(
        df=df,
        text_col=text_col,
        label_col=label_col,
        date_col=date_col,
        min_words=int(cfg["cleaning"]["min_words"]),
        keep_top_n_products=keep_top,
        drop_missing_narratives=bool(cfg["cleaning"]["drop_missing_narratives"]),
    )

    out_name = cfg["cleaning"]["output_parquet"]
    out_path = processed_dir / out_name
    cleaned.to_parquet(out_path, index=False)

    print(f"[OK] Cleaned from {src.name} -> {out_path}")
    print(f"     cleaned shape={cleaned.shape}")
    print(f"     products={cleaned['product'].nunique() if 'product' in cleaned.columns else 'NA'}")
    if "date_received" in cleaned.columns:
        print(f"     date range={cleaned['date_received'].min().date()} to {cleaned['date_received'].max().date()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    main(args.config)
