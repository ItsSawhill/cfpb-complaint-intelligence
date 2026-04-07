from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from tqdm import tqdm

from complaint_intel.utils.io import load_yaml, resolve_paths


def _extract_api_records(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Accept a few common CFPB-style response shapes and return a flat records list.
    """
    for key in ("hits", "results", "records", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for nested_key in ("hits", "results", "records"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, list):
                    return nested_value
    return []


def ingest_from_api(
    base_url: str,
    page_size: int = 100,
    max_pages: int = 50,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch complaints from the CFPB API using simple page/size pagination.
    """
    session = requests.Session()
    rows: list[Dict[str, Any]] = []

    for page in tqdm(range(max_pages), desc="Fetching API pages"):
        params = {"size": page_size, "page": page}
        response = session.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()

        payload = response.json()
        batch = _extract_api_records(payload)
        if not batch:
            break

        rows.extend(batch)
        if len(batch) < page_size:
            break

    if not rows:
        raise ValueError(f"No complaint records returned from API: {base_url}")

    return pd.DataFrame(rows)


def ingest_from_csv(
    csv_path: str | Path,
    sample_frac: float = 0.3,
    chunk_size: int = 200_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stream a large CSV and randomly sample a fraction of rows.
    Designed for very large CFPB CSVs (7GB+).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    usecols = [
        "Date received",
        "Product",
        "Issue",
        "Company",
        "State",
        "ZIP code",
        "Tags",
        "Consumer disputed?",
        "Timely response?",
        "Complaint ID",
        "Consumer complaint narrative",
        "Submitted via",
        "Company response to consumer",
        "Sub-product",
        "Sub-issue",
    ]

    dtype = {
        "Product": "string",
        "Issue": "string",
        "Company": "string",
        "State": "string",
        "ZIP code": "string",
        "Tags": "string",
        "Consumer disputed?": "string",
        "Timely response?": "string",
        "Complaint ID": "string",
        "Submitted via": "string",
        "Company response to consumer": "string",
        "Sub-product": "string",
        "Sub-issue": "string",
    }

    rng = random_state

    chunks = []
    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in reader:
        sampled = chunk.sample(frac=sample_frac, random_state=rng)
        chunks.append(sampled)

    df = pd.concat(chunks, ignore_index=True)
    return df




def main(config_path: str) -> None:
    cfg = resolve_paths(load_yaml(config_path))
    mode = cfg["ingest"].get("mode", "csv")

    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])

    if mode == "csv":
        csv_path = cfg["ingest"]["csv_path"]
        df = ingest_from_csv(
            csv_path,
            sample_frac=0.3,
        )
        out_path = interim_dir / "complaints_raw.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[OK] Ingested CSV with shape={df.shape} -> {out_path}")

    elif mode == "api":
        api_cfg = cfg["ingest"]["api"]
        df = ingest_from_api(
            base_url=api_cfg["base_url"],
            page_size=int(api_cfg.get("page_size", 100)),
            max_pages=int(api_cfg.get("max_pages", 50)),
        )
        out_path = interim_dir / "complaints_raw_api.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[OK] Ingested API with shape={df.shape} -> {out_path}")

    else:
        raise ValueError(f"Unknown ingest mode: {mode}. Use 'csv' or 'api'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    main(args.config)
