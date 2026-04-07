from __future__ import annotations

from typing import Iterable

import pandas as pd


VULNERABILITY_TAGS: tuple[str, ...] = ("servicemember", "older american")


def _string_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].astype("string").fillna("")
    return pd.Series("", index=df.index, dtype="string")


def _contains_any(series: pd.Series, terms: Iterable[str]) -> pd.Series:
    pattern = "|".join(term.replace(" ", r"\s+") for term in terms)
    return series.astype("string").str.lower().str.contains(pattern, regex=True, na=False)


def define_high_risk(
    df: pd.DataFrame,
    disputed_col: str = "Consumer disputed?",
    timely_col: str = "Timely response?",
    tags_col: str = "Tags",
) -> pd.DataFrame:
    """
    Define a business-facing binary target for regulatory triage.
    """
    out = df.copy()

    disputed = _string_series(out, disputed_col).str.lower().str.strip()
    timely = _string_series(out, timely_col).str.lower().str.strip()
    tags = _string_series(out, tags_col).str.lower()

    disputed_flag = disputed.eq("yes")
    untimely_flag = timely.eq("no")
    vulnerable_flag = _contains_any(tags, VULNERABILITY_TAGS)

    out["is_disputed"] = disputed_flag.astype(int)
    out["is_untimely"] = untimely_flag.astype(int)
    out["is_vulnerable_consumer"] = vulnerable_flag.astype(int)
    out["high_risk"] = (disputed_flag | untimely_flag | vulnerable_flag).astype(int)

    reasons: list[str] = []
    for idx in out.index:
        row_reasons: list[str] = []
        if disputed_flag.loc[idx]:
            row_reasons.append("consumer disputed")
        if untimely_flag.loc[idx]:
            row_reasons.append("untimely response")
        if vulnerable_flag.loc[idx]:
            row_reasons.append("vulnerable consumer tag")
        reasons.append("; ".join(row_reasons) if row_reasons else "no target risk signals")

    out["high_risk_definition_reason"] = reasons
    return out
