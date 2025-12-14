from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass(frozen=True)
class RiskRuleConfig:
    # base weights
    w_disputed: int = 20
    w_untimely: int = 25
    w_recent: int = 5

    # keyword scoring
    w_keyword_hit: int = 8
    w_multi_keyword_bonus: int = 10
    max_keyword_points: int = 35

    # thresholds
    high_threshold: int = 70
    med_threshold: int = 40


DEFAULT_KEYWORDS = [
    "fraud",
    "identity theft",
    "unauthorized",
    "scam",
    "chargeback",
    "foreclosure",
    "repossession",
    "lawsuit",
    "harass",
    "threat",
    "stolen",
    "dispute",
    "refused",
    "denied",
]


def _normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    t = _normalize_text(text)
    hits = []
    for kw in keywords:
        # simple containment; can be upgraded later to regex/lemmatization
        if kw in t:
            hits.append(kw)
    return hits


def risk_level(score: float, cfg: RiskRuleConfig) -> str:
    if score >= cfg.high_threshold:
        return "High"
    if score >= cfg.med_threshold:
        return "Medium"
    return "Low"


def compute_rule_risk(
    df: pd.DataFrame,
    text_col: str = "narrative",
    disputed_col: str = "Consumer disputed?",
    timely_col: str = "Timely response?",
    date_col: str = "date_received",
    keywords: List[str] = DEFAULT_KEYWORDS,
    cfg: RiskRuleConfig = RiskRuleConfig(),
) -> pd.DataFrame:
    """
    Adds rule-based risk columns:
      - risk_score_rule
      - risk_level_rule
      - risk_reasons_rule (semicolon-separated string)
    """
    out = df.copy()
    out[text_col] = out[text_col].astype("string").fillna("")

    # structured signals
    disputed = out[disputed_col].astype("string").str.lower().fillna("")
    timely = out[timely_col].astype("string").str.lower().fillna("")

    disputed_flag = disputed.eq("yes")
    untimely_flag = timely.eq("no")

    # recency: last 90 days gets +w_recent (optional)
    if date_col in out.columns:
        dt = pd.to_datetime(out[date_col], errors="coerce")
        max_dt = dt.max()
        recent_flag = (max_dt - dt).dt.days <= 90
        recent_flag = recent_flag.fillna(False)
    else:
        recent_flag = pd.Series(False, index=out.index)

    # keyword scoring
    hit_lists = out[text_col].apply(lambda x: keyword_hits(x, keywords))
    n_hits = hit_lists.apply(len)

    kw_points = (n_hits.clip(upper=4) * cfg.w_keyword_hit).astype(int)
    kw_points = kw_points.clip(upper=cfg.max_keyword_points)

    multi_bonus = (n_hits >= 3).astype(int) * cfg.w_multi_keyword_bonus

    score = (
        disputed_flag.astype(int) * cfg.w_disputed
        + untimely_flag.astype(int) * cfg.w_untimely
        + recent_flag.astype(int) * cfg.w_recent
        + kw_points
        + multi_bonus
    )

    score = score.clip(lower=0, upper=100).astype(int)

    # reasons
    reasons = []
    for i in range(len(out)):
        r = []
        if disputed_flag.iat[i]:
            r.append("consumer disputed")
        if untimely_flag.iat[i]:
            r.append("untimely response")
        if recent_flag.iat[i]:
            r.append("recent complaint")
        hits = hit_lists.iat[i][:5]
        if hits:
            r.append("keywords: " + ", ".join(hits))
        reasons.append("; ".join(r) if r else "no strong signals")

    out["risk_score_rule"] = score
    out["risk_level_rule"] = out["risk_score_rule"].apply(lambda s: risk_level(s, cfg))
    out["risk_reasons_rule"] = reasons
    out["keyword_hits"] = hit_lists.astype("object")
    out["n_keyword_hits"] = n_hits.astype(int)

    return out
