from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure expected directories exist and return config with Path objects.
    """
    paths = cfg.get("paths", {})
    raw_dir = ensure_dir(paths.get("raw_dir", "data/raw"))
    interim_dir = ensure_dir(paths.get("interim_dir", "data/interim"))
    processed_dir = ensure_dir(paths.get("processed_dir", "data/processed"))

    cfg["paths"]["raw_dir"] = str(raw_dir)
    cfg["paths"]["interim_dir"] = str(interim_dir)
    cfg["paths"]["processed_dir"] = str(processed_dir)
    return cfg
