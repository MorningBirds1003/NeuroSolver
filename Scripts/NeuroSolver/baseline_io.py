"""
baseline_io.py

Small JSON helpers for Phase 0 regression baselines.

Intent
------
Keep serialization logic in one place so regression snapshots are:
- easy to save,
- stable across runs,
- safe for NumPy-heavy payloads.

This module does not know anything about the nerve model itself. It only
converts common runtime objects into JSON-compatible Python objects and
loads/saves them to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

def to_serializable(value: Any) -> Any:
    """
    Recursively convert common NumPy-rich payloads into JSON-safe objects.

    Conversion rules
    ----------------
    - ndarray -> nested Python lists
    - NumPy scalar -> native Python scalar
    - dict -> recursively converted dict
    - list/tuple -> recursively converted list

    Notes
    -----
    The output is intentionally conservative: tuples are converted to lists so
    the result is always accepted by `json.dump`.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value

def save_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """
    Save a JSON payload to disk, creating parent directories if needed.

    Parameters
    ----------
    path
        Output file path.
    payload
        Arbitrary object to serialize.
    indent
        JSON indentation level for readability.

    Returns
    -------
    Path
        The resolved output path object used for writing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=indent, sort_keys=True)
    return path

def load_json(path: str | Path) -> Any:
    """
    Load a JSON file from disk.

    Parameters
    ----------
    path
        Input JSON path.

    Returns
    -------
    Any
        Parsed JSON payload.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
