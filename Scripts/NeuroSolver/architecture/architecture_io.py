"""
architecture_io.py

JSON import/export helpers for NerveArchitectureSpec.
"""

from __future__ import annotations

import json
from pathlib import Path
from Scripts.NeuroSolver.architecture.architecture_schema import NerveArchitectureSpec

def save_architecture_spec(spec: NerveArchitectureSpec, filepath: str | Path) -> Path:
    """
    Save a NerveArchitectureSpec to JSON.
    """
    spec.validate()
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2)

    return path

def load_architecture_spec(filepath: str | Path) -> NerveArchitectureSpec:
    """
    Load a NerveArchitectureSpec from JSON.
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    spec = NerveArchitectureSpec.from_dict(payload)
    spec.validate()
    return spec