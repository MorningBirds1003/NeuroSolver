"""
field_mapping.py

Utilities for sampling extracellular potential on regular spatial grids for
spatiotemporal field-map visualization.

This is used to build phi_e(x,t) and related maps for NeuroSolver bundle runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from Scripts.NeuroSolver.ECS.vc_solver import ElectrodeSamplePoint


@dataclass(frozen=True)
class FieldSampleGridSpec:
    """
    Specification for a regular 1D or 2D field sampling grid.

    For now, we mostly use 1D line sampling:
    - x varies
    - y and z fixed

    This is enough to generate phi_e(x,t) heatmaps.
    """
    name: str
    x_min_um: float
    x_max_um: float
    n_x: int
    y_um: float = 0.0
    z_um: float = 0.0

def build_line_sample_points(
    spec: FieldSampleGridSpec,
) -> tuple[np.ndarray, List[ElectrodeSamplePoint]]:
    """
    Build a regular x-line of sample points at fixed y,z.

    Returns
    -------
    x_um : np.ndarray
        Shape (n_x,)
    points : list[ElectrodeSamplePoint]
        One point per x-location
    """
    if spec.n_x < 2:
        raise ValueError("FieldSampleGridSpec.n_x must be >= 2.")

    x_um = np.linspace(float(spec.x_min_um), float(spec.x_max_um), int(spec.n_x), dtype=float)
    points = [
        ElectrodeSamplePoint(
            name=f"{spec.name}_x{i:03d}",
            x_um=float(x),
            y_um=float(spec.y_um),
            z_um=float(spec.z_um),
        )
        for i, x in enumerate(x_um)
    ]
    return x_um, points