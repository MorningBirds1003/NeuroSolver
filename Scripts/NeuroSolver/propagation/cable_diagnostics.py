"""
cable_diagnostics.py

Post-processing helpers for cable runs.

These functions do not advance the solver. They only analyze voltage traces
that were produced elsewhere.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def peak_voltage_by_compartment_mV(V_history_mV: np.ndarray) -> np.ndarray:
    """Return the peak voltage reached by each compartment."""
    V = np.asarray(V_history_mV, dtype=float)
    if V.ndim != 2:
        raise ValueError("V_history_mV must have shape (n_time, n_compartments).")
    return np.max(V, axis=0)


def minimum_voltage_by_compartment_mV(V_history_mV: np.ndarray) -> np.ndarray:
    """Return the minimum voltage reached by each compartment."""
    V = np.asarray(V_history_mV, dtype=float)
    if V.ndim != 2:
        raise ValueError("V_history_mV must have shape (n_time, n_compartments).")
    return np.min(V, axis=0)


def threshold_crossing_times_ms(
    t_ms: Sequence[float],
    V_history_mV: np.ndarray,
    threshold_mV: float = 0.0,
) -> np.ndarray:
    """
    Return the first upward threshold-crossing time for each compartment.

    Compartments that never cross return np.nan.
    """
    t = np.asarray(t_ms, dtype=float)
    V = np.asarray(V_history_mV, dtype=float)

    if V.ndim != 2:
        raise ValueError("V_history_mV must have shape (n_time, n_compartments).")
    if V.shape[0] != t.size:
        raise ValueError("Time axis length must match V_history first dimension.")

    n_comp = V.shape[1]
    crossings = np.full(n_comp, np.nan, dtype=float)

    above = V >= float(threshold_mV)
    for j in range(n_comp):
        idx = np.where(above[:, j])[0]
        if idx.size > 0:
            crossings[j] = t[idx[0]]

    return crossings


def estimate_conduction_velocity_m_per_s(
    x_um: Sequence[float],
    crossing_times_ms: Sequence[float],
    i0: int,
    i1: int,
) -> float:
    """
    Estimate conduction velocity between two compartments.

    Returns np.nan if either compartment never crosses or if the timing/order is invalid.
    """
    x = np.asarray(x_um, dtype=float)
    t = np.asarray(crossing_times_ms, dtype=float)

    if not (0 <= i0 < x.size and 0 <= i1 < x.size):
        raise IndexError("Compartment indices out of range.")

    t0 = t[i0]
    t1 = t[i1]
    if np.isnan(t0) or np.isnan(t1) or t1 <= t0:
        return float("nan")

    dx_m = abs(x[i1] - x[i0]) * 1.0e-6
    dt_s = (t1 - t0) * 1.0e-3
    return dx_m / dt_s


def summarize_cable_run(
    t_ms: Sequence[float],
    V_history_mV: np.ndarray,
    x_um: Sequence[float],
    threshold_mV: float = 0.0,
    probe_indices: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """
    Build a compact diagnostics dictionary for a cable run.

    Includes global extrema, threshold crossings, a simple end-to-end velocity,
    and selected probe summaries.
    """
    t = np.asarray(t_ms, dtype=float)
    V = np.asarray(V_history_mV, dtype=float)
    x = np.asarray(x_um, dtype=float)

    n_comp = V.shape[1]
    if probe_indices is None:
        probe_indices = [0, max(0, n_comp // 4), max(0, n_comp // 2), max(0, (3 * n_comp) // 4), n_comp - 1]

    probe_indices = [int(i) for i in probe_indices if 0 <= int(i) < n_comp]

    peak_V = peak_voltage_by_compartment_mV(V)
    min_V = minimum_voltage_by_compartment_mV(V)
    t_cross = threshold_crossing_times_ms(t, V, threshold_mV=threshold_mV)

    velocity_start_to_end = float("nan")
    if n_comp >= 2:
        velocity_start_to_end = estimate_conduction_velocity_m_per_s(x, t_cross, 0, n_comp - 1)

    probe_summary: List[Dict[str, float]] = []
    for i in probe_indices:
        probe_summary.append(
            {
                "index": i,
                "x_um": float(x[i]),
                "peak_mV": float(peak_V[i]),
                "min_mV": float(min_V[i]),
                "first_crossing_ms": float(t_cross[i]) if not np.isnan(t_cross[i]) else float("nan"),
            }
        )

    return {
        "n_time_samples": int(V.shape[0]),
        "n_compartments": int(n_comp),
        "global_peak_mV": float(np.max(V)),
        "global_min_mV": float(np.min(V)),
        "threshold_mV": float(threshold_mV),
        "crossing_times_ms": t_cross,
        "peak_voltage_by_compartment_mV": peak_V,
        "minimum_voltage_by_compartment_mV": min_V,
        "velocity_start_to_end_m_per_s": velocity_start_to_end,
        "probe_summary": probe_summary,
    }
