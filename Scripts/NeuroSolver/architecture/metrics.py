"""
metrics.py

Bundle-level architecture metrics for NeuroSolver sweep analysis.

These helpers extract higher-level propagation and recruitment-style quantities
from per-fiber cable results, such as:
- peak Vm per fiber
- threshold crossing times
- latency spread
- recruited fiber count
- conduction velocity estimates
- a simple CNAP-like amplitude proxy from VC traces
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)

def _get_time_by_compartment_matrix(
    cable_result: Dict[str, Any],
    field_key: str = "V_m_mV",
) -> Optional[np.ndarray]:
    """
    Return the requested matrix as shape (time, compartment), transposing if
    needed.

    Expected cases:
    - V.shape == (time, compartment)
    - V.shape == (compartment, time)

    Returns None if the orientation cannot be resolved against t_ms.
    """
    if field_key not in cable_result or "t_ms" not in cable_result:
        return None

    V = _as_float_array(cable_result[field_key])
    t_ms = _as_float_array(cable_result["t_ms"])

    if V.ndim != 2 or t_ms.ndim != 1 or t_ms.size == 0:
        return None

    # Preferred orientation: time x compartment
    if V.shape[0] == t_ms.size:
        return V

    # Alternate orientation: compartment x time
    if V.shape[1] == t_ms.size:
        return V.T

    return None

def _get_spatial_axis_um(cable_result: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Try a few common keys for compartment/node positions.
    """
    for key in (
        "x_um",
        "node_x_um",
        "compartment_x_um",
        "node_positions_um",
        "compartment_positions_um",
    ):
        if key in cable_result:
            arr = _as_float_array(cable_result[key])
            if arr.ndim == 1 and arr.size > 1:
                return arr
    return None

def compute_peak_vm_by_fiber(
    per_fiber_cable_results: Dict[int, Dict[str, Any]],
    field_key: str = "V_m_mV",
) -> Dict[int, float]:
    """
    Return peak field value by fiber ID.
    """
    peaks: Dict[int, float] = {}
    for fid, cable in per_fiber_cable_results.items():
        if field_key not in cable:
            continue

        arr = _as_float_array(cable[field_key])
        if arr.size == 0:
            continue

        peaks[int(fid)] = float(np.max(arr))

    return peaks

def compute_threshold_crossing_time_ms_for_trace(
    t_ms: np.ndarray,
    trace: np.ndarray,
    threshold_mV: float = 0.0,
) -> float:
    """
    Return the first time a 1D trace crosses threshold, else np.nan.
    """
    t_ms = _as_float_array(t_ms)
    trace = _as_float_array(trace)

    if t_ms.ndim != 1 or trace.ndim != 1 or t_ms.size == 0 or trace.size == 0:
        return float(np.nan)

    if t_ms.size != trace.size:
        return float(np.nan)

    idx = np.where(trace >= float(threshold_mV))[0]
    if idx.size == 0:
        return float(np.nan)

    return float(t_ms[int(idx[0])])

def compute_first_threshold_crossing_by_fiber(
    per_fiber_cable_results: Dict[int, Dict[str, Any]],
    threshold_mV: float = 0.0,
    probe_mode: str = "last_compartment",
) -> Dict[int, float]:
    """
    Compute one first-threshold-crossing time per fiber.

    Parameters
    ----------
    probe_mode:
        "max_over_compartments"
            Threshold is checked on the max Vm over compartments at each time.
        "last_compartment"
            Threshold is checked only at the distal compartment.
        "center_compartment"
            Threshold is checked at the center compartment.
        "first_compartment"
            Threshold is checked only at the proximal compartment.
    """
    crossings: Dict[int, float] = {}

    for fid, cable in per_fiber_cable_results.items():
        if "t_ms" not in cable or "V_m_mV" not in cable:
            crossings[int(fid)] = float(np.nan)
            continue

        t_ms = _as_float_array(cable["t_ms"])
        V = _get_time_by_compartment_matrix(cable, field_key="V_m_mV")

        if V is None or V.ndim != 2 or V.shape[0] == 0 or V.shape[1] == 0:
            crossings[int(fid)] = float(np.nan)
            continue

        if probe_mode == "max_over_compartments":
            probe = np.max(V, axis=1)
        elif probe_mode == "last_compartment":
            probe = V[:, -1]
        elif probe_mode == "center_compartment":
            probe = V[:, V.shape[1] // 2]
        elif probe_mode == "first_compartment":
            probe = V[:, 0]
        else:
            raise ValueError(f"Unsupported probe_mode: {probe_mode!r}")

        crossings[int(fid)] = compute_threshold_crossing_time_ms_for_trace(
            t_ms=t_ms,
            trace=probe,
            threshold_mV=threshold_mV,
        )

    return crossings

def compute_recruited_fiber_count(
    crossing_times_ms: Dict[int, float],
) -> int:
    """
    Count fibers with a valid threshold crossing.
    """
    count = 0
    for value in crossing_times_ms.values():
        if np.isfinite(value):
            count += 1
    return int(count)

def compute_latency_stats_ms(
    crossing_times_ms: Dict[int, float],
) -> Dict[str, float]:
    """
    Compute latency summary statistics across recruited fibers.
    """
    vals = np.asarray(
        [float(v) for v in crossing_times_ms.values() if np.isfinite(v)],
        dtype=float,
    )

    if vals.size == 0:
        return {
            "latency_min_ms": float(np.nan),
            "latency_max_ms": float(np.nan),
            "latency_mean_ms": float(np.nan),
            "latency_std_ms": float(np.nan),
            "latency_spread_ms": float(np.nan),
        }

    return {
        "latency_min_ms": float(np.min(vals)),
        "latency_max_ms": float(np.max(vals)),
        "latency_mean_ms": float(np.mean(vals)),
        "latency_std_ms": float(np.std(vals)),
        "latency_spread_ms": float(np.max(vals) - np.min(vals)),
    }

def compute_conduction_velocity_m_per_s_for_fiber(
    cable_result: Dict[str, Any],
    threshold_mV: float = -20.0,
) -> float:
    """
    Estimate conduction velocity from threshold-crossing times across multiple
    compartments.

    Uses a linear fit of position vs crossing time across valid propagating
    compartments. This is more robust than using only two endpoints.
    """
    if "t_ms" not in cable_result or "V_m_mV" not in cable_result:
        return float(np.nan)

    t_ms = _as_float_array(cable_result["t_ms"])
    V = _get_time_by_compartment_matrix(cable_result, field_key="V_m_mV")

    if V is None or V.ndim != 2 or V.shape[0] == 0 or V.shape[1] < 4:
        return float(np.nan)

    x_um = _get_spatial_axis_um(cable_result)
    if x_um is None or x_um.ndim != 1 or x_um.size != V.shape[1]:
        return float(np.nan)

    region_type = cable_result.get("region_type", None)

    # ------------------------------------------------------------------
    # Default fallback: avoid endpoint compartments
    # ------------------------------------------------------------------
    if V.shape[1] >= 5:
        valid_indices = list(range(1, V.shape[1] - 1))
    else:
        valid_indices = list(range(V.shape[1]))

    # ------------------------------------------------------------------
    # Prefer node compartments if available, but avoid first/last nodes
    # ------------------------------------------------------------------
    if region_type is not None:
        region_type_arr = np.asarray(region_type, dtype=object)
        if region_type_arr.ndim == 1 and region_type_arr.size == V.shape[1]:
            node_indices = [
                i for i, name in enumerate(region_type_arr)
                if "node" in str(name).strip().lower()
            ]

            if len(node_indices) >= 4:
                valid_indices = node_indices[1:-1]
            elif len(node_indices) >= 2:
                valid_indices = node_indices

    crossing_times_ms = []
    crossing_positions_um = []

    for i in valid_indices:
        t_cross = compute_threshold_crossing_time_ms_for_trace(
            t_ms=t_ms,
            trace=V[:, i],
            threshold_mV=threshold_mV,
        )
        if np.isfinite(t_cross):
            crossing_times_ms.append(float(t_cross))
            crossing_positions_um.append(float(x_um[i]))

    # Need enough points for a meaningful line fit
    if len(crossing_times_ms) < 3:
        return float(np.nan)

    t_arr_s = np.asarray(crossing_times_ms, dtype=float) * 1.0e-3
    x_arr_m = np.asarray(crossing_positions_um, dtype=float) * 1.0e-6

    # Sort by time for clean propagation ordering
    order = np.argsort(t_arr_s)
    t_arr_s = t_arr_s[order]
    x_arr_m = x_arr_m[order]

    # Reject degenerate cases
    if np.ptp(t_arr_s) <= 1.0e-9:
        return float(np.nan)
    if np.ptp(x_arr_m) <= 1.0e-12:
        return float(np.nan)

    # Fit x = v * t + b
    slope_m_per_s, _intercept = np.polyfit(t_arr_s, x_arr_m, 1)

    if not np.isfinite(slope_m_per_s) or slope_m_per_s <= 0.0:
        return float(np.nan)

    return float(slope_m_per_s)

def compute_conduction_velocity_stats_m_per_s(
    per_fiber_cable_results: Dict[int, Dict[str, Any]],
    threshold_mV: float = -20.0,
) -> Dict[str, float]:
    """
    Compute conduction velocity summary stats across fibers.
    """
    values = []

    for _fid, cable in per_fiber_cable_results.items():
        v = compute_conduction_velocity_m_per_s_for_fiber(
            cable_result=cable,
            threshold_mV=threshold_mV,
        )
        if np.isfinite(v):
            values.append(float(v))

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "cv_mean_m_per_s": float(np.nan),
            "cv_std_m_per_s": float(np.nan),
            "cv_min_m_per_s": float(np.nan),
            "cv_max_m_per_s": float(np.nan),
            "cv_valid_fiber_count": 0.0,
        }

    return {
        "cv_mean_m_per_s": float(np.mean(arr)),
        "cv_std_m_per_s": float(np.std(arr)),
        "cv_min_m_per_s": float(np.min(arr)),
        "cv_max_m_per_s": float(np.max(arr)),
        "cv_valid_fiber_count": float(arr.size),
    }

def compute_cnap_like_amplitude_mV(
    vc_result: Optional[Dict[str, Any]],
) -> float:
    """
    Simple CNAP-like proxy from VC traces.

    Current proxy:
        maximum peak-to-peak amplitude among available electrode traces.
    """
    if vc_result is None:
        return float(np.nan)

    ptps = []
    for key, value in vc_result.items():
        if key == "t_ms":
            continue
        arr = _as_float_array(value)
        if arr.size == 0:
            continue
        ptps.append(float(np.ptp(arr)))

    if len(ptps) == 0:
        return float(np.nan)

    return float(np.max(np.asarray(ptps, dtype=float)))

print("ACTIVE METRICS FILE:", __file__)
def compute_bundle_metrics(
    result,
    threshold_mV: float = -20.0,
) -> Dict[str, float]:
    """
    Extract bundle-level metrics from an ArchitectureSimulationResult-like object.
    """
    per_fiber = result.per_fiber_cable_results

    crossing_times = compute_first_threshold_crossing_by_fiber(
        per_fiber_cable_results=per_fiber,
        threshold_mV=threshold_mV,
        probe_mode="max_over_compartments",
    )

    recruited_count = compute_recruited_fiber_count(crossing_times)
    latency_stats = compute_latency_stats_ms(crossing_times)
    cv_stats = compute_conduction_velocity_stats_m_per_s(
        per_fiber_cable_results=per_fiber,
        threshold_mV=threshold_mV,
    )
    cnap_like = compute_cnap_like_amplitude_mV(result.vc_result)

    metrics: Dict[str, float] = {
        "recruited_fiber_count": float(recruited_count),
        "recruited_fraction": (
            float(recruited_count) / max(float(len(per_fiber)), 1.0)
            if len(per_fiber) > 0
            else float(np.nan)
        ),
        "cnap_like_amplitude_mV": float(cnap_like),
    }

    metrics.update(latency_stats)
    metrics.update(cv_stats)
    return metrics