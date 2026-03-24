"""
node_diagnostics.py

Diagnostics that focus on nodal compartments only.

This avoids mixing node and internode probes when estimating propagation timing
and conduction velocity.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import math
import numpy as np

def get_node_indices(geometry: Dict[str, np.ndarray]) -> np.ndarray:
    """Return indices of compartments labeled as nodes."""
    region_type = np.asarray(geometry["region_type"], dtype=object)
    return np.where(region_type == "node")[0]

def first_threshold_crossing_time_ms(
    t_ms: np.ndarray,
    V_trace_mV: np.ndarray,
    threshold_mV: float = 0.0,
) -> float:
    """Return the first time a trace crosses the given threshold, or nan."""
    t_ms = np.asarray(t_ms, dtype=float)
    V_trace_mV = np.asarray(V_trace_mV, dtype=float)

    crossing = np.where(V_trace_mV >= float(threshold_mV))[0]
    if crossing.size == 0:
        return math.nan
    return float(t_ms[int(crossing[0])])

def summarize_nodes(
    result: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray],
    threshold_mV: float = 0.0,
) -> List[Dict[str, float]]:
    """
    Build one summary dict per nodal compartment.

    Each row contains node order, compartment index, position, local extrema,
    and first threshold-crossing time.
    """
    t_ms = np.asarray(result["t_ms"], dtype=float)
    V_hist = np.asarray(result["V_m_mV"], dtype=float)
    x_um = np.asarray(geometry["x_um"], dtype=float)
    node_indices = get_node_indices(geometry)

    summaries: List[Dict[str, float]] = []
    for node_order, idx in enumerate(node_indices):
        trace = V_hist[:, idx]
        summaries.append(
            {
                "node_order": float(node_order),
                "compartment_index": float(idx),
                "x_um": float(x_um[idx]),
                "V_peak_mV": float(np.max(trace)),
                "V_min_mV": float(np.min(trace)),
                "first_cross_ms": first_threshold_crossing_time_ms(
                    t_ms=t_ms,
                    V_trace_mV=trace,
                    threshold_mV=threshold_mV,
                ),
            }
        )

    return summaries

def propagation_success(
    node_summaries: List[Dict[str, float]],
    tolerance_ms: float = 1.0e-9,
) -> bool:
    """
    Return True if all nodes cross in nondecreasing time.

    Tiny ties are allowed to avoid overreacting to discrete sampling.
    """
    if len(node_summaries) < 2:
        return False

    times = [float(item["first_cross_ms"]) for item in node_summaries]
    if any(math.isnan(t) for t in times):
        return False

    return all(times[i + 1] >= times[i] - tolerance_ms for i in range(len(times) - 1))

def end_to_end_velocity_m_per_s(node_summaries: List[Dict[str, float]]) -> float:
    """Estimate end-to-end conduction velocity using the first and last nodes."""
    if len(node_summaries) < 2:
        return math.nan

    first = node_summaries[0]
    last = node_summaries[-1]

    t0 = float(first["first_cross_ms"])
    t1 = float(last["first_cross_ms"])

    if math.isnan(t0) or math.isnan(t1) or t1 <= t0:
        return math.nan

    dx_m = (float(last["x_um"]) - float(first["x_um"])) * 1.0e-6
    dt_s = (t1 - t0) * 1.0e-3

    if dt_s <= 0.0:
        return math.nan

    return dx_m / dt_s

def adjacent_node_velocities_m_per_s(node_summaries: List[Dict[str, float]]) -> np.ndarray:
    """Compute adjacent-node velocities, returning nan for invalid pairs."""
    if len(node_summaries) < 2:
        return np.array([], dtype=float)

    velocities = np.full(len(node_summaries) - 1, np.nan, dtype=float)

    for i in range(len(node_summaries) - 1):
        a = node_summaries[i]
        b = node_summaries[i + 1]

        ta = float(a["first_cross_ms"])
        tb = float(b["first_cross_ms"])
        if math.isnan(ta) or math.isnan(tb) or tb <= ta:
            continue

        dx_m = (float(b["x_um"]) - float(a["x_um"])) * 1.0e-6
        dt_s = (tb - ta) * 1.0e-3
        if dt_s > 0.0:
            velocities[i] = dx_m / dt_s

    return velocities

def interior_node_velocity_m_per_s(
    node_summaries: List[Dict[str, float]],
    start_node_order: int,
    end_node_order: int,
) -> float:
    """Compute velocity between two specified node orders."""
    if len(node_summaries) == 0:
        return math.nan
    if start_node_order < 0 or end_node_order >= len(node_summaries):
        return math.nan
    if end_node_order <= start_node_order:
        return math.nan

    a = node_summaries[start_node_order]
    b = node_summaries[end_node_order]

    ta = float(a["first_cross_ms"])
    tb = float(b["first_cross_ms"])
    if math.isnan(ta) or math.isnan(tb) or tb <= ta:
        return math.nan

    dx_m = (float(b["x_um"]) - float(a["x_um"])) * 1.0e-6
    dt_s = (tb - ta) * 1.0e-3
    if dt_s <= 0.0:
        return math.nan

    return dx_m / dt_s

def nodal_peak_statistics(node_summaries: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute summary statistics of nodal peak voltages."""
    if len(node_summaries) == 0:
        return {
            "mean_peak_mV": math.nan,
            "std_peak_mV": math.nan,
            "cv_peak": math.nan,
            "min_peak_mV": math.nan,
            "max_peak_mV": math.nan,
        }

    peaks = np.asarray([float(item["V_peak_mV"]) for item in node_summaries], dtype=float)
    mean_peak = float(np.mean(peaks))
    std_peak = float(np.std(peaks, ddof=0))
    cv_peak = float(std_peak / mean_peak) if mean_peak != 0.0 else math.nan

    return {
        "mean_peak_mV": mean_peak,
        "std_peak_mV": std_peak,
        "cv_peak": cv_peak,
        "min_peak_mV": float(np.min(peaks)),
        "max_peak_mV": float(np.max(peaks)),
    }

def multi_threshold_velocity_summary(
    result: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray],
    thresholds_mV: Sequence[float] = (0.0, 10.0, 20.0),
    mid_start_node: int = 5,
    mid_end_node: int = 15,
) -> List[Dict[str, float]]:
    """Compute nodal velocity summaries for multiple thresholds."""
    summaries: List[Dict[str, float]] = []

    for thr in thresholds_mV:
        node_summaries = summarize_nodes(result=result, geometry=geometry, threshold_mV=float(thr))
        end_v = end_to_end_velocity_m_per_s(node_summaries)
        mid_v = interior_node_velocity_m_per_s(node_summaries, mid_start_node, mid_end_node)
        adj = adjacent_node_velocities_m_per_s(node_summaries)
        finite_adj = adj[np.isfinite(adj)]

        summaries.append(
            {
                "threshold_mV": float(thr),
                "propagation_success": float(propagation_success(node_summaries)),
                "end_to_end_velocity_m_per_s": float(end_v) if not math.isnan(end_v) else math.nan,
                "mid_cable_velocity_m_per_s": float(mid_v) if not math.isnan(mid_v) else math.nan,
                "mean_adjacent_velocity_m_per_s": float(np.mean(finite_adj)) if finite_adj.size > 0 else math.nan,
                "min_adjacent_velocity_m_per_s": float(np.min(finite_adj)) if finite_adj.size > 0 else math.nan,
                "max_adjacent_velocity_m_per_s": float(np.max(finite_adj)) if finite_adj.size > 0 else math.nan,
            }
        )

    return summaries

def format_node_report(
    result: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray],
    threshold_mV: float = 0.0,
) -> str:
    """Format a detailed single-threshold nodal report."""
    summaries = summarize_nodes(result=result, geometry=geometry, threshold_mV=threshold_mV)
    end_velocity = end_to_end_velocity_m_per_s(summaries)
    adj_vel = adjacent_node_velocities_m_per_s(summaries)
    success = propagation_success(summaries)
    peak_stats = nodal_peak_statistics(summaries)

    # Prefer an interior estimate to reduce onset and end-boundary effects.
    if len(summaries) >= 16:
        mid_velocity = interior_node_velocity_m_per_s(summaries, 5, 15)
        mid_label = "nodes 5->15"
    elif len(summaries) >= 4:
        mid_velocity = interior_node_velocity_m_per_s(summaries, 1, len(summaries) - 2)
        mid_label = f"nodes 1->{len(summaries)-2}"
    else:
        mid_velocity = math.nan
        mid_label = "n/a"

    finite_adj = adj_vel[np.isfinite(adj_vel)]

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("Node-only diagnostics")
    lines.append("-" * 72)
    lines.append(f"Node count: {len(summaries)}")
    lines.append(f"Threshold used: {threshold_mV:.3f} mV")
    lines.append(f"Propagation success: {success}")
    lines.append(f"End-to-end velocity: {end_velocity:.6f} m/s" if not math.isnan(end_velocity) else "End-to-end velocity: nan")
    lines.append(f"Interior velocity ({mid_label}): {mid_velocity:.6f} m/s" if not math.isnan(mid_velocity) else f"Interior velocity ({mid_label}): nan")

    if finite_adj.size > 0:
        lines.append(f"Mean adjacent-node velocity: {float(np.mean(finite_adj)):.6f} m/s")
        lines.append(f"Min adjacent-node velocity:  {float(np.min(finite_adj)):.6f} m/s")
        lines.append(f"Max adjacent-node velocity:  {float(np.max(finite_adj)):.6f} m/s")
    else:
        lines.append("Mean adjacent-node velocity: nan")
        lines.append("Min adjacent-node velocity:  nan")
        lines.append("Max adjacent-node velocity:  nan")

    lines.append("-" * 72)
    lines.append("Peak statistics")
    lines.append(f"Mean nodal peak: {peak_stats['mean_peak_mV']:.6f} mV" if not math.isnan(peak_stats["mean_peak_mV"]) else "Mean nodal peak: nan")
    lines.append(f"Std nodal peak:  {peak_stats['std_peak_mV']:.6f} mV" if not math.isnan(peak_stats["std_peak_mV"]) else "Std nodal peak: nan")
    lines.append(f"CV nodal peak:   {peak_stats['cv_peak']:.6f}" if not math.isnan(peak_stats["cv_peak"]) else "CV nodal peak: nan")
    lines.append(f"Min nodal peak:  {peak_stats['min_peak_mV']:.6f} mV" if not math.isnan(peak_stats["min_peak_mV"]) else "Min nodal peak: nan")
    lines.append(f"Max nodal peak:  {peak_stats['max_peak_mV']:.6f} mV" if not math.isnan(peak_stats["max_peak_mV"]) else "Max nodal peak: nan")

    lines.append("-" * 72)
    lines.append("Per-node summary")
    lines.append("node | comp_idx | x_um      | peak_mV   | min_mV    | first_cross_ms")

    for item in summaries:
        node_order = int(item["node_order"])
        comp_idx = int(item["compartment_index"])
        x_um = float(item["x_um"])
        V_peak = float(item["V_peak_mV"])
        V_min = float(item["V_min_mV"])
        t_cross = float(item["first_cross_ms"])

        t_str = f"{t_cross:12.6f}" if not math.isnan(t_cross) else "         nan"
        lines.append(
            f"{node_order:4d} | "
            f"{comp_idx:8d} | "
            f"{x_um:9.3f} | "
            f"{V_peak:9.3f} | "
            f"{V_min:9.3f} | "
            f"{t_str}"
        )

    lines.append("=" * 72)
    return "\n".join(lines)

def format_multi_threshold_report(
    result: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray],
    thresholds_mV: Sequence[float] = (0.0, 10.0, 20.0),
    mid_start_node: int = 5,
    mid_end_node: int = 15,
) -> str:
    """Format a compact velocity comparison across thresholds."""
    rows = multi_threshold_velocity_summary(
        result=result,
        geometry=geometry,
        thresholds_mV=thresholds_mV,
        mid_start_node=mid_start_node,
        mid_end_node=mid_end_node,
    )

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("Multi-threshold velocity summary")
    lines.append("-" * 72)
    lines.append(
        "thr_mV | success | end_to_end_m/s | mid_cable_m/s | "
        "mean_adj_m/s | min_adj_m/s | max_adj_m/s"
    )

    for row in rows:
        thr = row["threshold_mV"]
        success = bool(row["propagation_success"])
        end_v = row["end_to_end_velocity_m_per_s"]
        mid_v = row["mid_cable_velocity_m_per_s"]
        mean_adj = row["mean_adjacent_velocity_m_per_s"]
        min_adj = row["min_adjacent_velocity_m_per_s"]
        max_adj = row["max_adjacent_velocity_m_per_s"]

        def _fmt(x: float) -> str:
            return f"{x:13.6f}" if not math.isnan(x) else "         nan"

        lines.append(
            f"{thr:6.1f} | "
            f"{str(success):7s} | "
            f"{_fmt(end_v)} | "
            f"{_fmt(mid_v)} | "
            f"{_fmt(mean_adj)} | "
            f"{_fmt(min_adj)} | "
            f"{_fmt(max_adj)}"
        )

    lines.append("=" * 72)
    return "\n".join(lines)
