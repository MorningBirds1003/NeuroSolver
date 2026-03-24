"""
plotting.py

Minimal plotting helpers for custom-architecture NeuroSolver runs.

These routines are intentionally lightweight and operate directly on the current
bundle/cable/KNP/VC result dictionaries returned by the existing scheduler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

FigureAxes = Tuple[plt.Figure, plt.Axes]

def _finalize_axes_equal(ax) -> None:
    ax.set_aspect("equal")
    ax.relim()
    ax.autoscale_view()

def plot_vc_trace_windows(
    vc_result: Dict[str, np.ndarray],
    pulse_start_ms: float,
    pulse_count: int,
    pulse_interval_ms: float,
    window_pre_ms: float = 0.5,
    window_post_ms: float = 2.0,
) -> Dict[str, FigureAxes]:
    """
    Build one VC figure per pulse window.
    """
    figures: Dict[str, FigureAxes] = {}

    for k in range(int(pulse_count)):
        center = float(pulse_start_ms) + k * float(pulse_interval_ms)
        t0 = center - float(window_pre_ms)
        t1 = center + float(window_post_ms)
        figures[f"vc_window_{k+1}"] = plot_vc_traces(
            vc_result,
            t_min_ms=t0,
            t_max_ms=t1,
            title=f"Virtual electrode traces: pulse {k+1}",
        )

    return figures

def symmetric_abs_limit(arrays: list[np.ndarray], floor: float = 1.0e-12) -> tuple[float, float]:
    """
    Return symmetric (-absmax, +absmax) limits across multiple arrays.
    """
    absmax = float(
        max(
            floor,
            max(np.max(np.abs(np.asarray(a, dtype=float))) for a in arrays)
        )
    )
    return -absmax, absmax

def positive_limit(arrays: list[np.ndarray], floor: float = 1.0e-12) -> tuple[float, float]:
    """
    Return (0, max) limits across multiple arrays.
    """
    vmax = float(
        max(
            floor,
            max(np.max(np.asarray(a, dtype=float)) for a in arrays)
        )
    )
    return 0.0, vmax

def save_figure(fig, filepath: str, dpi: int = 200, close: bool = False) -> None:
    """
    Save a figure with tight bounding and optional close.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)

def plot_architecture_cross_section(
    bundle_geometry,
    show_fascicles: bool = True,
    architecture_spec=None,
    show_electrodes: bool = True,
) -> FigureAxes:
    """
    Plot the transverse y-z layout of the explicit fiber placements.

    Parameters
    ----------
    bundle_geometry:
        BundleGeometry object.
    show_fascicles:
        Draw declared fascicle boundaries if available.
    architecture_spec:
        Optional NerveArchitectureSpec. If not provided, attempts to use
        bundle_geometry.metadata["architecture_spec"].
    show_electrodes:
        Overlay point/ring/multicontact electrode positions in the transverse
        plane when architecture metadata is available.
    """
    fig, ax = plt.subplots()

    if architecture_spec is None:
        architecture_spec = bundle_geometry.metadata.get("architecture_spec", None)

    if show_fascicles and architecture_spec is not None:
        for fascicle in getattr(architecture_spec, "fascicles", []):
            boundary = plt.Circle(
                (float(fascicle.center_y_um), float(fascicle.center_z_um)),
                float(fascicle.radius_um),
                fill=False,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )
            ax.add_patch(boundary)
            if fascicle.label:
                ax.text(
                    float(fascicle.center_y_um),
                    float(fascicle.center_z_um) + float(fascicle.radius_um),
                    fascicle.label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    for placement in bundle_geometry.placements:
        geometry = bundle_geometry.get_fiber_geometry(placement.fiber_id)
        outer_diameter = np.asarray(geometry["outer_diameter_um"], dtype=float)
        radius_um = 0.5 * float(np.max(outer_diameter))

        circle = plt.Circle(
            (float(placement.center_y_um), float(placement.center_z_um)),
            radius_um,
            fill=False,
            linewidth=1.2,
        )
        ax.add_patch(circle)
        ax.text(
            float(placement.center_y_um),
            float(placement.center_z_um),
            str(int(placement.fiber_id)),
            ha="center",
            va="center",
            fontsize=8,
        )

    if show_electrodes and architecture_spec is not None:
        for electrode in getattr(architecture_spec, "electrodes", []):
            ey = float(electrode.y_um)
            ez = float(electrode.z_um)
            ax.plot([ey], [ez], marker="x", linestyle="None")
            if electrode.label:
                ax.text(ey, ez, electrode.label, ha="left", va="bottom", fontsize=8)

            if electrode.kind == "ring" and electrode.radius_um is not None:
                ring = plt.Circle(
                    (ey, ez),
                    float(electrode.radius_um),
                    fill=False,
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.8,
                )
                ax.add_patch(ring)

    ax.set_xlabel("y (um)")
    ax.set_ylabel("z (um)")
    ax.set_title(f"Bundle cross-section: {bundle_geometry.metadata.get('bundle_id', 'bundle')}")
    _finalize_axes_equal(ax)
    return fig, ax

def plot_reporting_fiber_propagation(
    cable_result: Dict[str, np.ndarray],
    geometry: Optional[Dict[str, np.ndarray]] = None,
    field_key: str = "V_m_mV",
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> FigureAxes:
    """
    Plot a cable field as an x-t heatmap for one reporting fiber.

    field_key examples
    ------------------
    - "V_m_mV"
    - "V_membrane_effective_mV"  (if present in result)
    """
    if field_key not in cable_result:
        raise KeyError(f"{field_key!r} not found in cable_result.")

    t_ms = np.asarray(cable_result["t_ms"], dtype=float)
    field = np.asarray(cable_result[field_key], dtype=float)
    x_um = np.asarray(
        cable_result["x_um"] if geometry is None else geometry["x_um"],
        dtype=float,
    )

    fig, ax = plt.subplots()
    im = ax.imshow(
        field.T,
        aspect="auto",
        origin="lower",
        extent=[float(t_ms[0]), float(t_ms[-1]), float(x_um[0]), float(x_um[-1])],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("x (um)")
    ax.set_title(title or f"Reporting fiber: {field_key}")
    fig.colorbar(im, ax=ax, label=field_key)
    return fig, ax

def plot_knp_phi_heatmap(
    knp_result: Dict[str, object],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    title: str = "Shared KNP extracellular potential",
) -> FigureAxes:
    """
    Plot shared KNP extracellular potential as an x-t heatmap.
    """
    t_ms = np.asarray(knp_result["t_ms"], dtype=float)
    x_um = np.asarray(knp_result["x_um"], dtype=float)
    phi_e_mV = np.asarray(knp_result["phi_e_mV"], dtype=float)

    fig, ax = plt.subplots()
    im = ax.imshow(
        phi_e_mV.T,
        aspect="auto",
        origin="lower",
        extent=[float(t_ms[0]), float(t_ms[-1]), float(x_um[0]), float(x_um[-1])],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("x (um)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="phi_e (mV)")
    return fig, ax

def plot_knp_species_heatmap(
    knp_result: Dict[str, object],
    species_name: str = "K",
    delta_from_baseline: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> FigureAxes:
    """
    Plot one KNP species concentration history as an x-t heatmap.

    Parameters
    ----------
    delta_from_baseline:
        If provided, subtract this scalar baseline concentration before plotting.
    vmin, vmax:
        Optional fixed color limits for cross-case comparisons.
    cmap:
        Matplotlib colormap name.
    """
    species_histories = knp_result.get("species_mM", None)
    if not isinstance(species_histories, dict) or species_name not in species_histories:
        available = [] if not isinstance(species_histories, dict) else sorted(species_histories.keys())
        raise KeyError(
            f"Species {species_name!r} not found in knp_result['species_mM']. Available: {available}"
        )

    t_ms = np.asarray(knp_result["t_ms"], dtype=float)
    x_um = np.asarray(knp_result["x_um"], dtype=float)
    species_mM = np.asarray(species_histories[species_name], dtype=float)

    label = f"[{species_name}] (mM)"
    title = f"Shared KNP species: {species_name}"

    if delta_from_baseline is not None:
        species_mM = species_mM - float(delta_from_baseline)
        label = f"Δ[{species_name}] (mM)"
        title = f"Shared KNP species delta: {species_name}"

    fig, ax = plt.subplots()
    im = ax.imshow(
        species_mM.T,
        aspect="auto",
        origin="lower",
        extent=[float(t_ms[0]), float(t_ms[-1]), float(x_um[0]), float(x_um[-1])],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("x (um)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    return fig, ax

def plot_vc_traces(
    vc_result: Dict[str, np.ndarray],
    t_min_ms: Optional[float] = None,
    t_max_ms: Optional[float] = None,
    title: str = "Virtual electrode traces",
) -> FigureAxes:
    """
    Plot VC waveforms. The current VC result format is:
        {"t_ms": ..., electrode_name_1: trace, electrode_name_2: trace, ...}

    Optional t_min_ms / t_max_ms let you inspect one pulse window at a time.
    """
    t_ms = np.asarray(vc_result["t_ms"], dtype=float)

    if t_min_ms is None:
        t_min_ms = float(t_ms[0])
    if t_max_ms is None:
        t_max_ms = float(t_ms[-1])

    mask = (t_ms >= float(t_min_ms)) & (t_ms <= float(t_max_ms))
    t_plot = t_ms[mask]

    fig, ax = plt.subplots()
    for key, value in vc_result.items():
        if key == "t_ms":
            continue
        trace = np.asarray(value, dtype=float)
        ax.plot(t_plot, trace[mask], label=str(key))

    ax.set_xlabel("t (ms)")
    ax.set_ylabel("phi_e (mV)")
    ax.set_title(title)
    ax.legend()
    return fig, ax

def plot_fiber_peak_vm_by_id(
    per_fiber_cable_results: Dict[int, Dict[str, np.ndarray]],
    field_key: str = "V_m_mV",
) -> FigureAxes:
    """
    Bar plot of peak field magnitude by fiber ID.
    """
    fiber_ids = sorted(int(fid) for fid in per_fiber_cable_results.keys())
    peaks = []
    for fid in fiber_ids:
        if field_key not in per_fiber_cable_results[fid]:
            raise KeyError(f"{field_key!r} missing from cable result for fiber {fid}.")
        peaks.append(
            float(np.max(np.asarray(per_fiber_cable_results[fid][field_key], dtype=float)))
        )

    fig, ax = plt.subplots()
    ax.bar(fiber_ids, peaks)
    ax.set_xlabel("Fiber ID")
    ax.set_ylabel(f"Peak {field_key}")
    ax.set_title(f"Peak {field_key} by fiber")
    return fig, ax

def plot_reporting_fiber_vm_traces_at_positions(
    cable_result: Dict[str, np.ndarray],
    geometry: Optional[Dict[str, np.ndarray]] = None,
    target_positions_um: tuple[float, ...] = (2000.0, 10000.0, 18000.0),
    field_key: str = "V_m_mV",
    title: str = "Reporting fiber V_m traces at selected x",
) -> FigureAxes:
    """
    Plot V_m(t) at a few selected x-locations along the reporting fiber.

    This is much easier to interpret than the full x-t heatmap when pulse trains
    overlap in space/time.
    """
    if field_key not in cable_result:
        raise KeyError(f"{field_key!r} not found in cable_result.")

    t_ms = np.asarray(cable_result["t_ms"], dtype=float)
    field = np.asarray(cable_result[field_key], dtype=float)
    x_um = np.asarray(
        cable_result["x_um"] if geometry is None else geometry["x_um"],
        dtype=float,
    )

    fig, ax = plt.subplots()

    for x_target in target_positions_um:
        idx = int(np.argmin(np.abs(x_um - float(x_target))))
        ax.plot(t_ms, field[:, idx], label=f"x={x_um[idx]:.0f} um")

    ax.set_xlabel("t (ms)")
    ax.set_ylabel(field_key)
    ax.set_title(title)
    ax.legend()
    return fig, ax

def plot_knp_phi_comparison_shared_limits(
    case_knp_results: Dict[str, Dict[str, object]],
) -> Dict[str, FigureAxes]:
    """
    Plot multiple KNP phi heatmaps with one shared color scale.
    """
    arrays = [
        np.asarray(knp_result["phi_e_mV"], dtype=float)
        for knp_result in case_knp_results.values()
    ]
    vmin, vmax = symmetric_abs_limit(arrays)

    figures: Dict[str, FigureAxes] = {}
    for name, knp_result in case_knp_results.items():
        figures[name] = plot_knp_phi_heatmap(
            knp_result,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            title=f"{name}: Shared KNP extracellular potential",
        )
    return figures

def plot_knp_delta_k_comparison_shared_limits(
    case_knp_results: Dict[str, Dict[str, object]],
    baseline_K_mM: float,
) -> Dict[str, FigureAxes]:
    """
    Plot multiple Δ[K] heatmaps with one shared color scale.
    """
    arrays = []
    for knp_result in case_knp_results.values():
        species_histories = knp_result["species_mM"]
        K_hist = np.asarray(species_histories["K"], dtype=float) - float(baseline_K_mM)
        arrays.append(K_hist)

    vmin, vmax = positive_limit(arrays)

    figures: Dict[str, FigureAxes] = {}
    for name, knp_result in case_knp_results.items():
        figures[name] = plot_knp_species_heatmap(
            knp_result,
            species_name="K",
            delta_from_baseline=float(baseline_K_mM),
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
    return figures

def plot_architecture_result_overview(result) -> Dict[str, FigureAxes]:
    """
    Convenience plotting wrapper for ArchitectureSimulationResult.

    Returns
    -------
    dict
        Keys may include:
        - "architecture"
        - "propagation"
        - "vm_traces"
        - "vc"
        - "knp_phi"
        - "fiber_peaks"
    """
    figures: Dict[str, FigureAxes] = {}

    figures["architecture"] = plot_architecture_cross_section(
        result.bundle_geometry,
        architecture_spec=result.architecture_spec,
    )

    reporting_cable = result.per_fiber_cable_results[result.reporting_fiber_id]

    figures["propagation"] = plot_reporting_fiber_propagation(
        reporting_cable,
        geometry=result.reporting_geometry,
        field_key="V_m_mV",
        title=f"Reporting fiber {result.reporting_fiber_id}: V_m_mV",
    )

    figures["vm_traces"] = plot_reporting_fiber_vm_traces_at_positions(
        reporting_cable,
        geometry=result.reporting_geometry,
        target_positions_um=(2000.0, 10000.0, 18000.0),
        field_key="V_m_mV",
        title=f"Reporting fiber {result.reporting_fiber_id}: V_m(t) at selected x",
    )

    figures["fiber_peaks"] = plot_fiber_peak_vm_by_id(
        result.per_fiber_cable_results,
        field_key="V_m_mV",
    )

    if result.vc_result is not None:
        figures["vc"] = plot_vc_traces(result.vc_result)

        pulse_start_ms = getattr(getattr(result, "metadata", {}), "get", lambda *a, **k: None)("pulse_start_ms", None)
        pulse_count = getattr(getattr(result, "metadata", {}), "get", lambda *a, **k: None)("pulse_count", None)
        pulse_interval_ms = getattr(getattr(result, "metadata", {}), "get", lambda *a, **k: None)("pulse_interval_ms", None)

        if pulse_start_ms is not None and pulse_count is not None and pulse_interval_ms is not None:
            vc_windows = plot_vc_trace_windows(
                result.vc_result,
                pulse_start_ms=float(pulse_start_ms),
                pulse_count=int(pulse_count),
                pulse_interval_ms=float(pulse_interval_ms),
            )
            figures.update(vc_windows)

    if result.knp_result is not None:
        figures["knp_phi"] = plot_knp_phi_heatmap(result.knp_result)

    return figures