"""
regression_cases.py

Phase 0 reference-case helpers for NeuroSolver.

Purpose
-------
Define compact, reproducible anchor cases that can be rerun after code changes.
These are not full experiments; they are regression fixtures used to verify that
basic propagation, VC, and KNP behavior remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from Scripts.NeuroSolver.baseline_io import save_json
from Scripts.NeuroSolver.params import SimulationParameters, build_medium_myelinated_physiology_params
from Scripts.NeuroSolver.propagation.myelin_geometry import build_node_internode_geometry
from Scripts.NeuroSolver.propagation.node_diagnostics import summarize_nodes, propagation_success, end_to_end_velocity_m_per_s, interior_node_velocity_m_per_s, nodal_peak_statistics, multi_threshold_velocity_summary
from Scripts.NeuroSolver.scheduler import run_multirate_simulation
from Scripts.NeuroSolver.ECS.vc_solver import ElectrodeSamplePoint

@dataclass(frozen=True)
class RegressionCase:
    """
    Self-contained regression case description.

    A case bundles the parameter set, resolved geometry, and any electrode sample
    points needed to reproduce a known reference run.
    """
    name: str
    description: str
    params: SimulationParameters
    geometry: Dict[str, np.ndarray]
    electrode_points: List[ElectrodeSamplePoint]

def build_reference_anchor_case(params: Optional[SimulationParameters] = None) -> RegressionCase:
    """
    Build a compact single-fiber physiological anchor case.

    Why compact?
    ------------
    Shorter geometries and shorter runtimes reduce regression cost while still
    exercising the cable, VC, and slow ECS plumbing.
    """
    anchor_params = build_medium_myelinated_physiology_params() if params is None else params
    anchor_params = replace(
        anchor_params,
        geometry=replace(anchor_params.geometry, use_total_length_driven_topology=False, internode_length_um=800.0, total_length_mm=5.0),
        topology=replace(anchor_params.topology, node_count=9),
        solver=replace(anchor_params.solver, t_stop_ms=8.0),
        stimulus=replace(anchor_params.stimulus, pulse_amplitude_uA=0.0015),
        validation=replace(anchor_params.validation, expected_peak_max_mV=100.0),
    )
    geometry = build_node_internode_geometry(anchor_params)
    x_mid = float(np.median(np.asarray(geometry["x_um"], dtype=float)))
    electrodes = [
        ElectrodeSamplePoint(name="vc_near_mid", x_um=x_mid, y_um=75.0, z_um=0.0),
        ElectrodeSamplePoint(name="vc_far_mid", x_um=x_mid, y_um=250.0, z_um=0.0),
    ]
    return RegressionCase(
        name="single_fiber_reference_anchor",
        description="Compact physiology-first medium myelinated single-fiber anchor used for regression.",
        params=anchor_params,
        geometry=geometry,
        electrode_points=electrodes,
    )

def summarize_reference_run(result, geometry: Dict[str, np.ndarray], params: SimulationParameters) -> Dict[str, object]:
    """
    Reduce a full simulation result to a regression-friendly summary payload.

    The goal is to keep the snapshot small, stable, and interpretable while
    preserving the most load-bearing metrics:
    resting voltage, peak voltage, propagation success, velocity, VC response,
    and slow KNP drift.
    """
    cable_result = result.cable_result
    node_summaries = summarize_nodes(cable_result, geometry, threshold_mV=float(params.validation.spike_detection_threshold_mV))
    peak_stats = nodal_peak_statistics(node_summaries)
    end_v = end_to_end_velocity_m_per_s(node_summaries)

    # Use an interior-node velocity estimate when enough nodes exist. This tends
    # to be more robust than only using the very first and very last active node.
    n_nodes = len(node_summaries)
    mid_start = min(2, max(0, n_nodes - 2))
    mid_end = max(mid_start + 1, min(6, max(1, n_nodes - 1)))

    interior_v = interior_node_velocity_m_per_s(node_summaries, mid_start, mid_end)
    V_hist = np.asarray(cable_result["V_m_mV"], dtype=float)

    summary: Dict[str, object] = {
        "case_name": getattr(getattr(params, "preset", None), "active_preset_name", "unspecified"),
        "n_compartments": int(geometry["n_compartments"]),
        "n_nodes": int(len(np.asarray(geometry.get("node_indices", []), dtype=int))),
        "resting_voltage_mean_mV": float(np.mean(V_hist[0, :])),
        "peak_voltage_max_mV": float(np.max(V_hist)),
        "peak_voltage_min_mV": float(np.min(V_hist)),
        "propagation_success": bool(propagation_success(node_summaries)),
        "end_to_end_velocity_m_per_s": float(end_v) if np.isfinite(end_v) else None,
        "interior_velocity_m_per_s": float(interior_v) if np.isfinite(interior_v) else None,
        "nodal_peak_statistics": peak_stats,
        "multi_threshold_velocity_summary": multi_threshold_velocity_summary(cable_result, geometry, mid_start_node=mid_start, mid_end_node=mid_end),
        "vc_summary": {},
        "knp_summary": {},
        "metadata": dict(result.metadata or {}),
    }

    if result.vc_result is not None:
        vc_summary: Dict[str, object] = {}
        for key, value in result.vc_result.items():
            if key == "t_ms":
                continue
            trace = np.asarray(value, dtype=float)
            vc_summary[key] = {
                "peak_abs_mV": float(np.max(np.abs(trace))),
                "peak_mV": float(np.max(trace)),
                "trough_mV": float(np.min(trace)),
            }
        summary["vc_summary"] = vc_summary

    if result.knp_result is not None:
        phi_hist = np.asarray(result.knp_result.get("phi_e_mV", []), dtype=float)
        species_hist = result.knp_result.get("species_mM", {})
        delta_summary: Dict[str, float] = {}

        baseline_species = {
            "Na": float(params.ions.sodium.extracellular_mM),
            "K": float(params.ions.potassium.extracellular_mM),
            "Cl": float(params.ions.chloride.extracellular_mM),
            "Ca": float(params.ions.calcium.extracellular_mM),
        }

        for name, arr in species_hist.items():
            arr_np = np.asarray(arr, dtype=float)
            if arr_np.size:
                delta_summary[name] = float(np.max(np.abs(arr_np - baseline_species.get(name, 0.0))))

        summary["knp_summary"] = {
            "phi_peak_abs_mV": float(np.max(np.abs(phi_hist))) if phi_hist.size else 0.0,
            "source_debug": result.knp_result.get("source_debug", {}),
            "species_max_abs_delta_mM": delta_summary,
        }

    return summary

def run_reference_anchor_case(params: Optional[SimulationParameters] = None, save_baseline_path: Optional[str | Path] = None) -> Dict[str, object]:
    """
    Build, run, summarize, and optionally save the anchor regression case.
    """
    case = build_reference_anchor_case(params=params)
    result = run_multirate_simulation(
        geometry=case.geometry,
        params=case.params,
        electrode_points=case.electrode_points,
        enable_vc=None,
        enable_knp=None,
        enable_ecs_feedback=None,
    )
    summary = summarize_reference_run(result, case.geometry, case.params)
    summary["description"] = case.description

    if save_baseline_path is not None:
        save_json(save_baseline_path, summary)

    return {"case": case, "result": result, "summary": summary}
