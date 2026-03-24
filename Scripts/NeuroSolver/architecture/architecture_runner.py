"""
architecture_runner.py

Top-level runner for custom nerve-architecture simulations in NeuroSolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from Scripts.NeuroSolver.params import DEFAULT_PARAMS, SimulationParameters
from Scripts.NeuroSolver.bundle_state import BundleRuntimeState, initialize_bundle_state
from Scripts.NeuroSolver.scheduler import run_multirate_bundle_simulation

from Scripts.NeuroSolver.architecture.architecture_builder import build_bundle_from_architecture
from Scripts.NeuroSolver.architecture.architecture_schema import NerveArchitectureSpec
from Scripts.NeuroSolver.architecture.electrode_geometry import (
    build_electrode_points_from_architecture,
)
from Scripts.NeuroSolver.architecture.vc_material_model import (
    build_default_peripheral_nerve_material_model,
)
from Scripts.NeuroSolver.architecture.field_mapping import (
    FieldSampleGridSpec,
    build_line_sample_points,
)
from Scripts.NeuroSolver.ECS.vc_solver import sample_phi_e_field_history_mV

@dataclass
class ArchitectureSimulationResult:
    architecture_spec: NerveArchitectureSpec
    bundle_geometry: Any
    bundle_state: BundleRuntimeState
    bundle_result: Any
    reporting_fiber_id: int
    reporting_geometry: Dict[str, Any]
    electrode_points: Sequence[Any]
    metadata: Dict[str, Any]
    material_model: Optional[Any] = None

    @property
    def per_fiber_cable_results(self) -> Dict[int, Dict[str, Any]]:
        return getattr(self.bundle_result, "per_fiber_cable_results", {})

    @property
    def vc_result(self) -> Optional[Dict[str, Any]]:
        return getattr(self.bundle_result, "vc_result", None)

    @property
    def knp_result(self) -> Optional[Dict[str, Any]]:
        return getattr(self.bundle_result, "knp_result", None)

    @property
    def bundle_history(self) -> Dict[str, Any]:
        return getattr(self.bundle_result, "bundle_history", {})

def _resolve_reporting_fiber_id(
    architecture_spec: NerveArchitectureSpec,
    reporting_fiber_id: Optional[int],
) -> int:
    if architecture_spec.fiber_count <= 0:
        raise ValueError("NerveArchitectureSpec must contain at least one fiber.")

    valid_ids = {int(f.fiber_id) for f in architecture_spec.fibers}
    if reporting_fiber_id is None:
        return int(architecture_spec.fibers[0].fiber_id)

    fid = int(reporting_fiber_id)
    if fid not in valid_ids:
        raise ValueError(
            f"reporting_fiber_id={fid} is not present in the architecture spec. "
            f"Valid IDs: {sorted(valid_ids)}"
        )
    return fid

def run_architecture_simulation(
    architecture_spec: NerveArchitectureSpec,
    params: SimulationParameters = DEFAULT_PARAMS,
    reporting_fiber_id: Optional[int] = None,
    stimulated_fiber_ids: Optional[Sequence[int]] = None,
    enable_vc: Optional[bool] = None,
    enable_knp: Optional[bool] = None,
    enable_ecs_feedback: Optional[bool] = None,
    knp_species: str = "K",
    bundle_state: Optional[BundleRuntimeState] = None,
) -> ArchitectureSimulationResult:
    architecture_spec.validate()
    fid_report = _resolve_reporting_fiber_id(architecture_spec, reporting_fiber_id)

    bundle_geometry = build_bundle_from_architecture(architecture_spec, params=params)
    # Build a lightweight material-aware VC model.
    # For now this is a concentric-layer approximation, not a FEM field solve.
    fascicle_radius_um = float(architecture_spec.fascicles[0].radius_um)

    cuff_inner_radius_um = None
    cuff_thickness_um = 0.0
    if len(architecture_spec.cuffs) > 0:
        cuff_inner_radius_um = float(architecture_spec.cuffs[0].inner_radius_um)
        cuff_thickness_um = float(architecture_spec.cuffs[0].thickness_um)

    if architecture_spec.outer_nerve_radius_um is not None:
        outer_nerve_radius_um = float(architecture_spec.outer_nerve_radius_um)
    elif cuff_inner_radius_um is not None:
        # If a cuff exists, keep the nerve just inside it.
        outer_nerve_radius_um = 0.95 * float(cuff_inner_radius_um)
    else:
        # Minimal fallback for simple demo architectures with one explicit fascicle.
        # Keep a small outer-nerve shell outside the fascicle boundary.
        outer_nerve_radius_um = 1.25 * float(fascicle_radius_um)

    material_model = build_default_peripheral_nerve_material_model(
        fascicle_radius_um=fascicle_radius_um,
        outer_nerve_radius_um=outer_nerve_radius_um,
        cuff_inner_radius_um=cuff_inner_radius_um,
        cuff_thickness_um=cuff_thickness_um,
        center_y_um=0.0,
        center_z_um=0.0,
    )

    if bundle_state is None:
        bundle_state = initialize_bundle_state(bundle_geometry=bundle_geometry, params=params)

    electrode_points = build_electrode_points_from_architecture(
        architecture_spec,
        ring_min_points=8,
        include_explicit_electrodes=True,
        include_cuffs=True,
        cuff_mode="ring",
        cuff_ring_contact_count=8,
    )

    bundle_result = run_multirate_bundle_simulation(
        bundle_geometry=bundle_geometry,
        bundle_state=bundle_state,
        params=params,
        electrode_points=electrode_points,
        stimulated_fiber_ids=stimulated_fiber_ids,
        enable_vc=enable_vc,
        enable_knp=enable_knp,
        enable_ecs_feedback=enable_ecs_feedback,
        knp_species=knp_species,
        material_model=material_model
    )

    reporting_geometry = bundle_geometry.get_fiber_geometry(fid_report)
    result_metadata = getattr(bundle_result, "metadata", {}) or {}
    # feedback fallback 
    last_phi_stats = result_metadata.get("feedback_last_phi_stats_mV", None)
    if last_phi_stats is None:
        by_fiber = result_metadata.get("feedback_last_phi_stats_mV_by_fiber", None)
        if isinstance(by_fiber, dict):
            last_phi_stats = by_fiber.get(fid_report, None)

    last_reversals = result_metadata.get("feedback_last_reversals_mV", None)
    if last_reversals is None:
        by_fiber = result_metadata.get("feedback_last_reversals_mV_by_fiber", None)
        if isinstance(by_fiber, dict):
            last_reversals = by_fiber.get(fid_report, None)

    metadata = {
    "bundle_id": bundle_geometry.metadata.get("bundle_id", architecture_spec.bundle_id),
    "layout_name": getattr(bundle_geometry, "layout_name", architecture_spec.layout_name),
    "fiber_count": int(getattr(bundle_geometry, "total_fiber_count", architecture_spec.fiber_count)),
    "reporting_fiber_id": int(fid_report),
    "stimulated_fiber_ids": None if stimulated_fiber_ids is None else [int(fid) for fid in stimulated_fiber_ids],
    "electrode_count": int(len(electrode_points)),
    "enable_vc": bool(result_metadata.get("enable_vc", False)) if result_metadata else bool(enable_vc),
    "enable_knp": bool(result_metadata.get("enable_knp", False)) if result_metadata else bool(enable_knp),
    "feedback_enabled": bool(result_metadata.get("feedback_enabled", False)) if result_metadata else bool(enable_ecs_feedback),

    "feedback_last_phi_stats_mV": last_phi_stats,
    "feedback_last_reversals_mV": last_reversals,
    
    "knp_species": str(knp_species),
    "vc_material_model": (
        "material_aware_shielded_concentric_layers"
        if material_model is not None
        else "homogeneous"
    ),
    "t_stop_ms": float(getattr(params.solver, "t_stop_ms", np.nan)),
    "dt_fast_ms": float(getattr(params.solver, "dt_fast_ms", np.nan)),
    "dt_slow_ms": float(getattr(params.solver, "dt_slow_ms", np.nan)),
}

    return ArchitectureSimulationResult(
        architecture_spec=architecture_spec,
        bundle_geometry=bundle_geometry,
        bundle_state=bundle_state,
        bundle_result=bundle_result,
        reporting_fiber_id=fid_report,
        reporting_geometry=reporting_geometry,
        electrode_points=electrode_points,
        metadata=metadata,
        material_model=material_model,
    )

def build_bundle_phi_e_xt_map(
    result: ArchitectureSimulationResult,
    params: SimulationParameters = DEFAULT_PARAMS,
    line_spec: Optional[FieldSampleGridSpec] = None,
    material_model: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """
    Build a bundle-level phi_e(x,t) map by summing field histories from all fibers
    on a regular x-line.
    """
    if line_spec is None:
        line_spec = FieldSampleGridSpec(
            name="bundle_centerline",
            x_min_um=0.0,
            x_max_um=float(result.architecture_spec.length_um),
            n_x=40,
            y_um=100.0,
            z_um=0.0,
        )

    print(
        f"[field_map] start bundle={result.architecture_spec.bundle_id} "
        f"n_fibers={len(result.per_fiber_cable_results)} "
        f"n_x={int(line_spec.n_x)} "
        f"line_y={float(line_spec.y_um):.1f} "
        f"line_z={float(line_spec.z_um):.1f}"
    )

    x_um, points = build_line_sample_points(line_spec)

    summed_phi = None
    t_ms_ref = None
    total_fibers = len(result.per_fiber_cable_results)

    for k, (fid, cable_result) in enumerate(result.per_fiber_cable_results.items(), start=1):
        print(f"[field_map] sampling fiber {k}/{total_fibers} (fid={fid})")

        geometry = result.bundle_geometry.get_fiber_geometry(fid)

        fiber_field = sample_phi_e_field_history_mV(
            cable_result=cable_result,
            geometry=geometry,
            sample_points=points,
            params=params,
            cable_y_um=float(geometry.get("fiber_center_y_um", 0.0)),
            cable_z_um=float(geometry.get("fiber_center_z_um", 0.0)),
            material_model=material_model,
        )

        if summed_phi is None:
            summed_phi = np.asarray(fiber_field["phi_e_mV"], dtype=float)
            t_ms_ref = np.asarray(fiber_field["t_ms"], dtype=float)
            print(
                f"[field_map] initialized map "
                f"shape={summed_phi.shape} "
                f"time_samples={t_ms_ref.size}"
            )
        else:
            summed_phi += np.asarray(fiber_field["phi_e_mV"], dtype=float)

    if summed_phi is None or t_ms_ref is None:
        raise ValueError("No per-fiber cable results were available for field mapping.")

    print(
        f"[field_map] done bundle={result.architecture_spec.bundle_id} "
        f"shape={summed_phi.shape}"
    )

    return {
        "t_ms": t_ms_ref,
        "x_um": x_um,
        "phi_e_mV": summed_phi,
        "y_um": np.full_like(x_um, float(line_spec.y_um), dtype=float),
        "z_um": np.full_like(x_um, float(line_spec.z_um), dtype=float),
    }

def summarize_architecture_run(result: ArchitectureSimulationResult) -> Dict[str, Any]:
    peak_vm_by_fiber: Dict[int, float] = {}
    for fid, cable in result.per_fiber_cable_results.items():
        vm = cable.get("V_m_mV", None)
        if vm is None:
            continue
        peak_vm_by_fiber[int(fid)] = float(np.max(np.asarray(vm, dtype=float)))

    summary: Dict[str, Any] = {
        "bundle_id": result.metadata.get("bundle_id"),
        "fiber_count": int(result.metadata.get("fiber_count", 0)),
        "reporting_fiber_id": int(result.reporting_fiber_id),
        "electrode_count": int(result.metadata.get("electrode_count", 0)),
        "peak_vm_by_fiber_mV": peak_vm_by_fiber,
    }

    if result.knp_result is not None:
        phi_e = result.knp_result.get("phi_e_mV", None)
        t_ms = result.knp_result.get("t_ms", None)
        species_mM = result.knp_result.get("species_mM", None)

        if phi_e is not None:
            phi_arr = np.asarray(phi_e, dtype=float)
            summary["knp_phi_min_mV"] = float(np.min(phi_arr))
            summary["knp_phi_max_mV"] = float(np.max(phi_arr))
            summary["knp_phi_final_min_mV"] = float(np.min(phi_arr[-1]))
            summary["knp_phi_final_max_mV"] = float(np.max(phi_arr[-1]))
            summary["knp_phi_final_mean_mV"] = float(np.mean(phi_arr[-1]))

            if t_ms is not None:
                t_arr = np.asarray(t_ms, dtype=float)
                if t_arr.ndim == 1 and t_arr.size == phi_arr.shape[0] and t_arr.size >= 2:
                    peak_abs_phi_t = np.max(np.abs(phi_arr), axis=1)
                    peak_idx = int(np.argmax(peak_abs_phi_t))
                    summary["knp_phi_time_to_peak_ms"] = float(t_arr[peak_idx])
                    summary["knp_phi_abs_time_integral_mV_ms"] = float(
                        np.trapezoid(peak_abs_phi_t, t_arr)
                    )

        if isinstance(species_mM, dict) and "K" in species_mM:
            K_hist = np.asarray(species_mM["K"], dtype=float)
            K_baseline = float(result.bundle_result.metadata.get("K_baseline_mM", 4.0)) \
                if hasattr(result.bundle_result, "metadata") else 4.0
            delta_K = K_hist - K_baseline

            summary["delta_K_peak_abs_mM"] = float(np.max(np.abs(delta_K)))
            summary["delta_K_final_max_mM"] = float(np.max(delta_K[-1]))
            summary["delta_K_final_mean_mM"] = float(np.mean(delta_K[-1]))
            summary["delta_K_final_l2_mM"] = float(np.linalg.norm(delta_K[-1]))

            if t_ms is not None:
                t_arr = np.asarray(t_ms, dtype=float)
                if t_arr.ndim == 1 and t_arr.size == K_hist.shape[0] and t_arr.size >= 2:
                    peak_abs_deltaK_t = np.max(np.abs(delta_K), axis=1)
                    peak_idx = int(np.argmax(peak_abs_deltaK_t))
                    summary["delta_K_time_to_peak_ms"] = float(t_arr[peak_idx])
                    summary["delta_K_abs_time_integral_mM_ms"] = float(
                        np.trapezoid(peak_abs_deltaK_t, t_arr)
                    )

        source_debug = result.knp_result.get("source_debug", None)
        if isinstance(source_debug, dict):
            summary["knp_max_abs_source_mM_per_ms"] = float(
                source_debug.get("max_abs_source_mM_per_ms", 0.0)
            )

        source_hist = result.knp_result.get("source_history_mM_per_ms", None)
        if source_hist is not None and t_ms is not None:
            src = np.asarray(source_hist, dtype=float)
            t_arr = np.asarray(t_ms, dtype=float)
            if src.ndim >= 2 and t_arr.ndim == 1 and t_arr.size == src.shape[0]:
                peak_abs_src_t = np.max(np.abs(src.reshape(src.shape[0], -1)), axis=1)
                summary["knp_source_abs_time_integral_mM"] = float(
                    np.trapezoid(peak_abs_src_t, t_arr)
                )

    if result.vc_result is not None:
        peak_vc_by_electrode = {}
        for key, value in result.vc_result.items():
            if key == "t_ms":
                continue
            arr = np.asarray(value, dtype=float)
            peak_vc_by_electrode[str(key)] = float(np.max(np.abs(arr)))
        summary["peak_vc_by_electrode_mV"] = peak_vc_by_electrode

    summary["feedback_enabled"] = bool(
        getattr(result, "metadata", {}).get("feedback_enabled", False)
    )

    last_phi_stats = getattr(result, "metadata", {}).get("feedback_last_phi_stats_mV", None)
    if isinstance(last_phi_stats, dict):
        summary["feedback_phi_mean_mV"] = float(last_phi_stats.get("mean", np.nan))
        summary["feedback_phi_min_mV"] = float(last_phi_stats.get("min", np.nan))
        summary["feedback_phi_max_mV"] = float(last_phi_stats.get("max", np.nan))
        summary["feedback_phi_peak_abs_mV"] = float(
            max(abs(summary["feedback_phi_min_mV"]), abs(summary["feedback_phi_max_mV"]))
        )
    else:
        summary["feedback_phi_mean_mV"] = float(np.nan)
        summary["feedback_phi_min_mV"] = float(np.nan)
        summary["feedback_phi_max_mV"] = float(np.nan)
        summary["feedback_phi_peak_abs_mV"] = float(np.nan)

    last_rev = getattr(result, "metadata", {}).get("feedback_last_reversals_mV", None)
    if isinstance(last_rev, dict):
        for key, value in last_rev.items():
            summary[f"feedback_{key}"] = float(value)

    return summary