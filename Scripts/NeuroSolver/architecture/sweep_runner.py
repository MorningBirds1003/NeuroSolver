"""
sweep_runner.py

Batch runner for NeuroSolver custom-architecture studies.

This script is designed to answer questions like:
- how do VC amplitude and KNP response scale with fiber count?
- how does spacing affect shared ECS effects?
- how does electrode offset affect readout strength?

Outputs
-------
1. Per-case figure folders
2. A CSV summary table with one row per run
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from Scripts.NeuroSolver.params import (
    DEFAULT_PARAMS,
    SimulationParameters,
    build_medium_myelinated_physiology_params,
)
from Scripts.NeuroSolver.architecture.architecture_runner import (
    run_architecture_simulation,
    summarize_architecture_run,
)
from Scripts.NeuroSolver.architecture.architecture_schema import (
    CuffSpec,
    ElectrodeSpec,
    FascicleSpec,
    FiberSpec,
    NerveArchitectureSpec,
)
from Scripts.NeuroSolver.architecture.plotting import (
    plot_architecture_result_overview,
    plot_knp_species_heatmap,
    save_figure,
)
from Scripts.NeuroSolver.architecture.postprocess_sweep import export_standard_sweep_plots
from Scripts.NeuroSolver.architecture.metrics import compute_bundle_metrics
from Scripts.NeuroSolver.architecture.field_mapping import FieldSampleGridSpec
from Scripts.NeuroSolver.architecture.field_postprocess import (
    plot_phi_e_xt_heatmap,
    plot_delta_phi_xt_heatmap,
)
from Scripts.NeuroSolver.architecture.architecture_runner import build_bundle_phi_e_xt_map

# -----------------------------------------------------------------------------
# Parameter helpers
# -----------------------------------------------------------------------------
def build_fast_sweep_params() -> SimulationParameters:
    """
    Reduced-cost parameters for architecture sweeps.

    Start here, then move to a fuller physiological run after confirming trends.
    """
    params = build_medium_myelinated_physiology_params(DEFAULT_PARAMS)

    solver = replace(
        params.solver,
        t_stop_ms=2.0,
        dt_fast_ms=0.01,
        dt_slow_ms=0.10,
    )
    topology = replace(
        params.topology,
        node_count=11,
    )

    return replace(params, solver=solver, topology=topology)

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)

def _safe_bool(value: Any, default: bool = False) -> bool:
    try:
        return bool(value)
    except Exception:
        return bool(default)

def _validate_cuff_consistency(spec: NerveArchitectureSpec) -> None:
    has_cuff_objects = len(spec.cuffs) > 0
    meta_include_cuff = bool(spec.metadata.get("include_cuff", False))

    if has_cuff_objects != meta_include_cuff:
        raise ValueError(
            f"Cuff metadata mismatch for {spec.bundle_id}: "
            f"metadata include_cuff={meta_include_cuff}, "
            f"but len(spec.cuffs)={len(spec.cuffs)}"
        )

def _validate_postrun_cuff_consistency(result) -> None:
    spec = result.architecture_spec
    has_cuff_objects = len(spec.cuffs) > 0
    meta_include_cuff = bool(spec.metadata.get("include_cuff", False))

    if has_cuff_objects != meta_include_cuff:
        raise RuntimeError(
            f"Post-run cuff metadata mismatch for {spec.bundle_id}: "
            f"metadata include_cuff={meta_include_cuff}, "
            f"but len(result.architecture_spec.cuffs)={len(spec.cuffs)}"
        )

def _extract_run_condition_metadata(
    result,
    params: Optional[SimulationParameters] = None,
    enable_vc: Optional[bool] = None,
    enable_knp: Optional[bool] = None,
    enable_ecs_feedback: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Extract execution-condition metadata robustly.

    Prefer result.metadata when available, but fall back to params / call args.
    """
    result_meta = getattr(result, "metadata", {}) or {}

    row: Dict[str, Any] = {}
    row["t_stop_ms"] = _safe_float(
        result_meta.get("t_stop_ms", getattr(getattr(params, "solver", None), "t_stop_ms", np.nan))
    )
    row["dt_fast_ms"] = _safe_float(
        result_meta.get("dt_fast_ms", getattr(getattr(params, "solver", None), "dt_fast_ms", np.nan))
    )
    row["dt_slow_ms"] = _safe_float(
        result_meta.get("dt_slow_ms", getattr(getattr(params, "solver", None), "dt_slow_ms", np.nan))
    )

    row["enable_vc"] = _safe_bool(result_meta.get("enable_vc", enable_vc))
    row["enable_knp"] = _safe_bool(result_meta.get("enable_knp", enable_knp))
    row["feedback_enabled"] = _safe_bool(
        result_meta.get("feedback_enabled", enable_ecs_feedback)
    )

    ecs_params = getattr(params, "ecs", None)
    row["ecs_alpha"] = _safe_float(
        result_meta.get(
            "ecs_alpha",
            getattr(ecs_params, "alpha", getattr(ecs_params, "volume_fraction", np.nan)),
        )
    )
    row["ecs_lambda"] = _safe_float(
        result_meta.get(
            "ecs_lambda",
            getattr(ecs_params, "lambda_tortuosity", getattr(ecs_params, "tortuosity", np.nan)),
        )
    )
    row["vc_material_model"] = str(
        result_meta.get(
            "vc_material_model",
            getattr(result, "metadata", {}).get("vc_material_model", "unknown"),
        )
    )    

    return row

def debug_field_line_peaks(
    result,
    params,
    material_model=None,
    y_offsets_um=(0.0, 50.0, 100.0, 200.0),
):
    for y_um in y_offsets_um:
        line_spec = FieldSampleGridSpec(
            name=f"debug_y{int(round(y_um))}",
            x_min_um=0.0,
            x_max_um=float(result.architecture_spec.length_um),
            n_x=60,
            y_um=float(y_um),
            z_um=0.0,
        )
        field_result = build_bundle_phi_e_xt_map(
            result=result,
            params=params,
            line_spec=line_spec,
            material_model=material_model,
        )
        phi = np.asarray(field_result["phi_e_mV"], dtype=float)
        dphi = phi - phi[0:1, :]
        print(
            f"sample line y={y_um:6.1f} um | "
            f"max|phi|={np.max(np.abs(phi)):.6f} mV | "
            f"max|delta_phi|={np.max(np.abs(dphi)):.6f} mV"
        )

def save_field_maps(
    result,
    output_dir: Path,
    params: SimulationParameters,
    material_model: Optional[Any] = None,
    sample_y_um: float = 100.0,
    sample_z_um: float = 0.0,
) -> None:
    """
    Save bundle-level VC-style spatiotemporal phi_e(x,t) maps for one run.

    Notes
    -----
    These maps are generated by sampling and summing per-fiber extracellular
    fields along a regular x-line. They are distinct from shared KNP phi_e maps.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    line_spec = FieldSampleGridSpec(
        name="bundle_centerline",
        x_min_um=0.0,
        x_max_um=float(result.architecture_spec.length_um),
        n_x=200,
        y_um=float(sample_y_um),
        z_um=float(sample_z_um),
    )

    resolved_material_model = (
        material_model
        if material_model is not None
        else getattr(result, "material_model", None)
    )
    field_result = build_bundle_phi_e_xt_map(
        result=result,
        params=params,
        line_spec=line_spec,
        material_model=resolved_material_model,
    )

    np.savez(
        output_dir / "phi_e_xt_map.npz",
        t_ms=field_result["t_ms"],
        x_um=field_result["x_um"],
        phi_e_mV=field_result["phi_e_mV"],
        y_um=field_result["y_um"],
        z_um=field_result["z_um"],
    )

    plot_phi_e_xt_heatmap(
        field_result,
        output_path=output_dir / "phi_e_xt_heatmap.png",
        title=f"{result.architecture_spec.bundle_id}: VC bundle phi_e(x,t)",
    )
    plot_delta_phi_xt_heatmap(
        field_result,
        output_path=output_dir / "delta_phi_e_xt_heatmap.png",
        title=f"{result.architecture_spec.bundle_id}: VC bundle delta phi_e(x,t)",
    )

# -----------------------------------------------------------------------------
# Architecture generators
# -----------------------------------------------------------------------------
def make_linear_architecture(
    *,
    fiber_count: int,
    spacing_um: float,
    length_um: float = 20000.0,
    fascicle_radius_um: Optional[float] = None,
    electrode_y_offsets_um: Sequence[float] = (100.0, 250.0),
    electrode_x_um: Sequence[float] = (5000.0, 10000.0),
    bundle_id: Optional[str] = None,
    layout_name: str = "linear_sweep",
    fiber_diameter_um: Optional[float] = None,
    axon_diameter_um: Optional[float] = None,
    include_cuff: bool = True,
    cuff_inner_radius_um: Optional[float] = None,
    cuff_thickness_um: float = 100.0,
    cuff_center_y_um: float = 0.0,
    cuff_center_z_um: float = 0.0,
) -> NerveArchitectureSpec:
    """
    Create a simple linear cross-sectional architecture.

    Fibers are placed along y with z = 0.
    """
    if fiber_count <= 0:
        raise ValueError("fiber_count must be positive.")
    if spacing_um <= 0.0:
        raise ValueError("spacing_um must be positive.")
    if len(electrode_x_um) != len(electrode_y_offsets_um):
        raise ValueError(
            "electrode_x_um and electrode_y_offsets_um must have the same length."
        )

    if fascicle_radius_um is None:
        fascicle_radius_um = max(160.0, 0.5 * (fiber_count - 1) * spacing_um + 80.0)

    if include_cuff and cuff_inner_radius_um is None:
        cuff_inner_radius_um = float(fascicle_radius_um) + 120.0

    half = 0.5 * (fiber_count - 1)
    fibers: List[FiberSpec] = []
    for i in range(fiber_count):
        y_um = (i - half) * spacing_um
        fibers.append(
            FiberSpec(
                fiber_id=i,
                fascicle_id=0,
                center_y_um=float(y_um),
                center_z_um=0.0,
                fiber_diameter_um=fiber_diameter_um,
                axon_diameter_um=axon_diameter_um,
                label=f"fiber_{i}",
            )
        )

    electrodes: List[ElectrodeSpec] = []
    for idx, (x_um, y_um) in enumerate(zip(electrode_x_um, electrode_y_offsets_um)):
        electrodes.append(
            ElectrodeSpec(
                kind="point",
                x_um=float(x_um),
                y_um=float(y_um),
                z_um=0.0,
                label=f"E_{idx}",
            )
        )

    cuffs: List[CuffSpec] = []
    if include_cuff:
        cuffs.append(
            CuffSpec(
                cuff_id="cuff_0",
                center_y_um=float(cuff_center_y_um),
                center_z_um=float(cuff_center_z_um),
                inner_radius_um=float(cuff_inner_radius_um),
                thickness_um=float(cuff_thickness_um),
                material_name="cuff_polymer",
                label="ring_cuff",
            )
        )

    spec = NerveArchitectureSpec(
        bundle_id=bundle_id or f"linear_fc{fiber_count}_sp{int(round(spacing_um))}",
        length_um=float(length_um),
        outer_nerve_radius_um=float(fascicle_radius_um) + 60.0,
        layout_name=layout_name,
        fascicles=[
            FascicleSpec(
                fascicle_id=0,
                center_y_um=0.0,
                center_z_um=0.0,
                radius_um=float(fascicle_radius_um),
                label="main_fascicle",
            )
        ],
        fibers=fibers,
        electrodes=electrodes,
        cuffs=cuffs,
        metadata={
            "generator": "make_linear_architecture",
            "family": "linear",
            "fiber_count": int(fiber_count),
            "spacing_um": float(spacing_um),
            "include_cuff": bool(include_cuff),
            "cuff_inner_radius_um": (
                np.nan if cuff_inner_radius_um is None else float(cuff_inner_radius_um)
            ),
            "cuff_thickness_um": float(cuff_thickness_um) if include_cuff else np.nan,
        },
    )
    spec.validate()
    return spec

def generate_linear_sweep_specs(
    fiber_counts: Sequence[int],
    spacings_um: Sequence[float],
    *,
    length_um: float = 20000.0,
    electrode_y_offsets_um: Sequence[float] = (100.0, 250.0),
) -> List[NerveArchitectureSpec]:
    """
    Generate a grid of simple linear architectures.
    """
    specs: List[NerveArchitectureSpec] = []
    for fiber_count in fiber_counts:
        for spacing_um in spacings_um:
            specs.append(
                make_linear_architecture(
                    fiber_count=int(fiber_count),
                    spacing_um=float(spacing_um),
                    length_um=float(length_um),
                    electrode_y_offsets_um=electrode_y_offsets_um,
                )
            )
    return specs

def generate_cuff_sweep_specs(
    *,
    fiber_counts: Sequence[int],
    spacings_um: Sequence[float],
    cuff_inner_radii_um: Sequence[float],
    cuff_thicknesses_um: Sequence[float],
    length_um: float = 20000.0,
) -> List[NerveArchitectureSpec]:
    """
    Generate one-fascicle cuffed architectures over cuff size and spacing.
    """
    specs: List[NerveArchitectureSpec] = []

    for fiber_count in fiber_counts:
        for spacing_um in spacings_um:
            for cuff_inner_radius_um in cuff_inner_radii_um:
                for cuff_thickness_um in cuff_thicknesses_um:
                    bundle_id = (
                        f"cuff_fc{int(fiber_count)}"
                        f"_sp{int(round(spacing_um))}"
                        f"_ir{int(round(cuff_inner_radius_um))}"
                        f"_th{int(round(cuff_thickness_um))}"
                    )
                    spec = make_linear_architecture(
                        fiber_count=int(fiber_count),
                        spacing_um=float(spacing_um),
                        length_um=float(length_um),
                        include_cuff=True,
                        cuff_inner_radius_um=float(cuff_inner_radius_um),
                        cuff_thickness_um=float(cuff_thickness_um),
                        bundle_id=bundle_id,
                        layout_name="cuff_linear",
                    )
                    spec.metadata["generator"] = "generate_cuff_sweep_specs"
                    spec.metadata["family"] = "cuff"
                    specs.append(spec)

    return specs

# -----------------------------------------------------------------------------
# Metric extraction
# -----------------------------------------------------------------------------
def extract_sweep_metrics(
    result,
    summary: Dict[str, Any],
    *,
    params: Optional[SimulationParameters] = None,
    enable_vc: Optional[bool] = None,
    enable_knp: Optional[bool] = None,
    enable_ecs_feedback: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Flatten one ArchitectureSimulationResult into CSV-friendly scalar metrics.
    """
    row: Dict[str, Any] = {}

    spec = result.architecture_spec
    row["bundle_id"] = spec.bundle_id
    row["layout_name"] = spec.layout_name
    row["family"] = spec.metadata.get("family", spec.layout_name)
    row["fiber_count"] = int(spec.fiber_count)
    row["fascicle_count"] = int(spec.fascicle_count)
    row["electrode_count"] = int(spec.electrode_count)
    row["length_um"] = float(spec.length_um)

    row["reporting_fiber_id"] = int(summary.get("reporting_fiber_id", 0))

    # Spec metadata
    row["generator"] = spec.metadata.get("generator", "")
    row["spacing_um"] = _safe_float(spec.metadata.get("spacing_um", np.nan))
    row["fascicle_separation_um"] = _safe_float(
        spec.metadata.get("fascicle_separation_um", np.nan)
    )
    row["intra_fascicle_spacing_um"] = _safe_float(
        spec.metadata.get("intra_fascicle_spacing_um", np.nan)
    )
    row["include_cuff"] = bool(spec.metadata.get("include_cuff", False))
    row["cuff_count"] = int(len(spec.cuffs))
    row["cuff_labels"] = ";".join(
        str(c.label) for c in spec.cuffs if getattr(c, "label", None)
    )
    row["cuff_inner_radius_um"] = _safe_float(
        spec.metadata.get("cuff_inner_radius_um", np.nan)
    )
    row["cuff_thickness_um"] = _safe_float(
        spec.metadata.get("cuff_thickness_um", np.nan)
    )

    row.update(
        _extract_run_condition_metadata(
            result,
            params=params,
            enable_vc=enable_vc,
            enable_knp=enable_knp,
            enable_ecs_feedback=enable_ecs_feedback,
        )
    )

    fiber_diameters = [
        float(f.fiber_diameter_um)
        for f in spec.fibers
        if f.fiber_diameter_um is not None
    ]
    if fiber_diameters:
        row["fiber_diameter_min_um"] = float(np.min(fiber_diameters))
        row["fiber_diameter_max_um"] = float(np.max(fiber_diameters))
        row["fiber_diameter_mean_um"] = float(np.mean(fiber_diameters))
        row["fiber_diameter_std_um"] = float(np.std(fiber_diameters))
    else:
        row["fiber_diameter_min_um"] = np.nan
        row["fiber_diameter_max_um"] = np.nan
        row["fiber_diameter_mean_um"] = np.nan
        row["fiber_diameter_std_um"] = np.nan

    # Peak Vm metrics
    peak_vm_by_fiber = summary.get("peak_vm_by_fiber_mV", {}) or {}
    peak_vm_values = (
        np.asarray([float(v) for v in peak_vm_by_fiber.values()], dtype=float)
        if peak_vm_by_fiber
        else np.asarray([], dtype=float)
    )

    row["max_peak_vm_mV"] = float(np.max(peak_vm_values)) if peak_vm_values.size else np.nan
    row["min_peak_vm_mV"] = float(np.min(peak_vm_values)) if peak_vm_values.size else np.nan
    row["mean_peak_vm_mV"] = float(np.mean(peak_vm_values)) if peak_vm_values.size else np.nan
    row["std_peak_vm_mV"] = float(np.std(peak_vm_values)) if peak_vm_values.size else np.nan

    # VC metrics
    peak_vc = summary.get("peak_vc_by_electrode_mV", {}) or {}
    for electrode_name, value in peak_vc.items():
        safe_name = str(electrode_name).replace(" ", "_")
        row[f"peak_vc_{safe_name}_mV"] = float(value)

    peak_vc_values = (
        np.asarray([float(v) for v in peak_vc.values()], dtype=float)
        if peak_vc
        else np.asarray([], dtype=float)
    )
    row["max_peak_vc_mV"] = float(np.max(peak_vc_values)) if peak_vc_values.size else np.nan
    row["mean_peak_vc_mV"] = float(np.mean(peak_vc_values)) if peak_vc_values.size else np.nan

    # KNP metrics
    row["knp_phi_min_mV"] = _safe_float(summary.get("knp_phi_min_mV", np.nan))
    row["knp_phi_max_mV"] = _safe_float(summary.get("knp_phi_max_mV", np.nan))
    row["knp_max_abs_source_mM_per_ms"] = _safe_float(
        summary.get("knp_max_abs_source_mM_per_ms", np.nan)
    )

    # Optional species-derived metrics
    knp_result = result.knp_result
    if knp_result is not None:
        species_mM = knp_result.get("species_mM", {})
        if isinstance(species_mM, dict) and "K" in species_mM:
            K_hist = np.asarray(species_mM["K"], dtype=float)
            if K_hist.size:
                K0 = K_hist[0]
                delta_K = K_hist - K0[None, :]
                row["max_abs_delta_K_mM"] = float(np.max(np.abs(delta_K)))
                row["mean_abs_delta_K_end_mM"] = float(np.mean(np.abs(delta_K[-1])))
            else:
                row["max_abs_delta_K_mM"] = np.nan
                row["mean_abs_delta_K_end_mM"] = np.nan
        else:
            row["max_abs_delta_K_mM"] = np.nan
            row["mean_abs_delta_K_end_mM"] = np.nan
    else:
        row["max_abs_delta_K_mM"] = np.nan
        row["mean_abs_delta_K_end_mM"] = np.nan

    bundle_metrics = compute_bundle_metrics(
        result,
        threshold_mV=0.0,
    )
    row.update(bundle_metrics)

    return row

# -----------------------------------------------------------------------------
# Plot/export helpers
# -----------------------------------------------------------------------------
def save_result_figures(
    result,
    output_dir: Path,
    potassium_baseline_mM: Optional[float] = None,
) -> None:
    """
    Save the standard figure set for one run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = plot_architecture_result_overview(result)
    for name, (fig, _ax) in figures.items():
        save_figure(fig, output_dir / f"{name}.png", close=True)

    if (
        potassium_baseline_mM is not None
        and result.knp_result is not None
        and isinstance(result.knp_result.get("species_mM", None), dict)
        and "K" in result.knp_result["species_mM"]
    ):
        fig, _ax = plot_knp_species_heatmap(
            result.knp_result,
            species_name="K",
            delta_from_baseline=float(potassium_baseline_mM),
        )
        save_figure(fig, output_dir / "knp_delta_K.png", close=True)

def write_summary_csv(rows: Sequence[Dict[str, Any]], csv_path: Path) -> None:
    """
    Write flattened sweep rows to CSV.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# -----------------------------------------------------------------------------
# Main sweep driver
# -----------------------------------------------------------------------------
def run_architecture_sweep(
    specs: Sequence[NerveArchitectureSpec],
    *,
    output_root: str = "outputs/architecture_sweep",
    params: Optional[SimulationParameters] = None,
    enable_vc: bool = True,
    enable_knp: bool = True,
    enable_ecs_feedback: bool = True,
    save_figures: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run a batch of architecture simulations and save a summary CSV.

    Returns
    -------
    list of dict
        One flattened metrics row per case.
    """
    if params is None:
        params = build_fast_sweep_params()

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for idx, spec in enumerate(specs):
        print(f"[{idx + 1}/{len(specs)}] Running {spec.bundle_id} ...")
        print(
            f"DEBUG pre spec={spec.bundle_id} "
            f"include_cuff_meta={spec.metadata.get('include_cuff')} "
            f"cuff_count={len(spec.cuffs)} "
            f"layout={spec.layout_name}"
        )

        _validate_cuff_consistency(spec)

        result = run_architecture_simulation(
            architecture_spec=spec,
            params=params,
            reporting_fiber_id=None,
            stimulated_fiber_ids=None,
            enable_vc=enable_vc,
            enable_knp=enable_knp,
            enable_ecs_feedback=enable_ecs_feedback,
            knp_species="K",
            bundle_state=None,
        )

        print(
            f"DEBUG post spec={result.architecture_spec.bundle_id} "
            f"include_cuff_meta_postrun={result.architecture_spec.metadata.get('include_cuff')} "
            f"cuff_count_postrun={len(result.architecture_spec.cuffs)} "
            f"layout_postrun={result.architecture_spec.layout_name}"
        )

        _validate_postrun_cuff_consistency(result)

        summary = summarize_architecture_run(result)
        row = extract_sweep_metrics(
            result,
            summary,
            params=params,
            enable_vc=enable_vc,
            enable_knp=enable_knp,
            enable_ecs_feedback=enable_ecs_feedback,
        )
        print(sorted(row.keys()))
        rows.append(row)

        if save_figures:
            case_dir = output_root_path / spec.bundle_id
            save_result_figures(
                result,
                output_dir=case_dir,
                potassium_baseline_mM=float(params.ions.potassium.extracellular_mM),
            )
            # save spatiotemporal field maps
            if spec.fiber_count >= 2:
                debug_field_line_peaks(
                    result=result,
                    params=params,
                    material_model=getattr(result, "material_model", None),
                    y_offsets_um=(0.0, 50.0, 100.0, 200.0),
                )
                save_field_maps(
                    result=result,
                    output_dir=case_dir / "field_maps",
                    params=params,
                    material_model=getattr(result, "material_model", None),
                )        

    csv_path = output_root_path / "sweep_summary.csv"
    write_summary_csv(rows, csv_path)
    print(f"Wrote summary CSV: {csv_path.resolve()}")

    if len(rows) >= 2:
        post_dir = output_root_path / "postprocess"
        export_standard_sweep_plots(csv_path, post_dir)
        print(f"Wrote postprocessed plots: {post_dir.resolve()}")
    else:
        print("Skipping postprocess plots: fewer than 2 rows.")

    return rows

# -----------------------------------------------------------------------------
# Richer architecture generators
# -----------------------------------------------------------------------------
def make_mixed_diameter_linear_architecture(
    *,
    fiber_diameters_um: Sequence[float],
    spacing_um: float,
    length_um: float = 20000.0,
    fascicle_radius_um: Optional[float] = None,
    electrode_positions: Sequence[tuple[float, float, float]] = (
        (5000.0, 100.0, 0.0),
        (10000.0, 250.0, 0.0),
    ),
    bundle_id: Optional[str] = None,
    layout_name: str = "mixed_diameter_linear",
    include_cuff: bool = True,
    cuff_inner_radius_um: Optional[float] = None,
    cuff_thickness_um: float = 100.0,
    cuff_center_y_um: float = 0.0,
    cuff_center_z_um: float = 0.0,
) -> NerveArchitectureSpec:
    """
    One fascicle, linear transverse arrangement, user-specified fiber diameters.

    Optional cuff support is included so mixed-diameter and cuff effects can be
    studied together.
    """
    if len(fiber_diameters_um) == 0:
        raise ValueError("fiber_diameters_um must contain at least one value.")
    if spacing_um <= 0.0:
        raise ValueError("spacing_um must be positive.")

    fiber_count = len(fiber_diameters_um)
    max_d = max(float(d) for d in fiber_diameters_um)

    if fascicle_radius_um is None:
        fascicle_radius_um = max(
            160.0,
            0.5 * (fiber_count - 1) * spacing_um + max_d + 80.0,
        )

    if include_cuff and cuff_inner_radius_um is None:
        cuff_inner_radius_um = float(fascicle_radius_um) + 120.0

    half = 0.5 * (fiber_count - 1)
    fibers: List[FiberSpec] = []
    for i, d_um in enumerate(fiber_diameters_um):
        y_um = (i - half) * spacing_um
        fibers.append(
            FiberSpec(
                fiber_id=i,
                fascicle_id=0,
                center_y_um=float(y_um),
                center_z_um=0.0,
                fiber_diameter_um=float(d_um),
                label=f"fiber_{i}_d{float(d_um):.1f}",
            )
        )

    electrodes: List[ElectrodeSpec] = []
    for idx, (x_um, y_um, z_um) in enumerate(electrode_positions):
        electrodes.append(
            ElectrodeSpec(
                kind="point",
                x_um=float(x_um),
                y_um=float(y_um),
                z_um=float(z_um),
                label=f"E_{idx}",
            )
        )

    cuffs: List[CuffSpec] = []
    if include_cuff:
        cuffs.append(
            CuffSpec(
                cuff_id="cuff_0",
                center_y_um=float(cuff_center_y_um),
                center_z_um=float(cuff_center_z_um),
                inner_radius_um=float(cuff_inner_radius_um),
                thickness_um=float(cuff_thickness_um),
                material_name="cuff_polymer",
                label="ring_cuff",
            )
        )

    spec = NerveArchitectureSpec(
        bundle_id=bundle_id or f"mixeddiam_fc{fiber_count}_sp{int(round(spacing_um))}",
        length_um=float(length_um),
        outer_nerve_radius_um=float(fascicle_radius_um) + 60.0,
        layout_name=layout_name,
        fascicles=[
            FascicleSpec(
                fascicle_id=0,
                center_y_um=0.0,
                center_z_um=0.0,
                radius_um=float(fascicle_radius_um),
                label="main_fascicle",
            )
        ],
        fibers=fibers,
        electrodes=electrodes,
        cuffs=cuffs,
        metadata={
            "generator": "make_mixed_diameter_linear_architecture",
            "family": "mixed_diameter",
            "fiber_count": int(fiber_count),
            "spacing_um": float(spacing_um),
            "diameter_min_um": float(min(fiber_diameters_um)),
            "diameter_max_um": float(max(fiber_diameters_um)),
            "include_cuff": bool(include_cuff),
            "cuff_inner_radius_um": (
                np.nan if cuff_inner_radius_um is None else float(cuff_inner_radius_um)
            ),
            "cuff_thickness_um": float(cuff_thickness_um) if include_cuff else np.nan,
        },
    )
    spec.validate()
    return spec

def make_asymmetric_electrode_architecture(
    *,
    fiber_count: int,
    spacing_um: float,
    length_um: float = 20000.0,
    electrode_positions: Sequence[tuple[float, float, float]] = (
        (5000.0, 40.0, 0.0),
        (10000.0, 220.0, 80.0),
        (15000.0, -180.0, -40.0),
    ),
    fascicle_radius_um: Optional[float] = None,
    bundle_id: Optional[str] = None,
    layout_name: str = "asymmetric_electrode_linear",
) -> NerveArchitectureSpec:
    """
    Symmetric fiber placement with intentionally asymmetric electrode placement.
    Useful for readout selectivity tests.
    """
    base = make_linear_architecture(
        fiber_count=fiber_count,
        spacing_um=spacing_um,
        length_um=length_um,
        fascicle_radius_um=fascicle_radius_um,
        electrode_y_offsets_um=(100.0, 250.0),
        bundle_id=bundle_id or f"asymelec_fc{fiber_count}_sp{int(round(spacing_um))}",
        layout_name=layout_name,
    )

    electrodes: List[ElectrodeSpec] = []
    for idx, (x_um, y_um, z_um) in enumerate(electrode_positions):
        electrodes.append(
            ElectrodeSpec(
                kind="point",
                x_um=float(x_um),
                y_um=float(y_um),
                z_um=float(z_um),
                label=f"E_{idx}",
            )
        )

    base.electrodes = electrodes
    base.metadata["generator"] = "make_asymmetric_electrode_architecture"
    base.metadata["family"] = "asymmetric_electrode"
    base.metadata["electrode_geometry"] = "asymmetric"
    base.validate()
    return base

def make_two_fascicle_architecture(
    *,
    fibers_per_fascicle: int = 5,
    intra_fascicle_spacing_um: float = 50.0,
    fascicle_separation_um: float = 300.0,
    length_um: float = 20000.0,
    fascicle_radius_um: Optional[float] = None,
    electrode_positions: Sequence[tuple[float, float, float]] = (
        (5000.0, 0.0, 0.0),
        (10000.0, 250.0, 0.0),
    ),
    diameter_pattern_um: Optional[Sequence[float]] = None,
    bundle_id: Optional[str] = None,
    layout_name: str = "two_fascicle",
) -> NerveArchitectureSpec:
    """
    Two fascicles separated along y, each with a short linear internal packing.
    """
    if fibers_per_fascicle <= 0:
        raise ValueError("fibers_per_fascicle must be positive.")
    if intra_fascicle_spacing_um <= 0.0:
        raise ValueError("intra_fascicle_spacing_um must be positive.")
    if fascicle_separation_um <= 0.0:
        raise ValueError("fascicle_separation_um must be positive.")

    if diameter_pattern_um is not None and len(diameter_pattern_um) not in {1, fibers_per_fascicle}:
        raise ValueError(
            "diameter_pattern_um must be None, length 1, or length fibers_per_fascicle."
        )

    if fascicle_radius_um is None:
        fascicle_radius_um = max(
            160.0,
            0.5 * (fibers_per_fascicle - 1) * intra_fascicle_spacing_um + 80.0,
        )

    fascicle_centers_y = (-0.5 * fascicle_separation_um, 0.5 * fascicle_separation_um)
    half_local = 0.5 * (fibers_per_fascicle - 1)

    fibers: List[FiberSpec] = []
    next_fiber_id = 0
    for fascicle_id, fascicle_center_y in enumerate(fascicle_centers_y):
        for local_idx in range(fibers_per_fascicle):
            local_offset_y = (local_idx - half_local) * intra_fascicle_spacing_um
            diameter_um = None
            if diameter_pattern_um is not None:
                if len(diameter_pattern_um) == 1:
                    diameter_um = float(diameter_pattern_um[0])
                else:
                    diameter_um = float(diameter_pattern_um[local_idx])

            fibers.append(
                FiberSpec(
                    fiber_id=next_fiber_id,
                    fascicle_id=fascicle_id,
                    center_y_um=float(fascicle_center_y + local_offset_y),
                    center_z_um=0.0,
                    fiber_diameter_um=diameter_um,
                    label=f"f{fascicle_id}_fiber_{local_idx}",
                )
            )
            next_fiber_id += 1

    fascicles = [
        FascicleSpec(
            fascicle_id=0,
            center_y_um=float(fascicle_centers_y[0]),
            center_z_um=0.0,
            radius_um=float(fascicle_radius_um),
            label="fascicle_left",
        ),
        FascicleSpec(
            fascicle_id=1,
            center_y_um=float(fascicle_centers_y[1]),
            center_z_um=0.0,
            radius_um=float(fascicle_radius_um),
            label="fascicle_right",
        ),
    ]

    electrodes: List[ElectrodeSpec] = []
    for idx, (x_um, y_um, z_um) in enumerate(electrode_positions):
        electrodes.append(
            ElectrodeSpec(
                kind="point",
                x_um=float(x_um),
                y_um=float(y_um),
                z_um=float(z_um),
                label=f"E_{idx}",
            )
        )

    spec = NerveArchitectureSpec(
        bundle_id=bundle_id or (
            f"twofasc_fp{fibers_per_fascicle}_sep{int(round(fascicle_separation_um))}"
        ),
        length_um=float(length_um),
        layout_name=layout_name,
        fascicles=fascicles,
        fibers=fibers,
        electrodes=electrodes,
        metadata={
            "generator": "make_two_fascicle_architecture",
            "family": "two_fascicle",
            "fibers_per_fascicle": int(fibers_per_fascicle),
            "fascicle_separation_um": float(fascicle_separation_um),
            "intra_fascicle_spacing_um": float(intra_fascicle_spacing_um),
            "include_cuff": False,
            "cuff_inner_radius_um": np.nan,
            "cuff_thickness_um": np.nan,
        },
    )
    spec.validate()
    return spec

def generate_mixed_diameter_sweep_specs(
    *,
    diameter_sets_um: Sequence[Sequence[float]],
    spacings_um: Sequence[float],
    include_cuff: bool = True,
    cuff_inner_radii_um: Optional[Sequence[float]] = None,
    cuff_thicknesses_um: Optional[Sequence[float]] = None,
    length_um: float = 20000.0,
) -> List[NerveArchitectureSpec]:
    """
    Generate one-fascicle mixed-diameter cases, optionally over cuff geometry.
    """
    specs: List[NerveArchitectureSpec] = []

    if include_cuff:
        cuff_inner_radii_um = cuff_inner_radii_um or [260.0]
        cuff_thicknesses_um = cuff_thicknesses_um or [80.0]
    else:
        cuff_inner_radii_um = [None]
        cuff_thicknesses_um = [np.nan]

    for diameters in diameter_sets_um:
        for spacing_um in spacings_um:
            for cuff_ir in cuff_inner_radii_um:
                for cuff_th in cuff_thicknesses_um:
                    diam_tag = "_".join(str(int(round(float(d)))) for d in diameters)
                    specs.append(
                        make_mixed_diameter_linear_architecture(
                            fiber_diameters_um=diameters,
                            spacing_um=float(spacing_um),
                            length_um=float(length_um),
                            include_cuff=include_cuff,
                            cuff_inner_radius_um=None if cuff_ir is None else float(cuff_ir),
                            cuff_thickness_um=80.0 if np.isnan(cuff_th) else float(cuff_th),
                            bundle_id=(
                                f"mixeddiam_{diam_tag}"
                                f"_sp{int(round(spacing_um))}"
                                f"_cuff{int(include_cuff)}"
                                + (
                                    ""
                                    if cuff_ir is None
                                    else f"_ir{int(round(float(cuff_ir)))}_th{int(round(float(cuff_th)))}"
                                )
                            ),
                        )
                    )
    return specs

def generate_asymmetric_electrode_sweep_specs(
    *,
    fiber_counts: Sequence[int],
    spacings_um: Sequence[float],
    electrode_position_sets: Sequence[Sequence[tuple[float, float, float]]],
    length_um: float = 20000.0,
) -> List[NerveArchitectureSpec]:
    """
    Generate cases with fixed fiber geometry and varied electrode asymmetry.
    """
    specs: List[NerveArchitectureSpec] = []
    for fiber_count in fiber_counts:
        for spacing_um in spacings_um:
            for idx, electrode_positions in enumerate(electrode_position_sets):
                specs.append(
                    make_asymmetric_electrode_architecture(
                        fiber_count=int(fiber_count),
                        spacing_um=float(spacing_um),
                        length_um=float(length_um),
                        electrode_positions=electrode_positions,
                        bundle_id=(
                            f"asymelec_fc{int(fiber_count)}_sp{int(round(spacing_um))}_cfg{idx}"
                        ),
                    )
                )
    return specs

def generate_two_fascicle_sweep_specs(
    *,
    fibers_per_fascicle_values: Sequence[int],
    fascicle_separations_um: Sequence[float],
    intra_fascicle_spacings_um: Sequence[float],
    length_um: float = 20000.0,
) -> List[NerveArchitectureSpec]:
    """
    Generate two-fascicle cases over separation and packing.
    """
    specs: List[NerveArchitectureSpec] = []
    for fibers_per_fascicle in fibers_per_fascicle_values:
        for fascicle_separation_um in fascicle_separations_um:
            for intra_spacing_um in intra_fascicle_spacings_um:
                specs.append(
                    make_two_fascicle_architecture(
                        fibers_per_fascicle=int(fibers_per_fascicle),
                        intra_fascicle_spacing_um=float(intra_spacing_um),
                        fascicle_separation_um=float(fascicle_separation_um),
                        length_um=float(length_um),
                        bundle_id=(
                            f"twofasc_fp{int(fibers_per_fascicle)}"
                            f"_sep{int(round(fascicle_separation_um))}"
                            f"_sp{int(round(intra_spacing_um))}"
                        ),
                    )
                )
    return specs

# -----------------------------------------------------------------------------
# Example execution
# -----------------------------------------------------------------------------
"""
def main() -> None:
    specs = [
        make_linear_architecture(
            fiber_count=5,
            spacing_um=60.0,
            length_um=20000.0,
            include_cuff=True,
            cuff_inner_radius_um=260.0,
            cuff_thickness_um=80.0,
            bundle_id="cuff_fc5_sp60_ir260_th80",
            layout_name="cuff_linear_debug",
        ),
        make_mixed_diameter_linear_architecture(
            fiber_diameters_um=[6.0, 8.0, 10.0, 12.0, 14.0],
            spacing_um=60.0,
            length_um=20000.0,
            bundle_id="mixeddiam_cuff_test",
            layout_name="mixed_diameter_cuff_test",
            include_cuff=True,
            cuff_inner_radius_um=260.0,
            cuff_thickness_um=80.0,
        ),
    ]

    run_architecture_sweep(
        specs,
        output_root="outputs/architecture_sweep_full_debug",
        params=build_fast_sweep_params(),
        enable_vc=True,
        enable_knp=True,
        enable_ecs_feedback=True,
        save_figures=False,
    )
"""
def main() -> None:
    """
    Physiologically oriented architecture debug set:

    1. Single-fiber reference:
       - isolated myelinated axon
       - no cuff
       - useful as a regression anchor

    2. Five-fiber compact bundle:
       - moderate fascicle radius
       - moderate cuff radius/thickness
       - useful for small-bundle architecture comparisons

    3. Twelve-fiber denser bundle:
       - tighter packing
       - slightly larger cuff
       - useful for stronger VC/CNAP and ECS loading trends
    """

    specs = [
        # --------------------------------------------------------------
        # 1) Single-fiber physiological reference
        # --------------------------------------------------------------
        make_linear_architecture(
            fiber_count=1,
            spacing_um=120.0,              # irrelevant for 1 fiber but kept explicit
            length_um=20000.0,             # ~20 mm comparable branch scale
            fascicle_radius_um=160.0,
            fiber_diameter_um=10.0,        # medium myelinated reference fiber
            include_cuff=False,
            bundle_id="physio_single_fiber_fc1",
            layout_name="physio_single_fiber",
            electrode_x_um=(5000.0, 10000.0),
            electrode_y_offsets_um=(100.0, 250.0),
        ),

        # --------------------------------------------------------------
        # 2) Five-fiber physiological compact bundle
        # --------------------------------------------------------------
        make_mixed_diameter_linear_architecture(
            fiber_diameters_um=[8.0, 9.0, 10.0, 11.0, 12.0],
            spacing_um=55.0,
            length_um=20000.0,
            fascicle_radius_um=220.0,
            include_cuff=True,
            cuff_inner_radius_um=300.0,
            cuff_thickness_um=90.0,
            bundle_id="physio_bundle_fc5",
            layout_name="physio_bundle_5fiber",
            electrode_positions=(
                (5000.0, 90.0, 0.0),
                (10000.0, 220.0, 0.0),
            ),
        ),

        # --------------------------------------------------------------
        # 3) Twelve-fiber physiological denser bundle
        # --------------------------------------------------------------
        make_mixed_diameter_linear_architecture(
            fiber_diameters_um=[
                6.0, 7.0, 7.5, 8.0, 8.5, 9.0,
                9.5, 10.0, 10.5, 11.0, 12.0, 13.0,
            ],
            spacing_um=40.0,
            length_um=20000.0,
            fascicle_radius_um=360.0,
            include_cuff=True,
            cuff_inner_radius_um=470.0,
            cuff_thickness_um=110.0,
            bundle_id="physio_bundle_fc12",
            layout_name="physio_bundle_12fiber",
            electrode_positions=(
                (5000.0, 100.0, 0.0),
                (10000.0, 260.0, 0.0),
            ),
        ),
    ]

    rows = run_architecture_sweep(
        specs,
        output_root="outputs/architecture_sweep_physiology",
        params=build_fast_sweep_params(),
        enable_vc=True,
        enable_knp=True,
        enable_ecs_feedback=True,
        save_figures=True,
    )

    print("\nCompleted physiological architecture sweep.")
    print(f"Rows written: {len(rows)}")
if __name__ == "__main__":
    main()