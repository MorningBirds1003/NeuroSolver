from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import numpy as np

from Scripts.NeuroSolver.baseline_io import save_json
from Scripts.NeuroSolver.params import (
    DEFAULT_PARAMS,
    SimulationParameters,
    build_medium_myelinated_physiology_params,
)
from Scripts.NeuroSolver.regression_cases import (
    build_reference_anchor_case,
    summarize_reference_run,
)
from Scripts.NeuroSolver.scheduler import run_multirate_simulation
from Scripts.NeuroSolver.architecture.architecture_presets import (
    make_three_fiber_demo_architecture,
)
from Scripts.NeuroSolver.architecture.architecture_io import save_architecture_spec
from Scripts.NeuroSolver.architecture.architecture_runner import (
    run_architecture_simulation,
    summarize_architecture_run,
    build_bundle_phi_e_xt_map,
)
from Scripts.NeuroSolver.architecture.plotting import (
    plot_architecture_result_overview,
    plot_knp_species_heatmap,
    plot_vc_trace_windows,
    save_figure,
)
from Scripts.NeuroSolver.architecture.metrics import compute_bundle_metrics
from Scripts.NeuroSolver.architecture.field_mapping import FieldSampleGridSpec
from Scripts.NeuroSolver.architecture.field_postprocess import (
    plot_phi_e_xt_heatmap,
    plot_delta_phi_xt_heatmap,
)

OUTPUT_ROOT = Path("outputs") / "poc_hh_knp"

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def build_poc_fast_params() -> SimulationParameters:
    """
    Reduced-cost but still meaningful POC runtime.

    Goal:
    - keep the fast cable stable,
    - keep KNP active,
    - finish quickly enough for iterative testing.
    """
    params = build_medium_myelinated_physiology_params(DEFAULT_PARAMS)

    solver = replace(
        params.solver,
        t_stop_ms=6.0,
        dt_fast_ms=0.01,
        dt_slow_ms=0.10,
    )

    topology = replace(
        params.topology,
        node_count=11,
    )

    return replace(
        params,
        solver=solver,
        topology=topology,
    )

def build_poc_physio_train_params(
    *,
    source_scale_physiologic_mM_per_ms_per_uA: float = 1.0e-5,
    clearance_tau_K_ms: float = 500.0,
    volume_fraction: float = 0.20,
    tortuosity: float = 1.6,
) -> SimulationParameters:
    """
    Physiologic short-train calibration case.

    Keep feedback gain = 1.0 and adjust only source/loading and clearance.
    """
    params = build_poc_fast_params()

    ecs = replace(
        params.ecs,
        volume_fraction=float(volume_fraction),
        tortuosity=float(tortuosity),
        diffusivity_scale_factor=1.0 / (float(tortuosity) ** 2),
    )

    stimulus = replace(
        params.stimulus,
        pulse_count=5,
        pulse_interval_ms=3.0,
        pulse_width_ms=0.8,
        pulse_amplitude_uA=0.003,
        pulse_amplitude_uA_per_cm2=1000.0,
    )

    solver = replace(
        params.solver,
        t_stop_ms=20.0,
        dt_fast_ms=0.01,
        dt_slow_ms=0.10,
    )

    knp = replace(
        params.knp,
        source_scale_physiologic_mM_per_ms_per_uA=float(source_scale_physiologic_mM_per_ms_per_uA),
        clearance_tau_K_ms=float(clearance_tau_K_ms),
        feedback_phi_gain=1.0,
    )

    return replace(
        params,
        ecs=ecs,
        stimulus=stimulus,
        solver=solver,
        knp=knp,
    )

def build_poc_harsh_ecs_params() -> SimulationParameters:
    """
    Make ECS loading visibly stronger without rewriting the solver.

    This is intentionally a regime-scaling test, not a realism claim.
    """
    params = build_poc_fast_params()

    ecs = replace(
        params.ecs,
        volume_fraction=0.10,   # tighter ECS
        tortuosity=2.0,         # more restricted diffusion
        diffusivity_scale_factor=1.0 / (2.0 ** 2),
    )

    stimulus = replace(
        params.stimulus,
        pulse_count=3,
        pulse_interval_ms=2.0,
        pulse_width_ms=0.8,
        pulse_amplitude_uA=0.003,
        pulse_amplitude_uA_per_cm2=1000.0,
    )

    solver = replace(
        params.solver,
        t_stop_ms=12.0,
        dt_fast_ms=0.01,
        dt_slow_ms=0.10,
    )

    return replace(
        params,
        ecs=ecs,
        stimulus=stimulus,
        solver=solver,
    )

def _with_feedback_gain(
    params: SimulationParameters,
    *,
    phi_gain: float = 1.0,
    phi_clamp_abs_mV: float = 5.0,
    reversal_clamp_abs_shift_mV: float = 20.0,
) -> SimulationParameters:
    """
    Return params with adjusted ECS-feedback gain/clamp settings.

    This is a controlled sensitivity tool for validation, not a realism claim.
    """
    knp = replace(
        params.knp,
        feedback_phi_gain=float(phi_gain),
        feedback_phi_clamp_abs_mV=float(phi_clamp_abs_mV),
        feedback_reversal_clamp_abs_shift_mV=float(reversal_clamp_abs_shift_mV),
    )
    return replace(params, knp = knp)

def build_poc_harsh_train_params() -> SimulationParameters:
    """
    Stronger repeated-activity validation regime.
    """
    params = build_poc_fast_params()

    ecs = replace(
        params.ecs,
        volume_fraction=0.08,
        tortuosity=2.4,
        diffusivity_scale_factor=1.0 / (2.4 ** 2),
    )

    stimulus = replace(
        params.stimulus,
        pulse_count=8,
        pulse_interval_ms=1.5,
        pulse_width_ms=0.8,
        pulse_amplitude_uA=0.0035,
        pulse_amplitude_uA_per_cm2=1200.0,
    )

    solver = replace(
        params.solver,
        t_stop_ms=18.0,
        dt_fast_ms=0.01,
        dt_slow_ms=0.10,
    )

    return replace(
        params,
        ecs=ecs,
        stimulus=stimulus,
        solver=solver,
    )

def build_poc_harsh_train_high_gain_params(phi_gain: float = 5.0) -> SimulationParameters:
    """
    Same harsh train regime, but with amplified ECS->cable phi feedback for sensitivity testing.
    """
    params = build_poc_harsh_train_params()
    return _with_feedback_gain(
        params,
        phi_gain=float(phi_gain),
        phi_clamp_abs_mV=20.0,
        reversal_clamp_abs_shift_mV=40.0,
    )

def _save_architecture_figures(
    arch_result,
    params: SimulationParameters,
    outdir: Path,
    save_bundle_field_maps: bool = False,
    field_line_y_um: float = 100.0,
    field_line_z_um: float = 0.0,
    field_n_x: int = 80,
) -> None:
    _ensure_dir(outdir)

    figures = plot_architecture_result_overview(arch_result)
    for name, (fig, _ax) in figures.items():
        save_figure(fig, outdir / f"{name}.png", close=True)

    if arch_result.vc_result is not None:

        vc_windows = plot_vc_trace_windows(
            arch_result.vc_result,
            pulse_start_ms=float(params.stimulus.pulse_start_ms),
            pulse_count=int(params.stimulus.pulse_count),
            pulse_interval_ms=float(params.stimulus.pulse_interval_ms),
            window_pre_ms=0.5,
            window_post_ms=2.0,
        )
        for name, (fig, _ax) in vc_windows.items():
            save_figure(fig, outdir / f"{name}.png", close=True)
    
    if arch_result.knp_result is not None:
        species_mM = arch_result.knp_result.get("species_mM", {})
        if isinstance(species_mM, dict) and "K" in species_mM:
            fig, _ax = plot_knp_species_heatmap(
                arch_result.knp_result,
                species_name="K",
                delta_from_baseline=float(params.ions.potassium.extracellular_mM),
            )
            save_figure(fig, outdir / "knp_delta_K_heatmap.png", close=True)

    # Optional bundle-level VC field map
    if save_bundle_field_maps and len(arch_result.per_fiber_cable_results) >= 2:
        field_result = build_bundle_phi_e_xt_map(
            result=arch_result,
            params=params,
            line_spec=FieldSampleGridSpec(
                name=f"bundle_line_y{int(round(field_line_y_um))}",
                x_min_um=0.0,
                x_max_um=float(arch_result.architecture_spec.length_um),
                n_x=int(field_n_x),
                y_um=float(field_line_y_um),
                z_um=float(field_line_z_um),
            ),
            material_model=getattr(arch_result, "material_model", None),
        )

        np.savez(
            outdir / "vc_bundle_phi_e_xt_map.npz",
            t_ms=field_result["t_ms"],
            x_um=field_result["x_um"],
            phi_e_mV=field_result["phi_e_mV"],
            y_um=field_result["y_um"],
            z_um=field_result["z_um"],
        )

        plot_phi_e_xt_heatmap(
            field_result,
            output_path=outdir / "vc_bundle_phi_e_xt_heatmap.png",
            title=f"{arch_result.architecture_spec.bundle_id}: VC bundle phi_e(x,t)",
        )
        plot_delta_phi_xt_heatmap(
            field_result,
            output_path=outdir / "vc_bundle_delta_phi_e_xt_heatmap.png",
            title=f"{arch_result.architecture_spec.bundle_id}: VC bundle delta phi_e(x,t)",
        )

def run_three_fiber_case(
    output_root: Path,
    case_name: str,
    params: SimulationParameters,
    enable_ecs_feedback: bool,
    save_bundle_field_maps: bool = False,
) -> Dict[str, Any]:
    outdir = _ensure_dir(output_root / case_name)

    spec = make_three_fiber_demo_architecture(length_um=20000.0)
    save_architecture_spec(spec, outdir / "architecture.json")

    arch_result = run_architecture_simulation(
        architecture_spec=spec,
        params=params,
        reporting_fiber_id=0,
        stimulated_fiber_ids=None,
        enable_vc=True,
        enable_knp=True,
        enable_ecs_feedback=enable_ecs_feedback,
        knp_species="K",
        bundle_state=None,
    )

    summary = summarize_architecture_run(arch_result)
    bundle_metrics = compute_bundle_metrics(arch_result, threshold_mV=-20.0)

    payload = {
        "case_name": case_name,
        "summary": summary,
        "bundle_metrics": bundle_metrics,
        "metadata": dict(getattr(arch_result, "metadata", {}) or {}),
    }

    _save_architecture_figures(
        arch_result,
        params=params,
        outdir=outdir,
        save_bundle_field_maps=save_bundle_field_maps,
    )
    save_json(outdir / "summary.json", payload)
    return payload

def run_single_fiber_anchor_case(output_root: Path) -> Dict[str, Any]:
    outdir = _ensure_dir(output_root / "single_fiber_anchor")

    params = build_poc_fast_params()
    case = build_reference_anchor_case(params=params)

    result = run_multirate_simulation(
        geometry=case.geometry,
        params=case.params,
        electrode_points=case.electrode_points,
        stimulated_index=None,
        enable_vc=True,
        enable_knp=True,
        enable_ecs_feedback=False,
        knp_species="K",
        cable_y_um=float(case.geometry.get("fiber_center_y_um", 0.0)),
        cable_z_um=float(case.geometry.get("fiber_center_z_um", 0.0)),
    )

    summary = summarize_reference_run(
        result=result,
        geometry=case.geometry,
        params=case.params,
    )

    save_json(outdir / "summary.json", summary)
    return summary
"""
#def run_three_fiber_weak_regime_case(output_root: Path) -> Dict[str, Any]:
    outdir = _ensure_dir(output_root / "three_fiber_weak_regime")

    params = build_poc_fast_params()
    spec = make_three_fiber_demo_architecture(length_um=20000.0)
    save_architecture_spec(spec, outdir / "architecture.json")

    arch_result = run_architecture_simulation(
        architecture_spec=spec,
        params=params,
        reporting_fiber_id=0,
        stimulated_fiber_ids=None,
        enable_vc=True,
        enable_knp=True,
        enable_ecs_feedback=True,
        knp_species="K",
        bundle_state=None,
    )

    summary = summarize_architecture_run(arch_result)
    bundle_metrics = compute_bundle_metrics(arch_result, threshold_mV=-20.0)

    payload = {
        "case_name": "three_fiber_weak_regime",
        "summary": summary,
        "bundle_metrics": bundle_metrics,
        "metadata": dict(getattr(arch_result, "metadata", {}) or {}),
    }

    _save_architecture_figures(arch_result, params=params, outdir=outdir)
    save_json(outdir / "summary.json", payload)
    return payload

#def run_three_fiber_harsh_ecs_case(output_root: Path) -> Dict[str, Any]:
    outdir = _ensure_dir(output_root / "three_fiber_harsh_ecs")

    params = build_poc_harsh_ecs_params()
    spec = make_three_fiber_demo_architecture(length_um=20000.0)
    save_architecture_spec(spec, outdir / "architecture.json")

    arch_result = run_architecture_simulation(
        architecture_spec=spec,
        params=params,
        reporting_fiber_id=0,
        stimulated_fiber_ids=None,
        enable_vc=True,
        enable_knp=True,
        enable_ecs_feedback=True,
        knp_species="K",
        bundle_state=None,
    )

    summary = summarize_architecture_run(arch_result)
    bundle_metrics = compute_bundle_metrics(arch_result, threshold_mV=-20.0)
    payload = {
        "case_name": "three_fiber_harsh_ecs",
        "summary": summary,
        "bundle_metrics": bundle_metrics,
        "metadata": dict(getattr(arch_result, "metadata", {}) or {}),
    }

    _save_architecture_figures(arch_result, params=params, outdir=outdir)
    save_json(outdir / "summary.json", payload)
    return payload
"""
def run_three_fiber_weak_regime_case(output_root: Path) -> Dict[str, Any]:
    return run_three_fiber_case(
        output_root=output_root,
        case_name="three_fiber_weak_regime",
        params=build_poc_fast_params(),
        enable_ecs_feedback=True,
        save_bundle_field_maps=False,
    )

def run_three_fiber_harsh_ecs_case(output_root: Path) -> Dict[str, Any]:
    return run_three_fiber_case(
        output_root=output_root,
        case_name="three_fiber_harsh_ecs",
        params=build_poc_harsh_ecs_params(),
        enable_ecs_feedback=True,
        save_bundle_field_maps=False,
    )

def run_three_fiber_weak_feedback_off_case(output_root: Path) -> Dict[str, Any]:
    return run_three_fiber_case(
        output_root=output_root,
        case_name="three_fiber_weak_feedback_off",
        params=build_poc_fast_params(),
        enable_ecs_feedback=False,
        save_bundle_field_maps=False,
    )

def run_three_fiber_weak_feedback_on_case(output_root: Path) -> Dict[str, Any]:
    return run_three_fiber_case(
        output_root=output_root,
        case_name="three_fiber_weak_feedback_on",
        params=build_poc_fast_params(),
        enable_ecs_feedback=True,
        save_bundle_field_maps=False,
    )

def run_three_fiber_harsh_feedback_off_case(output_root: Path) -> Dict[str, Any]:
    return run_three_fiber_case(
        output_root=output_root,
        case_name="three_fiber_harsh_feedback_off",
        params=build_poc_harsh_ecs_params(),
        enable_ecs_feedback=False,
        save_bundle_field_maps=False,
    )

def run_three_fiber_harsh_feedback_on_case(output_root: Path) -> Dict[str, Any]:
    return run_three_fiber_case(
        output_root=output_root,
        case_name="three_fiber_harsh_feedback_on",
        params=build_poc_harsh_ecs_params(),
        enable_ecs_feedback=True,
        save_bundle_field_maps=False,
    )
"""
def build_poc_harsh_train_params() -> SimulationParameters:
    params = build_poc_fast_params()

    ecs = replace(
        params.ecs,
        volume_fraction=0.10,
        tortuosity=2.0,
        diffusivity_scale_factor=1.0 / (2.0 ** 2),
    )

    stimulus = replace(
        params.stimulus,
        pulse_count=5,
        pulse_interval_ms=2.0,
        pulse_width_ms=0.8,
        pulse_amplitude_uA=0.003,
        pulse_amplitude_uA_per_cm2=1000.0,
    )

    solver = replace(
        params.solver,
        t_stop_ms=14.0,
        dt_fast_ms=0.01,
        dt_slow_ms=0.10,
    )

    return replace(
        params,
        ecs=ecs,
        stimulus=stimulus,
        solver=solver,
    )
"""
def run_three_fiber_harsh_train_feedback_off_case(output_root: Path) -> Dict[str, Any]:
    return run_three_fiber_case(
        output_root=output_root,
        case_name="three_fiber_harsh_train_feedback_off",
        params=build_poc_harsh_train_params(),
        enable_ecs_feedback=False,
        save_bundle_field_maps=False,
    )

def run_three_fiber_harsh_train_feedback_on_case(output_root: Path) -> Dict[str, Any]:
    return run_three_fiber_case(
        output_root=output_root,
        case_name="three_fiber_harsh_train_feedback_on",
        params=build_poc_harsh_train_params(),
        enable_ecs_feedback=True,
        save_bundle_field_maps=False,
    )

def run_three_fiber_harsh_train_gain5_feedback_on_case(output_root: Path) -> Dict[str, Any]:
    return run_three_fiber_case(
        output_root=output_root,
        case_name="three_fiber_harsh_train_gain5_feedback_on",
        params=build_poc_harsh_train_high_gain_params(phi_gain=5.0),
        enable_ecs_feedback=True,
        save_bundle_field_maps=False,
    )

def run_three_fiber_harsh_train_gain10_feedback_on_case(output_root: Path) -> Dict[str, Any]:
    return run_three_fiber_case(
        output_root=output_root,
        case_name="three_fiber_harsh_train_gain10_feedback_on",
        params=build_poc_harsh_train_high_gain_params(phi_gain=10.0),
        enable_ecs_feedback=True,
        save_bundle_field_maps=False,
    )

def run_three_fiber_physio_train_calibration_case(
    output_root: Path,
    case_name: str,
    *,
    source_scale_physiologic_mM_per_ms_per_uA: float,
    clearance_tau_K_ms: float,
    volume_fraction: float = 0.20,
    tortuosity: float = 1.6,
) -> Dict[str, Any]:
    params = build_poc_physio_train_params(
        source_scale_physiologic_mM_per_ms_per_uA=source_scale_physiologic_mM_per_ms_per_uA,
        clearance_tau_K_ms=clearance_tau_K_ms,
        volume_fraction=volume_fraction,
        tortuosity=tortuosity,
    )
    return run_three_fiber_case(
        output_root=output_root,
        case_name=case_name,
        params=params,
        enable_ecs_feedback=True,
        save_bundle_field_maps=False,
    )

def run_all_poc_cases(output_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    output_root = _ensure_dir(output_root)
    all_results: Dict[str, Any] = {}

    print("[POC] running single_fiber_anchor")
    all_results["single_fiber_anchor"] = run_single_fiber_anchor_case(output_root)

    print("[POC] running three_fiber_weak_feedback_off")
    all_results["three_fiber_weak_feedback_off"] = run_three_fiber_weak_feedback_off_case(output_root)

    print("[POC] running three_fiber_weak_feedback_on")
    all_results["three_fiber_weak_feedback_on"] = run_three_fiber_weak_feedback_on_case(output_root)

    print("[POC] running three_fiber_harsh_feedback_off")
    all_results["three_fiber_harsh_feedback_off"] = run_three_fiber_harsh_feedback_off_case(output_root)

    print("[POC] running three_fiber_harsh_feedback_on")
    all_results["three_fiber_harsh_feedback_on"] = run_three_fiber_harsh_feedback_on_case(output_root)

    print("[POC] running three_fiber_harsh_train_feedback_off")
    all_results["three_fiber_harsh_train_feedback_off"] = run_three_fiber_harsh_train_feedback_off_case(output_root)

    print("[POC] running three_fiber_harsh_train_feedback_on")
    all_results["three_fiber_harsh_train_feedback_on"] = run_three_fiber_harsh_train_feedback_on_case(output_root)
    print("[POC] running three_fiber_harsh_train_gain5_feedback_on")
    all_results["three_fiber_harsh_train_gain5_feedback_on"] = run_three_fiber_harsh_train_gain5_feedback_on_case(output_root)

    print("[POC] running three_fiber_harsh_train_gain10_feedback_on")
    all_results["three_fiber_harsh_train_gain10_feedback_on"] = run_three_fiber_harsh_train_gain10_feedback_on_case(output_root)    
    print("[POC] running three_fiber_physio_train_calib_low_source")
    all_results["three_fiber_physio_train_calib_low_source"] = run_three_fiber_physio_train_calibration_case(
        output_root=output_root,
        case_name="three_fiber_physio_train_calib_low_source",
        source_scale_physiologic_mM_per_ms_per_uA=5.0e-6,
        clearance_tau_K_ms=500.0,
    )

    print("[POC] running three_fiber_physio_train_calib_baseline")
    all_results["three_fiber_physio_train_calib_baseline"] = run_three_fiber_physio_train_calibration_case(
        output_root=output_root,
        case_name="three_fiber_physio_train_calib_baseline",
        source_scale_physiologic_mM_per_ms_per_uA=1.0e-5,
        clearance_tau_K_ms=500.0,
    )

    print("[POC] running three_fiber_physio_train_calib_high_source")
    all_results["three_fiber_physio_train_calib_high_source"] = run_three_fiber_physio_train_calibration_case(
        output_root=output_root,
        case_name="three_fiber_physio_train_calib_high_source",
        source_scale_physiologic_mM_per_ms_per_uA=2.0e-5,
        clearance_tau_K_ms=500.0,
    )    
    save_json(output_root / "poc_manifest.json", all_results)
    return all_results
"""
if __name__ == "__main__":
    results = run_all_poc_cases()
    print("Saved POC outputs to:", OUTPUT_ROOT.resolve())
    print("Cases:", list(results.keys()))
"""
def run_physio_calibration_cases(output_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    output_root = _ensure_dir(output_root)
    all_results: Dict[str, Any] = {}

    print("[POC] running single_fiber_anchor")
    all_results["single_fiber_anchor"] = run_single_fiber_anchor_case(output_root)

    print("[POC] running three_fiber_physio_train_calib_low_source")
    all_results["three_fiber_physio_train_calib_low_source"] = run_three_fiber_physio_train_calibration_case(
        output_root=output_root,
        case_name="three_fiber_physio_train_calib_low_source",
        source_scale_physiologic_mM_per_ms_per_uA=5.0e-6,
        clearance_tau_K_ms=500.0,
    )

    print("[POC] running three_fiber_physio_train_calib_baseline")
    all_results["three_fiber_physio_train_calib_baseline"] = run_three_fiber_physio_train_calibration_case(
        output_root=output_root,
        case_name="three_fiber_physio_train_calib_baseline",
        source_scale_physiologic_mM_per_ms_per_uA=1.0e-5,
        clearance_tau_K_ms=500.0,
    )

    print("[POC] running three_fiber_physio_train_calib_high_source")
    all_results["three_fiber_physio_train_calib_high_source"] = run_three_fiber_physio_train_calibration_case(
        output_root=output_root,
        case_name="three_fiber_physio_train_calib_high_source",
        source_scale_physiologic_mM_per_ms_per_uA=2.0e-5,
        clearance_tau_K_ms=500.0,
    )

    save_json(output_root / "poc_manifest_physio.json", all_results)
    return all_results

if __name__ == "__main__":
    results = run_physio_calibration_cases()
    print("Saved physiologic calibration outputs to:", OUTPUT_ROOT.resolve())
    print("Cases:", list(results.keys()))