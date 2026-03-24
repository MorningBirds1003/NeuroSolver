"""
run_architecture_case.py

Standalone runner for a JSON-defined architecture.

This version is intentionally conservative for first-pass architecture tests:
- it uses a reduced-cost runtime preset by default
- it can disable VC/KNP/feedback for a fast smoke test
- once the architecture path is validated, you can turn the full layers back on
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Tuple, Dict, Any

from Scripts.NeuroSolver.params import (
    DEFAULT_PARAMS,
    build_medium_myelinated_physiology_params,
    SimulationParameters,
)
from Scripts.NeuroSolver.architecture.architecture_io import load_architecture_spec
from Scripts.NeuroSolver.architecture.architecture_runner import (
    run_architecture_simulation,
    summarize_architecture_run,
)
from Scripts.NeuroSolver.architecture.plotting import (
    plot_architecture_result_overview,
    plot_knp_species_heatmap,
    save_figure,
)
from Scripts.NeuroSolver.architecture.sweep_runner import save_field_maps

def build_fast_architecture_test_params() -> SimulationParameters:
    """
    Reduced-cost runtime parameters for architecture smoke testing.

    Why:
    ----
    The architecture path is already reaching the real bundle scheduler.
    If the run stalls in gating/cable stepping, the first thing to do is
    shorten and coarsen the test so the architecture/IO/plot path can be
    validated before running the full physiological case.
    """
    params = build_medium_myelinated_physiology_params(DEFAULT_PARAMS)

    # Keep this conservative and easy to finish on a laptop.
    solver = replace(
        params.solver,
        t_stop_ms=2.0,      # shorter run
        dt_fast_ms=0.01,    # coarser fast step
        dt_slow_ms=0.10,    # coarser slow step
    )

    topology = replace(
        params.topology,
        node_count=11,      # fewer nodes than the full validation harness
    )

    return replace(
        params,
        solver=solver,
        topology=topology,
    )

def run_architecture_case(
    architecture_json_path: str,
    output_dir: str = "outputs/architecture_case",
    use_fast_test_params: bool = False,
    enable_vc: bool = False,
    enable_knp: bool = False,
    enable_ecs_feedback: bool = False,
    save_field_maps_enabled: bool = True,
    field_map_only_if_multifiber: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a JSON architecture, run the simulation, and save standard plots.

    Recommended progression
    -----------------------
    1) Start with:
         enable_vc=False
         enable_knp=False
         enable_ecs_feedback=False
    2) Then re-run with VC on
    3) Then KNP on
    4) Then ECS feedback on

    Notes
    -----
    This case runner now exports the same bundle-level phi_e(x,t) field maps
    used by the sweep path, so one-off architecture runs and sweep runs produce
    comparable artifacts.
    """
    if use_fast_test_params:
        params = build_fast_architecture_test_params()
    else:
        params = build_medium_myelinated_physiology_params(DEFAULT_PARAMS)

    spec = load_architecture_spec(architecture_json_path)

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

    summary = summarize_architecture_run(result)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    figures = plot_architecture_result_overview(result)
    for name, (fig, _ax) in figures.items():
        save_figure(fig, outdir / f"{name}.png", close=True)

    if result.knp_result is not None:
        species_mM = result.knp_result.get("species_mM", {})
        if isinstance(species_mM, dict) and "K" in species_mM:
            fig, _ax = plot_knp_species_heatmap(
                result.knp_result,
                species_name="K",
                delta_from_baseline=float(params.ions.potassium.extracellular_mM),
            )
            save_figure(fig, outdir / "knp_delta_K.png", close=True)

    should_save_field_maps = bool(save_field_maps_enabled)
    if field_map_only_if_multifiber:
        should_save_field_maps = should_save_field_maps and (int(spec.fiber_count) >= 2)

    if should_save_field_maps:
        save_field_maps(
            result=result,
            output_dir=outdir / "field_maps",
            params=params,
            material_model=getattr(result, "material_model", None),
        )

    return result, summary

if __name__ == "__main__":
    architecture_json = "outputs/architectures/demo_architecture.json"

    result, summary = run_architecture_case(
        architecture_json_path=architecture_json,
        output_dir="outputs/architecture_case",
        use_fast_test_params=True,
        enable_vc=True,
        enable_knp=True,
        enable_ecs_feedback=True,
        save_field_maps_enabled=True,
        field_map_only_if_multifiber=True,
    )

    print(summary)