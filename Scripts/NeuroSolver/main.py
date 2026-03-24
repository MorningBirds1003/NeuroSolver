"""
main.py

Integrated test harness for the NeuroSolver stack.

Purpose
-------
This file is the main executable entrypoint for quick validation of the current
NeuroSolver implementation. It is not just a "run script"; it acts as a
structured systems-level smoke test for the major subsystems:

1. parameter construction / preset selection,
2. stimulus generation,
3. single-node excitability,
4. cable propagation,
5. bundle-level scheduling,
6. extracellular volume conduction (VC),
7. slow KNP extracellular concentration evolution,
8. ECS feedback into membrane/reversal behavior,
9. architecture-level bundle specification and plotting.

Design intent
-------------
- Does the current parameter preset look physiologically sensible?
- Do the stimulus primitives behave as expected?
- Can a node spike?
- Can propagation occur across a fiber?
- Can a bundle be built and scheduled?
- Do the VC and KNP pathways execute without breaking?
- Are bundle and architecture metadata visible enough for debugging?

Important implementation note
-----------------------------
Phase 1 / Phase 1.5 / Phase 2 bundle support in this file does NOT necessarily
mean that the entire multiscale system is "fully physiologically complete."
Instead, this file progressively activates more of the code stack so you can
validate current coupling layers from a single entrypoint.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

# =============================================================================
# Core parameter imports
# =============================================================================
# DEFAULT_PARAMS:
#   Baseline full-structure parameter object. This is the canonical parameter
#   tree that downstream modules expect.
#
# SimulationParameters:
#   Main dataclass/type for passing a complete simulation parameter bundle.
#
# build_medium_myelinated_physiology_params:
#   Convenience builder that applies the intended physiological preset for a
#   medium myelinated mammalian fiber. This is important because we want the
#   default executable path to represent a physiological baseline, not a debug
#   or visibility-test configuration.
#
# summarize_reversal_potentials_mV:
#   Utility to compute/report ion reversal potentials from the current ion pools.
#   Used here purely for visibility and sanity checking.
#
# cylinder_lateral_area_cm2:
#   Used to convert an absolute intracellular stimulus current [uA] into an
#   equivalent current density [uA/cm^2] for single-node testing.
from Scripts.NeuroSolver.params import (
    DEFAULT_PARAMS,
    SimulationParameters,
    build_medium_myelinated_physiology_params,
    summarize_reversal_potentials_mV,
    cylinder_lateral_area_cm2,
)

# =============================================================================
# Stimulus imports
# =============================================================================
# These are the stimulus primitives used for smoke tests.
#
# BiphasicPulseSpec / PulseTrainSpec:
#   Small spec containers defining biphasic pulses and repeated rectangular pulses.
#
# intracellular_node_stimulus_uA:
#   Absolute current stimulus function used by the cable / intracellular injection path.
#
# biphasic_pulse / pulse_train:
#   Generic waveform utilities used to verify timing and amplitude conventions.
#
# stimulus_vector_single_index:
#   Helper for vectorized stimulation, useful when only one fiber/node/etc. is active.
from Scripts.NeuroSolver.stimuli import (
    BiphasicPulseSpec,
    PulseTrainSpec,
    intracellular_node_stimulus_uA,
    biphasic_pulse,
    pulse_train,
    stimulus_vector_single_index,
)

# =============================================================================
# Fast membrane/node solver imports
# =============================================================================
# initialize_node_state:
#   Creates the initial state of a single node at rest.
#
# run_single_node_pulse_test:
#   Executes a single-node excitability test so we can verify that the chosen
#   membrane + stimulus combination is reasonable before moving to full cable propagation.
from Scripts.NeuroSolver.propagation.node_model import (
    initialize_node_state,
    run_single_node_pulse_test,
)

# =============================================================================
# Geometry builders
# =============================================================================
# build_node_internode_geometry:
#   Builds the 1D compartmental geometry for a single myelinated fiber.
#
# BundleGeometry / build_bundle_geometry:
#   Bundle-level geometric container and constructor. Used when testing multi-fiber
#   scaling and shared ECS/VC interaction setup.
from Scripts.NeuroSolver.propagation.myelin_geometry import build_node_internode_geometry
from Scripts.NeuroSolver.propagation.bundle_geometry import (
    BundleGeometry,
    build_bundle_geometry,
)

# =============================================================================
# Bundle runtime-state imports
# =============================================================================
# BundleRuntimeState:
#   Container for the runtime state of an entire bundle, including per-fiber fast
#   states and shared ECS/VC/material placeholders.
#
# initialize_bundle_state:
#   Constructs the runtime bundle state from bundle geometry and parameters.
from Scripts.NeuroSolver.bundle_state import (
    BundleRuntimeState,
    initialize_bundle_state,
)

# =============================================================================
# Diagnostics / reporting imports
# =============================================================================
# format_node_report:
#   Human-readable report for a cable/fiber result at one threshold.
#
# format_multi_threshold_report:
#   Same idea, but evaluates multiple thresholds. Useful because propagation
#   latency and apparent conduction depend somewhat on what threshold is used.
from Scripts.NeuroSolver.propagation.node_diagnostics import (
    format_node_report,
    format_multi_threshold_report,
)

# =============================================================================
# Schedulers
# =============================================================================
# run_multirate_simulation:
#   Single-fiber multirate path (fast cable dynamics + optional slower ECS coupling).
#
# run_multirate_bundle_simulation:
#   Bundle-aware version that advances multiple fibers and aggregates their ECS/VC impact.
from Scripts.NeuroSolver.scheduler import (
    run_multirate_simulation,
    run_multirate_bundle_simulation,
)

# =============================================================================
# VC solver utility type
# =============================================================================
# ElectrodeSamplePoint:
#   Defines electrode locations for extracellular potential sampling.
from Scripts.NeuroSolver.ECS.vc_solver import ElectrodeSamplePoint

# =============================================================================
# Architecture-level specification imports
# =============================================================================
# These provide a more formal, explicit way of defining a nerve bundle.
#
# ElectrodeSpec / FiberSpec / FascicleSpec / NerveArchitectureSpec:
#   Declarative objects for defining geometry and sensing configuration.
#
# run_architecture_simulation:
#   Higher-level execution path that takes the architecture spec and routes it
#   through bundle construction and execution.
#
# summarize_architecture_run:
#   Utility for condensed reporting from an architecture result.
from Scripts.NeuroSolver.architecture.architecture_schema import (
    ElectrodeSpec,
    FiberSpec,
    FascicleSpec,
    NerveArchitectureSpec,
)
from Scripts.NeuroSolver.architecture.architecture_runner import (
    run_architecture_simulation,
    summarize_architecture_run,
)

# =============================================================================
# Plotting imports
# =============================================================================
# save_figure:
#   Save a matplotlib figure and optionally close it.
#
# plot_architecture_result_overview:
#   Makes general overview figures from an architecture run.
#
# plot_knp_species_heatmap:
#   Generates a space-time concentration heatmap for a selected ionic species.
from Scripts.NeuroSolver.architecture.plotting import (
    save_figure,
    plot_architecture_result_overview,
    plot_knp_species_heatmap,
)

# =============================================================================
# Runtime mode selection
# =============================================================================
# These switches determine which execution path is taken.
#
# USE_ARCHITECTURE_MODE:
#   If True, the script routes through the higher-level architecture stack.
#   This is the most explicit and future-facing path because it uses a formal
#   bundle/fascicle/fiber/electrode specification.
#
#   Why keep this?
#   Because architecture mode is the best representation of the intended final
#   workflow for customized nerve-geometry studies.
USE_ARCHITECTURE_MODE = True

# USE_BUNDLE_MODE:
#   Legacy/fallback direct bundle mode. Only used when architecture mode is False.
#   This allows testing bundle mechanics without going through the formal
#   architecture-spec layer.
USE_BUNDLE_MODE = True

# BUNDLE_FIBER_COUNT:
#   Number of fibers to build in fallback bundle mode.
#
#   Why 3?
#   Three fibers is a reasonable minimal nontrivial bundle:
#   - one central/reference fiber,
#   - two neighboring fibers,
#   - enough to test superposition / aggregation behavior,
#   - cheap enough to run quickly during debugging.
BUNDLE_FIBER_COUNT = 3

# BUNDLE_LAYOUT_NAME:
#   Geometric arrangement used in fallback bundle mode.
#
#   Why "hex"?
#   Hex-like or hex-packed arrangements are common simplifications for local
#   packing because they are symmetric and spatially reasonable without needing
#   a full histology-derived placement.
BUNDLE_LAYOUT_NAME = "hex"

# BUNDLE_TEST_FIBER_ID:
#   The fiber selected for reporting in bundle mode.
#
#   Why report a single fiber?
#   Existing diagnostics are mostly single-fiber oriented, so one representative
#   fiber is extracted for readable reporting while the bundle still runs as a whole.
BUNDLE_TEST_FIBER_ID = 0

# =============================================================================
# Plot/export controls
# =============================================================================
# SAVE_ARCHITECTURE_PLOTS:
#   If True, save overview figures and KNP heatmaps for architecture-mode runs.
#
#   Why default to True?
#   Because the architecture path is meant to be inspectable and presentable, not
#   just executable.
SAVE_ARCHITECTURE_PLOTS = True

# ARCHITECTURE_PLOT_DIR:
#   Output directory for saved architecture figures.
ARCHITECTURE_PLOT_DIR = Path("outputs") / "architecture_demo"


def print_param_summary(params: SimulationParameters = DEFAULT_PARAMS) -> None:
    """
    Print a human-readable summary of the active simulation parameters.

    Why this exists
    ---------------
    In a multiscale simulator, hidden parameter drift is one of the easiest ways
    to invalidate results. This summary prints the most load-bearing quantities:
    temperature, resting potential, membrane capacitance, axial resistivity,
    stimulus scale, whether KNP/VC/feedback are active, and reversal-potential policy.

    This is primarily a sanity-check utility.
    """
    print("=" * 72)
    print("NeuroSolver parameter summary")
    print("=" * 72)
    print(f"Preset = {getattr(params.preset, 'active_preset_name', 'unspecified')}")
    print(f"T = {params.temperature.celsius:.2f} C")
    print(f"V_rest = {params.membrane.resting_potential_mV:.2f} mV")
    print(f"Cm = {params.membrane.membrane_capacitance_uF_per_cm2:.4f} uF/cm^2")
    print(f"rho_i = {params.membrane.axial_resistivity_ohm_cm:.2f} ohm*cm")
    print(f"node_count default = {params.topology.node_count}")

    # pulse_amplitude_uA:
    #   Absolute intracellular current used by the cable/injection path.
    # pulse_amplitude_uA_per_cm2:
    #   Area-normalized current density used in the single-node path.
    print(f"stim abs current = {getattr(params.stimulus, 'pulse_amplitude_uA', 0.0):.6f} uA")
    print(f"stim density    = {params.stimulus.pulse_amplitude_uA_per_cm2:.3f} uA/cm^2")

    # KNP enabled:
    #   Whether slow extracellular ionic concentration evolution is active.
    # VC enabled:
    #   Whether extracellular potentials from current sources are sampled.
    # ECS feedback:
    #   Whether slow extracellular state is fed back into fast membrane dynamics.
    print(f"KNP enabled default     = {bool(getattr(params.knp, 'enabled', False))}")
    print(f"VC enabled default      = {bool(getattr(params.vc, 'enabled', False))}")
    print(f"ECS feedback default    = {bool(getattr(params.policy, 'use_dynamic_ecs_feedback', False))}")
    print(f"KNP mode                = {getattr(params.knp, 'coupling_mode', 'physiologic')}")
    print(f"Nernst reversal policy  = {bool(getattr(params.policy, 'use_nernst_reversal_from_ion_pools', False))}")

    print("-" * 72)
    print("Nernst reversals from ion pools:")
    for key, value in summarize_reversal_potentials_mV(params).items():
        print(f"  {key}: {value:.3f} mV")
    print("=" * 72)


def build_validation_run_params(
    params: SimulationParameters = DEFAULT_PARAMS,
) -> SimulationParameters:
    """
    Build a validation-specific parameter set in which the single-node test and
    cable test use approximately equivalent stimulus strength.

    Why this is needed
    ------------------
    The codebase uses two stimulus conventions in different places:

    1. Cable simulation:
       absolute current [uA]

    2. Single-node test:
       current density [uA/cm^2]

    If these are left unrelated, then the single-node and cable tests may probe
    very different excitation regimes even when they are supposed to represent
    the same pulse. This helper converts the absolute nodal pulse into an
    equivalent nodal current density using nodal lateral area.

    Important numerical note
    ------------------------
    eq_density is clipped to 1000 uA/cm^2 to avoid pathological values if the
    node area becomes very small. This is a safety cap rather than a deep
    physiological principle.
    """
    # Compute nodal membrane area so we can translate absolute current into
    # current density for the single-node formulation.
    node_area_cm2 = cylinder_lateral_area_cm2(
        length_um=float(params.geometry.node_length_um),
        diameter_um=float(params.geometry.axon_diameter_um),
    )

    # Absolute pulse amplitude used by cable/intracellular stimulation path.
    abs_amp_uA = float(getattr(params.stimulus, "pulse_amplitude_uA", 0.0))

    # Convert to equivalent current density.
    eq_density = abs_amp_uA / max(node_area_cm2, 1.0e-15)

    # Numerical safeguard: prevent extreme values from accidental tiny-area divisions.
    eq_density = min(eq_density, 1000.0)

    # Create a modified stimulus object preserving all other stimulus settings.
    stimulus = replace(
        params.stimulus,
        pulse_amplitude_uA=abs_amp_uA,
        pulse_amplitude_uA_per_cm2=eq_density,
    )

    # Return a new SimulationParameters object with only the stimulus replaced.
    return replace(params, stimulus=stimulus)


def test_stimuli(params: SimulationParameters = DEFAULT_PARAMS) -> None:
    """
    Smoke-test the stimulus generators.

    Why this exists
    ---------------
    Before debugging membrane or propagation behavior, it is useful to confirm
    that the input waveform machinery itself behaves correctly in time. This
    function checks:
    - rectangular absolute-current pulse,
    - biphasic pulse,
    - pulse train,
    - vectorized single-index stimulation.

    The chosen sample times intentionally straddle pre-pulse, pulse-on,
    inter-pulse, and late-time intervals.
    """
    print("=" * 72)
    print("Stimuli smoke test")
    print("-" * 72)

    # Times chosen to sample before, during, and after representative pulses.
    times_ms = [0.5, 1.0, 1.5, 2.5, 6.0]

    print("Rectangular abs-current pulse samples [uA]:")
    for t in times_ms:
        val = intracellular_node_stimulus_uA(
            t_ms=t,
            start_ms=params.stimulus.pulse_start_ms,
            width_ms=params.stimulus.pulse_width_ms,
            amplitude_uA=getattr(params.stimulus, "pulse_amplitude_uA", 0.0),
        )
        print(f"  t={t:4.1f} ms -> {val:.6f}")

    # Biphasic pulse:
    # - phase1 positive,
    # - small interphase gap,
    # - phase2 negative.
    #
    # Why this waveform?
    # Biphasic pulses are common in stimulation because they can reduce net
    # injected charge and are relevant for neuromodulation/electrode contexts.
    biphasic = BiphasicPulseSpec(
        start_ms=1.0,
        phase1_width_ms=0.5,
        phase1_amplitude=1.0,
        interphase_gap_ms=0.1,
        phase2_width_ms=0.5,
        phase2_amplitude=-1.0,
    )

    print("Biphasic pulse samples [arb units]:")
    for t in times_ms:
        print(f"  t={t:4.1f} ms -> {biphasic_pulse(t, biphasic):.6f}")

    # Pulse train:
    # - three identical pulses,
    # - fixed interval,
    # - arbitrary amplitude.
    #
    # Why this matters:
    # Repetitive stimulation is often more relevant than single pulses for
    # physiology and can reveal accumulation or refractory behavior later.
    train = PulseTrainSpec(
        start_ms=1.0,
        width_ms=0.5,
        amplitude=1.0,
        pulse_count=3,
        interval_ms=2.0,
    )

    print("Pulse-train samples [arb units]:")
    for t in times_ms:
        print(f"  t={t:4.1f} ms -> {pulse_train(t, train):.6f}")

    # Example of a sparse stimulation vector where only one entry is active.
    vec = stimulus_vector_single_index(n_entries=8, active_index=2, amplitude=3.5)
    print(f"Single-index vector test: {vec}")
    print("=" * 72)


def test_single_node(params: SimulationParameters = DEFAULT_PARAMS) -> None:
    """
    Run a single-node excitability test and print the peak response.

    Why this exists
    ---------------
    This is the fastest meaningful physiological test in the stack. If a single
    node cannot respond sensibly, then debugging full cable/bundle/ECS behavior
    becomes premature.

    Reported outputs
    ----------------
    - initial membrane voltage,
    - peak membrane voltage,
    - time of peak.
    """
    print("Single-node pulse test")
    print("-" * 72)

    # Initial node state at rest.
    node0 = initialize_node_state(params)

    # Time trace from the single-node pulse test.
    trace = run_single_node_pulse_test(params=params)

    # Extract membrane voltage and time arrays for simple reporting.
    v = np.asarray([sample["V_m_mV"] for sample in trace], dtype=float)
    t = np.asarray([sample["t_ms"] for sample in trace], dtype=float)

    # Index of the maximal membrane voltage.
    peak_idx = int(np.argmax(v))

    print(
        f"Initial Vm: -{abs(node0.V_m_mV):.3f} mV"
        if node0.V_m_mV < 0
        else f"Initial Vm: {node0.V_m_mV:.3f} mV"
    )
    print(f"Peak Vm:    {v[peak_idx]:.3f} mV")
    print(f"Peak time:  {t[peak_idx]:.3f} ms")
    print("=" * 72)


def _print_geometry_summary(geometry) -> None:
    """
    Print a concise summary of a single-fiber geometry dictionary.

    Why this exists
    ---------------
    Geometry errors are common and can silently alter propagation behavior.
    This function gives quick visibility into:
    - number of compartments,
    - how many are nodes vs internodes,
    - fiber identity/placement,
    - resolved lengths,
    - first several compartment locations.
    """
    region = np.asarray(geometry["region_type"], dtype=object)
    x_um = np.asarray(geometry["x_um"], dtype=float)

    n_total = int(geometry["n_compartments"])
    n_nodes = int(np.sum(region == "node"))
    n_internodes = int(np.sum(region == "internode"))

    print(f"Compartment count: {n_total}")
    print(f"Node compartments: {n_nodes}")
    print(f"Internode compartments: {n_internodes}")

    # These metadata fields are not guaranteed to exist in every geometry object,
    # so they are checked conditionally.
    if "fiber_id" in geometry:
        print(f"Fiber ID: {int(geometry['fiber_id'])}")
    if "fascicle_id" in geometry and geometry["fascicle_id"] is not None:
        print(f"Fascicle ID: {int(geometry['fascicle_id'])}")
    if "fiber_center_y_um" in geometry:
        print(f"Fiber center y: {float(geometry['fiber_center_y_um']):.3f} um")
    if "fiber_center_z_um" in geometry:
        print(f"Fiber center z: {float(geometry['fiber_center_z_um']):.3f} um")

    # "Resolved" values reflect whatever the geometry builder actually produced
    # after applying internal discretization logic.
    if "node_count_resolved" in geometry:
        print(f"Resolved node count: {int(geometry['node_count_resolved'])}")
    if "internode_length_resolved_um" in geometry:
        print(f"Resolved internode length: {float(geometry['internode_length_resolved_um']):.3f} um")
    if "total_length_resolved_um" in geometry:
        print(f"Resolved total length: {float(geometry['total_length_resolved_um']) / 1000.0:.3f} mm")

    print("First 12 compartments:")
    for i in range(min(12, n_total)):
        print(f"  idx={i:3d} | region={str(region[i]):>9s} | x_center={x_um[i]:10.3f} um")


def _print_bundle_summary(bundle: BundleGeometry) -> None:
    """
    Print a concise bundle-geometry summary.

    Why this exists
    ---------------
    Once multiple fibers are present, geometric placement matters for ECS and VC.
    This summary helps verify:
    - number of fibers,
    - total compartments,
    - bundle radius,
    - spatial extents,
    - actual per-fiber placement coordinates.
    """
    print("=" * 72)
    print("Bundle geometry summary")
    print("-" * 72)
    print(f"Layout:                 {bundle.layout_name}")
    print(f"Fiber count:            {bundle.total_fiber_count}")
    print(f"Total compartments:     {bundle.total_compartment_count}")
    print(f"Bundle radius:          {bundle.bundle_radius_um:.3f} um")
    print(f"y extent:               [{bundle.y_extent_um[0]:.3f}, {bundle.y_extent_um[1]:.3f}] um")
    print(f"z extent:               [{bundle.z_extent_um[0]:.3f}, {bundle.z_extent_um[1]:.3f}] um")
    print("Fiber placements:")
    for p in bundle.placements:
        print(
            f"  fiber_id={p.fiber_id:3d} | "
            f"fascicle={p.fascicle_id:3d} | "
            f"y={p.center_y_um:9.3f} um | "
            f"z={p.center_z_um:9.3f} um | "
            f"label={p.label}"
        )
    print("=" * 72)


def _print_bundle_state_summary(bundle: BundleGeometry, bundle_state: BundleRuntimeState) -> None:
    """
    Print the initialized runtime-state contents for a bundle.

    Why this exists
    ---------------
    Bundle geometry alone is not enough; we also need to confirm that runtime
    state has been initialized consistently. This includes:
    - per-fiber fast membrane state,
    - shared ECS placeholders,
    - shared VC placeholders,
    - shared material state,
    - diagnostics storage.

    This is especially useful before moving into more coupled bundle phases.
    """
    print("=" * 72)
    print("Bundle runtime-state summary")
    print("-" * 72)
    print(
        f"Bundle ID:              "
        f"{bundle_state.metadata.get('bundle_id', bundle.metadata.get('bundle_id', 'bundle_0'))}"
    )
    print(f"Fiber states tracked:   {len(bundle_state.fiber_states)}")
    print(f"Shared ECS keys:        {sorted(bundle_state.shared_ecs_state.keys())}")
    print(f"Shared VC keys:         {sorted(bundle_state.shared_vc_state.keys())}")
    print(f"Shared material keys:   {sorted(bundle_state.shared_material_state.keys())}")
    print(f"Diagnostics keys:       {sorted(bundle_state.diagnostics_state.keys())}")

    # Per-fiber summary: number of compartments, mean initial Vm, spatial center.
    for fiber_id in bundle.fiber_ids:
        g = bundle.get_fiber_geometry(fiber_id)
        s = bundle_state.fiber_states[int(fiber_id)]
        v = np.asarray(s["V_m_mV"], dtype=float)
        print(
            f"  fiber_id={int(fiber_id):3d} | "
            f"n_comp={int(g['n_compartments']):5d} | "
            f"Vm0_mean={float(np.mean(v)):.3f} mV | "
            f"center=({float(g.get('fiber_center_y_um', 0.0)):.3f}, "
            f"{float(g.get('fiber_center_z_um', 0.0)):.3f}) um"
        )
    print("=" * 72)


def _build_demo_architecture_spec(
    params: SimulationParameters,
) -> NerveArchitectureSpec:
    """
    Build a minimal explicit architecture specification for demonstration runs.

    Why this exists
    ---------------
    This provides a formal architecture-layer input without needing an external
    file or UI. It is intentionally small but meaningful:
    - one fascicle,
    - three fibers,
    - two electrodes at different distances.

    Why these numbers?
    ------------------
    - One fascicle keeps geometry simple.
    - Three fibers is the smallest nontrivial fiber-interaction test.
    - One "near" and one "far" electrode lets you verify distance-dependent VC signals.
    """
    return NerveArchitectureSpec(
        bundle_id="architecture_validation",

        # Use the parameter-set length if available; otherwise fall back to 20 mm.
        # Why 20,000 um?
        # This is long enough to observe propagation behavior and sample electrodes
        # at intermediate distances without being excessively expensive.
        length_um=float(
            getattr(params.geometry, "total_length_um", 20000.0)
            if hasattr(params, "geometry")
            else 20000.0
        ),

        layout_name="custom_demo",

        # One centrally located fascicle.
        fascicles=[
            FascicleSpec(
                fascicle_id=0,
                center_y_um=0.0,
                center_z_um=0.0,
                radius_um=160.0,
                label="main_fascicle",
            )
        ],

        # Three fibers: center, right, left.
        # Why +/-80 um offsets?
        # Large enough to create spatial separation, small enough to remain within
        # the fascicle radius and represent local neighboring fibers.
        fibers=[
            FiberSpec(
                fiber_id=0,
                fascicle_id=0,
                center_y_um=0.0,
                center_z_um=0.0,
                label="center_fiber",
            ),
            FiberSpec(
                fiber_id=1,
                fascicle_id=0,
                center_y_um=80.0,
                center_z_um=0.0,
                label="right_fiber",
            ),
            FiberSpec(
                fiber_id=2,
                fascicle_id=0,
                center_y_um=-80.0,
                center_z_um=0.0,
                label="left_fiber",
            ),
        ],

        # Two simple point electrodes.
        # Why x=5000 and 10000 um?
        # These sample the propagating waveform at mid-domain positions.
        #
        # Why y=100 and 250 um?
        # These provide one near-field and one farther-field extracellular sample.
        electrodes=[
            ElectrodeSpec(
                kind="point",
                x_um=5000.0,
                y_um=100.0,
                z_um=0.0,
                label="E_near",
            ),
            ElectrodeSpec(
                kind="point",
                x_um=10000.0,
                y_um=250.0,
                z_um=0.0,
                label="E_far",
            ),
        ],

        metadata={"source": "main_demo_architecture"},
    )


def _build_active_geometry_and_bundle_state(
    params: SimulationParameters,
) -> Tuple[dict, Optional[BundleGeometry], Optional[BundleRuntimeState]]:
    """
    Resolve the active geometry mode for the current run.

    Returns
    -------
    geometry : dict
        Single reporting geometry. In bundle mode this is the chosen reporting fiber.
    bundle : BundleGeometry or None
        Full bundle geometry if bundle mode is active.
    bundle_state : BundleRuntimeState or None
        Bundle runtime-state if bundle mode is active.

    Why this exists
    ---------------
    This isolates the "what geometry am I actually using?" logic so the main
    integration test remains cleaner.
    """
    if USE_BUNDLE_MODE:
        bundle = build_bundle_geometry(
            params=params,
            n_fibers=BUNDLE_FIBER_COUNT,
            layout_name=BUNDLE_LAYOUT_NAME,
            bundle_id="bundle_validation",
        )
        bundle_state = initialize_bundle_state(bundle_geometry=bundle, params=params)

        # Defensive check: the requested reporting fiber must exist.
        if int(BUNDLE_TEST_FIBER_ID) not in bundle.fibers:
            raise ValueError(
                f"BUNDLE_TEST_FIBER_ID={BUNDLE_TEST_FIBER_ID} not present in built bundle."
            )

        # Return the geometry for the selected reporting fiber.
        geometry = bundle.get_fiber_geometry(BUNDLE_TEST_FIBER_ID)
        return geometry, bundle, bundle_state

    # Single-fiber fallback geometry.
    geometry = build_node_internode_geometry(params=params)
    return geometry, None, None


def _print_probe_summary(cable_result, geometry) -> None:
    """
    Print summary statistics for a handful of representative compartments.

    Why this exists
    ---------------
    Full voltage matrices are too large to inspect manually. This function picks
    several positions along the cable and reports:
    - peak voltage,
    - minimum voltage,
    - first threshold crossing time at 0 mV.

    The selected probe locations are:
    - start,
    - quarter,
    - middle,
    - three-quarter,
    - end.

    This gives a quick sense of whether propagation is occurring across the domain.
    """
    V_hist = np.asarray(cable_result["V_m_mV"], dtype=float)
    t_ms = np.asarray(cable_result["t_ms"], dtype=float)
    x_um = np.asarray(geometry["x_um"], dtype=float)

    n_compartments = V_hist.shape[1]
    probe_indices = [
        0,
        n_compartments // 4,
        n_compartments // 2,
        (3 * n_compartments) // 4,
        n_compartments - 1,
    ]

    print("Probe compartments:")
    for idx in probe_indices:
        trace = V_hist[:, idx]
        peak = float(np.max(trace))
        vmin = float(np.min(trace))
        crossing = np.where(trace >= 0.0)[0]

        if crossing.size > 0:
            first_cross = float(t_ms[int(crossing[0])])
            print(
                f"  idx={idx:3d} | x={x_um[idx]:10.3f} um | "
                f"peak={peak:9.3f} mV | min={vmin:9.3f} mV | first_cross={first_cross:.6f} ms"
            )
        else:
            print(
                f"  idx={idx:3d} | x={x_um[idx]:10.3f} um | "
                f"peak={peak:9.3f} mV | min={vmin:9.3f} mV | first_cross=nan"
            )


def _extract_reporting_fiber_result(bundle_result, fiber_id: int):
    """
    Extract the cable result for one reporting fiber from a bundle result.

    Why this exists
    ---------------
    Much of the existing reporting infrastructure assumes a single-fiber result
    dictionary. Rather than rewriting all downstream diagnostics, we extract one
    representative fiber and pass that along.

    Fallback behavior
    -----------------
    If the requested fiber_id is missing, the first available fiber result is used.
    """
    fiber_results = getattr(bundle_result, "per_fiber_cable_results", None)
    if fiber_results is None:
        fiber_results = bundle_result.fiber_results

    if int(fiber_id) not in fiber_results:
        first_id = sorted(fiber_results.keys())[0]
        return fiber_results[first_id], int(first_id)

    return fiber_results[int(fiber_id)], int(fiber_id)


def _save_architecture_figures(arch_result, params: SimulationParameters) -> None:
    """
    Save overview and KNP diagnostic figures from an architecture run.

    Why this exists
    ---------------
    Architecture mode is meant to be visually inspectable. Saved figures are useful
    for debugging, reporting, and checking whether space-time structure looks plausible.

    K heatmap logic
    ---------------
    If KNP ran and potassium is present, a delta-K heatmap is also saved relative
    to the extracellular potassium baseline.
    """
    if not SAVE_ARCHITECTURE_PLOTS:
        return

    ARCHITECTURE_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    figures = plot_architecture_result_overview(arch_result)

    for name, (fig, _ax) in figures.items():
        save_figure(fig, ARCHITECTURE_PLOT_DIR / f"{name}.png", close=True)

    if arch_result.knp_result is not None and "species_mM" in arch_result.knp_result:
        species_mM = arch_result.knp_result["species_mM"]
        if isinstance(species_mM, dict) and "K" in species_mM:
            fig, _ax = plot_knp_species_heatmap(
                arch_result.knp_result,
                species_name="K",

                # Baseline extracellular potassium used for delta-from-baseline visualization.
                delta_from_baseline=float(params.ions.potassium.extracellular_mM),
            )
            save_figure(fig, ARCHITECTURE_PLOT_DIR / "knp_delta_K.png", close=True)

    print(f"Saved architecture figures to: {ARCHITECTURE_PLOT_DIR.resolve()}")


def test_integrated_multirate(params: SimulationParameters = DEFAULT_PARAMS) -> None:
    """
    Run the integrated multirate system test.

    This is the main systems-level test in the file.

    Execution modes
    ---------------
    1. Architecture mode:
       Uses the formal architecture specification and routes through the full
       architecture runner.

    2. Bundle mode:
       Builds a bundle directly and runs the bundle-aware multirate scheduler.

    3. Single-fiber mode:
       Runs the multirate scheduler on one fiber.

    Outputs inspected
    -----------------
    - cable result,
    - VC result,
    - KNP result,
    - bundle summaries,
    - geometry summaries,
    - probe summaries,
    - threshold-based diagnostics,
    - VC test values,
    - slow ECS feedback summaries.
    """
    print("Integrated multirate test")
    print("-" * 72)

    # bundle / bundle_state remain None in pure single-fiber mode.
    bundle = None
    bundle_state = None

    # Default reporting fiber for bundle/architecture runs.
    reporting_fiber_id = BUNDLE_TEST_FIBER_ID

    # -------------------------------------------------------------------------
    # Path 1: formal architecture mode
    # -------------------------------------------------------------------------
    if USE_ARCHITECTURE_MODE:
        architecture_spec = _build_demo_architecture_spec(params)

        # Run the architecture-layer simulation.
        arch_result = run_architecture_simulation(
            architecture_spec=architecture_spec,
            params=params,
            reporting_fiber_id=reporting_fiber_id,

            # stimulated_fiber_ids=None lets the runner use its own default behavior.
            stimulated_fiber_ids=None,

            # enable_vc / enable_knp / enable_ecs_feedback = None means:
            # "use the parameter defaults rather than forcing a new state here."
            enable_vc=None,
            enable_knp=None,
            enable_ecs_feedback=None,

            # K is the typical first species to inspect because extracellular K+
            # is often the most immediately interpretable slow variable in neural ECS models.
            knp_species="K",

            # If None, the architecture runner will initialize or manage bundle_state internally.
            bundle_state=None,
        )

        bundle = arch_result.bundle_geometry
        bundle_state = arch_result.bundle_state
        bundle_result = arch_result.bundle_result
        geometry = arch_result.reporting_geometry
        reporting_fiber_id = arch_result.reporting_fiber_id

        # Extract single reporting fiber for existing diagnostics.
        cable_result, reporting_fiber_id = _extract_reporting_fiber_result(
            bundle_result,
            reporting_fiber_id,
        )
        vc_result = bundle_result.vc_result
        knp_result = bundle_result.knp_result
        metadata = bundle_result.metadata if bundle_result.metadata is not None else {}

        print("Architecture mode summary")
        print("-" * 72)
        arch_summary = summarize_architecture_run(arch_result)
        print(f"Bundle ID:        {arch_summary.get('bundle_id')}")
        print(f"Fiber count:      {arch_summary.get('fiber_count')}")
        print(f"Reporting fiber:  {arch_summary.get('reporting_fiber_id')}")
        print(f"Electrode count:  {arch_summary.get('electrode_count')}")
        print("=" * 72)

        _save_architecture_figures(arch_result, params)

    # -------------------------------------------------------------------------
    # Path 2/3: direct bundle mode or single-fiber mode
    # -------------------------------------------------------------------------
    else:
        geometry, bundle, bundle_state = _build_active_geometry_and_bundle_state(params=params)

        if bundle is not None:
            _print_bundle_summary(bundle)
        if bundle is not None and bundle_state is not None:
            _print_bundle_state_summary(bundle, bundle_state)

        # Default electrode pair for non-architecture execution.
        electrodes = [
            ElectrodeSamplePoint(name="E_near", x_um=5000.0, y_um=100.0, z_um=0.0),
            ElectrodeSamplePoint(name="E_far", x_um=10000.0, y_um=250.0, z_um=0.0),
        ]

        # -----------------------------
        # Path 2: direct bundle run
        # -----------------------------
        if bundle is not None and bundle_state is not None:
            bundle_result = run_multirate_bundle_simulation(
                bundle_geometry=bundle,
                bundle_state=bundle_state,
                params=params,
                electrode_points=electrodes,

                # Leave as None so runtime uses the parameter default policies.
                enable_vc=None,
                enable_knp=None,
                enable_ecs_feedback=None,

                # K chosen as the representative KNP species for reporting.
                knp_species="K",
            )

            reporting_result, reporting_fiber_id = _extract_reporting_fiber_result(
                bundle_result,
                BUNDLE_TEST_FIBER_ID,
            )

            cable_result = reporting_result
            vc_result = bundle_result.vc_result
            knp_result = bundle_result.knp_result
            metadata = bundle_result.metadata if bundle_result.metadata is not None else {}
            geometry = bundle.get_fiber_geometry(reporting_fiber_id)

            print(f"Bundle scheduler reporting fiber: {reporting_fiber_id}")
            print(f"Bundle scheduler fiber count run: {metadata.get('fiber_count', len(bundle.fiber_ids))}")

        # -----------------------------
        # Path 3: single-fiber run
        # -----------------------------
        else:
            result = run_multirate_simulation(
                geometry=geometry,
                params=params,
                electrode_points=electrodes,

                # None means let downstream code determine default stimulation target.
                stimulated_index=None,

                # Leave subsystem switches at parameter-defined defaults.
                enable_vc=None,
                enable_knp=None,
                enable_ecs_feedback=None,

                knp_species="K",

                # Fiber center coordinates are passed explicitly so ECS/VC sampling
                # can be spatially aware even in single-fiber mode.
                cable_y_um=float(geometry.get("fiber_center_y_um", 0.0)),
                cable_z_um=float(geometry.get("fiber_center_z_um", 0.0)),
            )
            cable_result = result.cable_result
            vc_result = result.vc_result
            knp_result = result.knp_result
            metadata = result.metadata if result.metadata is not None else {}

    # Reprint bundle summaries if available after the run.
    if bundle is not None:
        _print_bundle_summary(bundle)
    if bundle is not None and bundle_state is not None:
        _print_bundle_state_summary(bundle, bundle_state)

    # -------------------------------------------------------------------------
    # Cable-result summary
    # -------------------------------------------------------------------------
    V_hist = np.asarray(cable_result["V_m_mV"], dtype=float)
    global_v_peak = float(np.max(V_hist))
    global_v_min = float(np.min(V_hist))

    print(
        f"Runtime preset:           "
        f"{metadata.get('preset_name', getattr(params.preset, 'active_preset_name', 'unspecified'))}"
    )
    print(f"Runtime VC enabled:       {bool(metadata.get('enable_vc', False))}")
    print(f"Runtime KNP enabled:      {bool(metadata.get('enable_knp', False))}")
    print(f"Runtime ECS feedback:     {bool(metadata.get('feedback_enabled', False))}")
    print(f"Feedback policy default:  {bool(getattr(params.policy, 'use_dynamic_ecs_feedback', False))}")
    print(f"Cable supports feedback:  {bool(metadata.get('feedback_cable_support', False))}")

    _print_geometry_summary(geometry)
    print(f"Time samples:      {cable_result['t_ms'].shape[0]}")
    print(f"Voltage shape:     {V_hist.shape}")
    print(f"Global V_peak:     {global_v_peak:.3f} mV")
    print(f"Global V_min:      {global_v_min:.3f} mV")

    # Effective membrane voltage may differ from raw Vm when extracellular
    # feedback/coupling is incorporated into the membrane driving quantity.
    if "V_membrane_effective_mV" in cable_result:
        V_eff = np.asarray(cable_result["V_membrane_effective_mV"], dtype=float)
        print(f"Global V_eff peak: {float(np.max(V_eff)):.3f} mV")
        print(f"Global V_eff min:  {float(np.min(V_eff)):.3f} mV")

    _print_probe_summary(cable_result, geometry)
    print("=" * 72)

    # Threshold-based propagation reporting.
    print(format_node_report(cable_result, geometry, threshold_mV=10.0))
    print(format_multi_threshold_report(cable_result, geometry, thresholds_mV=(0.0, 10.0, 20.0)))

    # -------------------------------------------------------------------------
    # Volume conduction summary
    # -------------------------------------------------------------------------
    print("Volume conduction tests")
    print("-" * 72)

    # Local import keeps this demonstration function near where it is used.
    from Scripts.NeuroSolver.ECS.vc_solver import compute_point_source_phi_e_mV

    # Simple analytic demonstration:
    # extracellular potential 100 um away from a 10 uA point source.
    #
    # Why these numbers?
    # - 10 uA gives a clearly nonzero test value.
    # - 100 um is a plausible micrometer-scale neural distance.
    phi_demo = compute_point_source_phi_e_mV(
        source_current_uA=10.0,
        source_xyz_um=(0.0, 0.0, 0.0),
        sample_xyz_um=(100.0, 0.0, 0.0),
        sigma_S_per_m=params.vc.sigma_x_S_per_m,
        epsilon_um=params.vc.singularity_epsilon_um,
    )
    print(f"phi_e at 100 um from 10 uA point source: {phi_demo:.6f} mV")

    # If full VC traces were produced, report simple summary stats.
    if vc_result is not None:
        for name, trace in vc_result.items():
            if name == "t_ms":
                continue
            arr = np.asarray(trace, dtype=float)
            print(
                f"{name}: peak={np.max(arr):.6f} mV | "
                f"min={np.min(arr):.6f} mV | ptp={np.ptp(arr):.6f} mV"
            )
    print("=" * 72)

    # -------------------------------------------------------------------------
    # KNP / slow ECS summary
    # -------------------------------------------------------------------------
    print("KNP / coupled slow ECS test")
    print("-" * 72)
    if knp_result is not None:
        x_um = np.asarray(knp_result["x_um"], dtype=float)
        t_ms = np.asarray(knp_result["t_ms"], dtype=float)
        phi_e_mV = np.asarray(knp_result["phi_e_mV"], dtype=float)
        species_mM = knp_result["species_mM"]

        # Mid-domain point used for simple concentration reporting.
        center_idx = int(np.argmin(np.abs(x_um - 0.5 * x_um[-1])))

        active_mode = knp_result.get(
            "coupling_mode",
            metadata.get("knp_coupling_mode", getattr(params.knp, "coupling_mode", "physiologic")),
        )
        active_scale = float(
            knp_result.get(
                "coupling_scale_mM_per_ms_per_uA",
                metadata.get("knp_coupling_scale_mM_per_ms_per_uA", 0.0),
            )
        )
        source_payload_metadata = knp_result.get("source_payload_metadata", {})

        print(f"Domain points: {x_um.size}")
        print(f"Time samples:  {t_ms.size}")
        print(f"Tracked species: {', '.join(sorted(species_mM.keys()))}")
        print(f"KNP coupling mode:  {active_mode}")
        print(f"KNP coupling scale: {active_scale:.6e} mM/ms/uA")

        # Debug metadata about how the KNP source term was assembled.
        if source_payload_metadata:
            print(f"Source payload species: {source_payload_metadata.get('species_names', 'n/a')}")
            print(f"Source payload fiber count: {source_payload_metadata.get('fiber_count', 'n/a')}")
            print(
                f"Source payload fallback species: "
                f"{source_payload_metadata.get('fallback_species_name', 'n/a')}"
            )
            print(
                f"Source volume-fraction scaling: "
                f"{bool(getattr(params.knp, 'source_volume_fraction_scaling', False))}"
            )
            print(f"Clearance enabled: {bool(getattr(params.knp, 'clearance_enabled', False))}")

        # Potassium-specific reporting.
        # Why K?
        # Extracellular K+ is often the clearest first indicator of activity-dependent
        # ECS changes and is highly interpretable physiologically.
        if "K" in species_mM:
            K_baseline = float(params.ions.potassium.extracellular_mM)
            K_hist = np.asarray(species_mM["K"], dtype=float)
            delta_K = K_hist - K_baseline
            max_abs_delta_K = float(np.max(np.abs(delta_K)))

            print(f"Center concentration start: {K_hist[0, center_idx]:.6f} mM")
            print(f"Center concentration end:   {K_hist[-1, center_idx]:.6f} mM")
            print(f"Center delta-K end:         {delta_K[-1, center_idx]:.6e} mM")
            print(f"Max |delta K| over domain/time: {max_abs_delta_K:.6e} mM")

        max_abs_phi = float(np.max(np.abs(phi_e_mV)))
        print(f"phi_e range: [{np.min(phi_e_mV):.6f}, {np.max(phi_e_mV):.6f}] mV")
        print(f"Max |phi_e| over domain/time: {max_abs_phi:.6e} mV")

        # Source-term debug info if available.
        if "source_debug" in knp_result:
            source_debug = knp_result["source_debug"]
            print(
                "Max projected KNP source magnitude: "
                f"{float(source_debug.get('max_abs_source_mM_per_ms', 0.0)):.6e} mM/ms"
            )

            per_fiber_debug = source_debug.get("max_abs_source_per_fiber", None)
            if isinstance(per_fiber_debug, dict):
                print("Per-fiber max projected source magnitude [mM/ms]:")
                for fid in sorted(per_fiber_debug.keys()):
                    print(f"  fiber {int(fid)}: {float(per_fiber_debug[fid]):.6e}")
    else:
        print("KNP result unavailable.")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Slow ECS feedback summary
    # -------------------------------------------------------------------------
    # feedback_history may exist directly for single-fiber runs or be stored
    # per-fiber in bundle runs.
    feedback_hist = metadata.get("feedback_history", None)

    if feedback_hist is None and bundle is not None:
        per_fiber_feedback = metadata.get("per_fiber_feedback_history", None)
        if isinstance(per_fiber_feedback, dict):
            feedback_hist = per_fiber_feedback.get(reporting_fiber_id, None)

    print("Slow ECS feedback summary")
    print("-" * 72)
    print(f"Feedback enabled: {bool(metadata.get('feedback_enabled', False))}")

    if isinstance(feedback_hist, dict) and feedback_hist:
        phi_hist = np.asarray(feedback_hist.get("phi_e_cable_mV", []), dtype=float)
        E_na_hist = np.asarray(feedback_hist.get("E_na_mV", []), dtype=float)
        E_k_hist = np.asarray(feedback_hist.get("E_k_mV", []), dtype=float)

        print(f"Snapshots:        {phi_hist.shape[0] if phi_hist.ndim >= 1 else 0}")

        if phi_hist.size > 0:
            print(
                f"phi_e feedback range: "
                f"[{float(np.min(phi_hist)):.6f}, {float(np.max(phi_hist)):.6f}] mV"
            )
            print(f"Max |phi_e feedback|: {float(np.max(np.abs(phi_hist))):.6e} mV")

        if E_na_hist.size > 0:
            print(f"E_Na slow range: [{float(np.min(E_na_hist)):.6f}, {float(np.max(E_na_hist)):.6f}] mV")

        if E_k_hist.size > 0:
            print(f"E_K slow range:  [{float(np.min(E_k_hist)):.6f}, {float(np.max(E_k_hist)):.6f}] mV")
    else:
        print("Snapshots:        0")

    # Last reversal snapshot may also be stored differently for bundle runs.
    last_rev = metadata.get("feedback_last_reversals_mV", None)
    if last_rev is None and bundle is not None:
        last_rev_by_fiber = metadata.get("feedback_last_reversals_mV_by_fiber", None)
        if isinstance(last_rev_by_fiber, dict):
            last_rev = last_rev_by_fiber.get(reporting_fiber_id, None)

    if isinstance(last_rev, dict):
        print("Last slow reversal snapshot:")
        for key in sorted(last_rev.keys()):
            print(f"  {key}: {float(last_rev[key]):.6f} mV")

    # Summary stats for the last slow extracellular potential feedback state.
    last_phi_stats = metadata.get("feedback_last_phi_stats_mV", None)
    if last_phi_stats is None and bundle is not None:
        last_phi_stats_by_fiber = metadata.get("feedback_last_phi_stats_mV_by_fiber", None)
        if isinstance(last_phi_stats_by_fiber, dict):
            last_phi_stats = last_phi_stats_by_fiber.get(reporting_fiber_id, None)

    if isinstance(last_phi_stats, dict):
        print(
            "Last slow phi stats [mV]: "
            f"mean={float(last_phi_stats.get('mean', 0.0)):.6f}, "
            f"min={float(last_phi_stats.get('min', 0.0)):.6f}, "
            f"max={float(last_phi_stats.get('max', 0.0)):.6f}"
        )
    print("=" * 72)


def main() -> None:
    """
    Main entrypoint.

    Run sequence
    ------------
    1. Build the medium-myelinated physiological parameter preset.
    2. Convert stimulus settings into a validation-consistent parameter set.
    3. Print parameter summary.
    4. Smoke-test stimuli.
    5. Run single-node excitability test.
    6. Run integrated multirate test.

    Order goes from cheapest / most local validation to most integrated:
    parameters -> waveform -> node -> cable/bundle/ECS.
    """
    params = build_medium_myelinated_physiology_params(DEFAULT_PARAMS)
    run_params = build_validation_run_params(params)

    print_param_summary(run_params)
    test_stimuli(run_params)
    test_single_node(run_params)
    test_integrated_multirate(run_params)


if __name__ == "__main__":
    main()