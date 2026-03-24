"""
scheduler.py

Multirate scheduler for NeuroSolver.

This revision keeps the feedback plumbing but makes the runtime switches derive
from the active preset by default, so the physiological preset becomes the
actual source of truth for an integrated run.

Phase 2 extension
-----------------
This file now supports both:
- the existing single-fiber multirate scheduler, and
- a bundle-aware fast scheduler that advances all fibers each fast step,
  aggregates shared KNP source loading, and superposes VC traces across fibers.

Important current scope
-----------------------
This is an intentionally conservative Phase 2 implementation:
- fast cable stepping is explicit per fiber,
- shared KNP loading is formed by summing per-fiber projected sources,
- shared ECS feedback is sampled back onto each fiber from the same slow domain,
- VC traces are produced by superposing per-fiber traces, using either the
    homogeneous-medium approximation or the material-aware shielded VC model
    when a material model is provided.
This preserves the current 1D KNP / homogeneous VC approximations while making
multi-fiber scaling emerge from explicit geometry rather than from multiplying a
single fiber by N.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Any
import inspect
import numpy as np

from Scripts.NeuroSolver.params import DEFAULT_PARAMS, SimulationParameters
from Scripts.NeuroSolver.state import (
    CableHistoryBuffer,
    KNPHistoryBuffer,
    MultirateSimulationResult,
)
from Scripts.NeuroSolver.propagation.cable_solver import (
    initialize_cable_state,
    advance_cable_state,
)
from Scripts.NeuroSolver.ECS.vc_solver import ElectrodeSamplePoint
from Scripts.NeuroSolver.ECS.coupling import (
    cable_to_vc_traces,
    #project_cable_density_to_knp_source_terms,
    project_cable_density_to_shared_source_payload,
    project_bundle_source_payloads_to_shared_terms,
)
from Scripts.NeuroSolver.ECS.knp_solver import (
    build_uniform_knp_domain_1d,
    initialize_species_concentrations_mM,
    initialize_phi_e_mV,
    advance_knp_state_1d,
)
from Scripts.NeuroSolver.ECS.ecs_feedback import (
    build_feedback_snapshot,
    build_initial_feedback_snapshot,
    resolve_feedback_config,
)

@dataclass
class BundleMultirateSimulationResult:
    """
    Output bundle for Phase 2 multi-fiber execution.

    Notes
    -----
    This is defined locally so the scheduler can move into Phase 2 without
    forcing an immediate edit to state.py.
    """
    per_fiber_cable_results: Dict[int, Dict[str, np.ndarray]]
    bundle_history: Dict[str, object]
    vc_result: Optional[Dict[str, np.ndarray]] = None
    knp_result: Optional[Dict[str, object]] = None
    metadata: Optional[Dict[str, object]] = None

    @property
    def fiber_results(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Backward-compatible alias expected by older main.py reporting helpers.
        """
        return self.per_fiber_cable_results

class FeedbackHistoryBuffer:
    """
    Time history of ECS feedback quantities mapped back onto the cable model.

    Stored separately from the generic cable/KNP histories so the regression
    anchor can explicitly track slow ECS-induced drift in:
    - extracellular cable potential,
    - Nernst reversal overrides.
    """

    def __init__(self) -> None:
        self.t_ms: List[float] = []
        self.phi_e_cable_mV: List[np.ndarray] = []
        self.E_na_mV: List[float] = []
        self.E_k_mV: List[float] = []
        self.E_cl_mV: List[float] = []
        self.E_ca_mV: List[float] = []

    def append(self, t_ms: float, snapshot) -> None:
        self.t_ms.append(float(t_ms))
        self.phi_e_cable_mV.append(np.asarray(snapshot.phi_e_cable_mV, dtype=float).copy())
        rev = snapshot.reversal_overrides_mV
        self.E_na_mV.append(float(rev.get("E_na_mV", np.nan)))
        self.E_k_mV.append(float(rev.get("E_k_mV", np.nan)))
        self.E_cl_mV.append(float(rev.get("E_cl_mV", np.nan)))
        self.E_ca_mV.append(float(rev.get("E_ca_mV", np.nan)))

    def finalize(self) -> Dict[str, np.ndarray]:
        return {
            "t_ms": np.asarray(self.t_ms, dtype=float),
            "phi_e_cable_mV": np.asarray(self.phi_e_cable_mV, dtype=float),
            "E_na_mV": np.asarray(self.E_na_mV, dtype=float),
            "E_k_mV": np.asarray(self.E_k_mV, dtype=float),
            "E_cl_mV": np.asarray(self.E_cl_mV, dtype=float),
            "E_ca_mV": np.asarray(self.E_ca_mV, dtype=float),
        }

class BundleHistoryBuffer:
    """
    Bundle-level fast history buffer.

    Stores per-fiber voltage and current outputs on the common fast timestep.
    This is intentionally lightweight so it can be used before more elaborate
    bundle diagnostics are added in later phases.
    """

    def __init__(self) -> None:
        self.t_ms: List[float] = []
        self.fiber_voltage_history: Dict[int, List[np.ndarray]] = {}
        self.fiber_ion_history: Dict[int, List[np.ndarray]] = {}
        self.fiber_app_history: Dict[int, List[np.ndarray]] = {}
        self.fiber_app_abs_history: Dict[int, List[np.ndarray]] = {}
        self.fiber_phi_history: Dict[int, List[np.ndarray]] = {}
        self.fiber_v_eff_history: Dict[int, List[np.ndarray]] = {}

    def append(self, t_ms: float, fiber_states: Dict[int, Dict[str, np.ndarray]]) -> None:
        self.t_ms.append(float(t_ms))
        for fiber_id, state in fiber_states.items():
            fid = int(fiber_id)
            v_vec = np.asarray(state["V_m_mV"], dtype=float)
            zero_vec = np.zeros_like(v_vec)

            self.fiber_voltage_history.setdefault(fid, []).append(v_vec.copy())
            self.fiber_ion_history.setdefault(fid, []).append(
                np.asarray(state.get("I_ion_uA_per_cm2", zero_vec), dtype=float).copy()
            )
            self.fiber_app_history.setdefault(fid, []).append(
                np.asarray(state.get("I_app_uA_per_cm2", zero_vec), dtype=float).copy()
            )
            self.fiber_app_abs_history.setdefault(fid, []).append(
                np.asarray(state.get("I_app_abs_uA", zero_vec), dtype=float).copy()
            )
            self.fiber_phi_history.setdefault(fid, []).append(
                np.asarray(state.get("phi_e_cable_mV", zero_vec), dtype=float).copy()
            )
            self.fiber_v_eff_history.setdefault(fid, []).append(
                np.asarray(state.get("V_membrane_effective_mV", v_vec), dtype=float).copy()
            )

    @staticmethod
    def _coerce_history_dict(name: str, store: Dict[int, List[np.ndarray]]) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for fiber_id, history in store.items():
            try:
                out[fiber_id] = np.asarray(history, dtype=float)
            except ValueError as exc:
                shapes = [tuple(np.asarray(x).shape) for x in history]
                raise ValueError(
                    f"{name} for fiber {fiber_id} has inconsistent shapes: {shapes}"
                ) from exc
        return out

    def finalize(self) -> Dict[str, object]:
        return {
            "t_ms": np.asarray(self.t_ms, dtype=float),
            "fiber_voltage_history": self._coerce_history_dict("fiber_voltage_history", self.fiber_voltage_history),
            "fiber_ion_history": self._coerce_history_dict("fiber_ion_history", self.fiber_ion_history),
            "fiber_app_history": self._coerce_history_dict("fiber_app_history", self.fiber_app_history),
            "fiber_app_abs_history": self._coerce_history_dict("fiber_app_abs_history", self.fiber_app_abs_history),
            "fiber_phi_history": self._coerce_history_dict("fiber_phi_history", self.fiber_phi_history),
            "fiber_v_eff_history": self._coerce_history_dict("fiber_v_eff_history", self.fiber_v_eff_history),
        }

def _resolve_knp_coupling_scale(params: SimulationParameters) -> float:
    """
    Resolve the effective cable->KNP source scaling.

    Two intended modes exist:
    - physiologic: small source magnitudes
    - visibility_test: intentionally amplified source magnitudes for debugging
    """
    mode = str(getattr(params.knp, "coupling_mode", "physiologic")).strip().lower()
    if mode == "visibility_test":
        return float(getattr(params.knp, "source_scale_visibility_mM_per_ms_per_uA", 1.0e-2))
    return float(getattr(params.knp, "source_scale_physiologic_mM_per_ms_per_uA", 1.0e-5))

def _supports_kwarg(func, name: str) -> bool:
    """
    Runtime compatibility helper.

    Some subsystems may lag behind interface changes. This lets the scheduler
    adapt without crashing if an optional kwarg is not yet supported.
    """
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False

def _resolve_runtime_switches(
    params: SimulationParameters,
    enable_vc: Optional[bool],
    enable_knp: Optional[bool],
    enable_ecs_feedback: Optional[bool],
) -> tuple[bool, bool, bool]:
    preset = getattr(params, "preset", None)

    if enable_vc is None:
        enable_vc = bool(
            getattr(preset, "runtime_enable_vc", getattr(params.vc, "enabled", False))
        )
    if enable_knp is None:
        enable_knp = bool(
            getattr(preset, "runtime_enable_knp", getattr(params.knp, "enabled", False))
        )

    feedback_config = resolve_feedback_config(params)
    if enable_ecs_feedback is None:
        feedback_enabled = bool(
            getattr(preset, "runtime_enable_ecs_feedback", feedback_config.enabled)
        ) and bool(enable_knp)
    else:
        feedback_enabled = bool(enable_ecs_feedback) and bool(enable_knp)

    return bool(enable_vc), bool(enable_knp), bool(feedback_enabled)

def _sum_source_terms(source_terms_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not source_terms_list:
        return {}
    out: Dict[str, np.ndarray] = {}
    for terms in source_terms_list:
        for species, arr in terms.items():
            arr_f = np.asarray(arr, dtype=float)
            if species not in out:
                out[species] = np.zeros_like(arr_f)
            out[species] += arr_f
    return out

def _sum_vc_results(vc_results: List[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
    if not vc_results:
        return None
    out: Dict[str, np.ndarray] = {}
    for result in vc_results:
        for key, value in result.items():
            arr = np.asarray(value, dtype=float)
            if key == "t_ms":
                if key not in out:
                    out[key] = arr.copy()
                continue
            if key not in out:
                out[key] = np.zeros_like(arr)
            out[key] += arr
    return out

def _build_single_fiber_feedback_map(
    geometry: Dict[str, np.ndarray],
    params: SimulationParameters,
    knp_domain,
    knp_concentrations,
    knp_phi_e_mV,
    initial_phi_reference: np.ndarray,
):
    return build_feedback_snapshot(
        knp_x_um=knp_domain.x_um,
        concentrations_mM=knp_concentrations,
        phi_e_mV=knp_phi_e_mV,
        cable_x_um=np.asarray(geometry["x_um"], dtype=float),
        params=params,
        initial_phi_reference_mV=initial_phi_reference,
    )

def run_multirate_simulation(
    geometry: Dict[str, np.ndarray],
    params: SimulationParameters = DEFAULT_PARAMS,
    electrode_points: Optional[Sequence[ElectrodeSamplePoint]] = None,
    stimulated_index: Optional[int] = None,
    enable_vc: Optional[bool] = None,
    enable_knp: Optional[bool] = None,
    enable_ecs_feedback: Optional[bool] = None,
    knp_species: str = "K",
    cable_y_um: Optional[float] = None,
    cable_z_um: Optional[float] = None,
    material_model: Optional[Any] = None,
) -> MultirateSimulationResult:
    """
    Run a single-fiber multirate simulation.

    Fast loop:
        cable / membrane solver

    Slow loop:
        KNP + ECS feedback

    Optional:
        VC electrode sampling

    Phase 1 compatibility:
    ----------------------
    geometry may now contain bundle metadata fields such as:
    - fiber_id
    - fascicle_id
    - fiber_center_y_um
    - fiber_center_z_um

    but only one fiber is stepped here.
    """
    enable_vc, enable_knp, feedback_enabled = _resolve_runtime_switches(
        params=params,
        enable_vc=enable_vc,
        enable_knp=enable_knp,
        enable_ecs_feedback=enable_ecs_feedback,
    )
    preset = getattr(params, "preset", None)

    dt_fast_ms = float(params.solver.dt_fast_ms)
    t_stop_ms = float(params.solver.t_stop_ms)
    n_fast_steps = int(round(t_stop_ms / dt_fast_ms))

    cable_state = initialize_cable_state(geometry=geometry, params=params)

    n_compartments = int(geometry["n_compartments"])
    zero_vec = np.zeros(n_compartments, dtype=float)

    cable_hist = CableHistoryBuffer()
    knp_hist: Optional[KNPHistoryBuffer] = None
    feedback_hist = FeedbackHistoryBuffer()

    knp_domain = None
    knp_concentrations = None
    knp_phi_e_mV = None

    dt_slow_ms = float(getattr(params.knp, "dt_ms", params.solver.dt_slow_ms))
    slow_stride = max(1, int(round(dt_slow_ms / dt_fast_ms)))

    knp_coupling_scale_mM_per_ms_per_uA = _resolve_knp_coupling_scale(params)
    knp_source_debug = {"max_abs_source_mM_per_ms": 0.0}

    active_feedback = build_initial_feedback_snapshot(geometry=geometry, params=params)
    initial_phi_reference = np.asarray(active_feedback.phi_e_cable_mV, dtype=float).copy()

    if enable_knp:
        knp_hist = KNPHistoryBuffer()
        knp_domain = build_uniform_knp_domain_1d(
            length_um=float(params.knp.domain_length_um),
            dx_um=float(params.knp.dx_um),
        )
        knp_concentrations = initialize_species_concentrations_mM(knp_domain, params)
        knp_phi_e_mV = initialize_phi_e_mV(knp_domain)

    feedback_hist.append(0.0, active_feedback)

    cable_accepts_feedback = _supports_kwarg(advance_cable_state, "ecs_feedback")
    cable_history_accepts_extra = _supports_kwarg(cable_hist.append, "extra")

    for step in range(n_fast_steps + 1):
        t_ms = step * dt_fast_ms

        append_kwargs = dict(
            t_ms=t_ms,
            V_m_mV=np.asarray(cable_state["V_m_mV"], dtype=float),
            I_ion_uA_per_cm2=np.asarray(cable_state.get("I_ion_uA_per_cm2", zero_vec), dtype=float),
            I_app_uA_per_cm2=np.asarray(cable_state.get("I_app_uA_per_cm2", zero_vec), dtype=float),
            I_app_abs_uA=np.asarray(cable_state.get("I_app_abs_uA", zero_vec), dtype=float),
        )
        if cable_history_accepts_extra:
            append_kwargs["extra"] = {
                "phi_e_cable_mV": np.asarray(cable_state.get("phi_e_cable_mV", zero_vec), dtype=float),
                "V_membrane_effective_mV": np.asarray(
                    cable_state.get("V_membrane_effective_mV", cable_state["V_m_mV"]), dtype=float
                ),
            }
        cable_hist.append(**append_kwargs)

        advance_kwargs = dict(
            state=cable_state,
            geometry=geometry,
            t_ms=t_ms,
            dt_ms=dt_fast_ms,
            params=params,
            stimulated_index=stimulated_index,
        )
        if cable_accepts_feedback:
            advance_kwargs["ecs_feedback"] = {
                "phi_e_cable_mV": active_feedback.phi_e_cable_mV,
                "reversal_overrides_mV": active_feedback.reversal_overrides_mV,
            } if feedback_enabled else None

        cable_state = advance_cable_state(**advance_kwargs)

        if enable_knp and knp_domain is not None and knp_hist is not None and step % slow_stride == 0:
            I_ion_density = np.asarray(cable_state.get("I_ion_uA_per_cm2", zero_vec), dtype=float)

            source_payload = project_cable_density_to_shared_source_payload(
                I_membrane_density_uA_per_cm2=I_ion_density,
                geometry=geometry,
                knp_x_um=knp_domain.x_um,
                species_current_density_uA_per_cm2=cable_state.get("species_current_density_uA_per_cm2", None),
                fallback_species_name=knp_species,
                scale_mM_per_ms_per_uA=knp_coupling_scale_mM_per_ms_per_uA,
            )
            source_terms = source_payload["species_sources_mM_per_ms"]

            if knp_species in source_terms:
                knp_source_debug["max_abs_source_mM_per_ms"] = max(
                    float(knp_source_debug["max_abs_source_mM_per_ms"]),
                    float(np.max(np.abs(source_terms[knp_species]))),
                )

            knp_updated = advance_knp_state_1d(
                concentrations_mM=knp_concentrations,
                phi_e_mV=knp_phi_e_mV,
                domain=knp_domain,
                dt_ms=dt_slow_ms,
                params=params,
                source_terms_mM_per_ms=source_terms,
            )
            knp_concentrations = knp_updated["concentrations_mM"]
            knp_phi_e_mV = knp_updated["phi_e_mV"]

            knp_hist.append(
                t_ms=t_ms,
                phi_e_mV=knp_phi_e_mV,
                concentrations_mM=knp_concentrations,
            )

            if feedback_enabled:
                active_feedback = _build_single_fiber_feedback_map(
                    geometry=geometry,
                    params=params,
                    knp_domain=knp_domain,
                    knp_concentrations=knp_concentrations,
                    knp_phi_e_mV=knp_phi_e_mV,
                    initial_phi_reference=initial_phi_reference,
                )
                feedback_hist.append(t_ms=t_ms, snapshot=active_feedback)

    cable_result = cable_hist.finalize(geometry)

    if cable_y_um is None:
        cable_y_um = float(geometry.get("fiber_center_y_um", 0.0))
    if cable_z_um is None:
        cable_z_um = float(geometry.get("fiber_center_z_um", 0.0))

    vc_result = None
    if enable_vc and electrode_points is not None and len(electrode_points) > 0:
        vc_result = cable_to_vc_traces(
            cable_result=cable_result,
            geometry=geometry,
            electrode_points=electrode_points,
            params=params,
            cable_y_um=float(cable_y_um),
            cable_z_um=float(cable_z_um),
            material_model=material_model
        )

    knp_result = None
    if enable_knp and knp_hist is not None and knp_domain is not None:
        knp_result = knp_hist.finalize(x_um=knp_domain.x_um)
        knp_result["source_debug"] = knp_source_debug
        if "source_payload" in locals():
            knp_result["source_payload_metadata"] = dict(source_payload["source_metadata"])
        knp_result["coupling_mode"] = str(getattr(params.knp, "coupling_mode", "physiologic"))
        knp_result["coupling_scale_mM_per_ms_per_uA"] = float(knp_coupling_scale_mM_per_ms_per_uA)

    metadata = {
    "preset_name": getattr(preset, "active_preset_name", "unspecified"),
    "dt_fast_ms": dt_fast_ms,
    "dt_slow_ms": dt_slow_ms,
    "enable_vc": bool(enable_vc),
    "enable_knp": bool(enable_knp),
    "feedback_enabled": bool(feedback_enabled),
    "feedback_policy_enabled": bool(getattr(params.policy, "use_dynamic_ecs_feedback", False)),
    "feedback_runtime_override": enable_ecs_feedback,
    "feedback_cable_support": bool(cable_accepts_feedback),
    "knp_species": knp_species,
    "slow_stride": slow_stride,
    "knp_coupling_mode": str(getattr(params.knp, "coupling_mode", "physiologic")),
    "knp_coupling_scale_mM_per_ms_per_uA": float(knp_coupling_scale_mM_per_ms_per_uA),
    "fiber_id": int(geometry.get("fiber_id", 0)),
    "fascicle_id": geometry.get("fascicle_id", None),
    "fiber_center_y_um": float(geometry.get("fiber_center_y_um", 0.0)),
    "fiber_center_z_um": float(geometry.get("fiber_center_z_um", 0.0)),
    "feedback_history": feedback_hist.finalize(),
    "feedback_last_reversals_mV": dict(active_feedback.reversal_overrides_mV),
    "feedback_last_phi_stats_mV": {
        "mean": float(np.mean(active_feedback.phi_e_cable_mV)),
        "min": float(np.min(active_feedback.phi_e_cable_mV)),
        "max": float(np.max(active_feedback.phi_e_cable_mV)),
    },
}

    return MultirateSimulationResult(
        cable_result=cable_result,
        vc_result=vc_result,
        knp_result=knp_result,
        metadata=metadata,
    )

def run_multirate_bundle_simulation(
    bundle_geometry: Any,
    bundle_state: Optional[Any] = None,
    params: SimulationParameters = DEFAULT_PARAMS,
    electrode_points: Optional[Sequence[ElectrodeSamplePoint]] = None,
    stimulated_fiber_ids: Optional[Sequence[int]] = None,
    enable_vc: Optional[bool] = None,
    enable_knp: Optional[bool] = None,
    enable_ecs_feedback: Optional[bool] = None,
    knp_species: str = "K",
    material_model: Optional[Any] = None
) -> BundleMultirateSimulationResult:
    """
    Run the Phase 2 bundle-aware multirate scheduler.

    Fast layer:
        advance every fiber each fast timestep using its own cable state

    Shared slow layer:
        aggregate all per-fiber projected KNP source terms into one shared domain

    Feedback:
        sample the shared ECS state back to each fiber independently using each
        fiber's x-axis geometry

    VC:
        compute per-fiber homogeneous-medium traces and superpose them
    """
    enable_vc, enable_knp, feedback_enabled = _resolve_runtime_switches(
        params=params,
        enable_vc=enable_vc,
        enable_knp=enable_knp,
        enable_ecs_feedback=enable_ecs_feedback,
    )
    preset = getattr(params, "preset", None)

    # Avoid a hard dependency on the bundle_state module path here.  If the
    # caller does not supply one, initialize it lazily using the in-project
    # module.
    if bundle_state is None:
        from Scripts.NeuroSolver.bundle_state import initialize_bundle_state  # type: ignore
        bundle_state = initialize_bundle_state(bundle_geometry=bundle_geometry, params=params)

    dt_fast_ms = float(params.solver.dt_fast_ms)
    t_stop_ms = float(params.solver.t_stop_ms)
    n_fast_steps = int(round(t_stop_ms / dt_fast_ms))

    dt_slow_ms = float(getattr(params.knp, "dt_ms", params.solver.dt_slow_ms))
    slow_stride = max(1, int(round(dt_slow_ms / dt_fast_ms)))

    cable_accepts_feedback = _supports_kwarg(advance_cable_state, "ecs_feedback")
    knp_coupling_scale_mM_per_ms_per_uA = _resolve_knp_coupling_scale(params)

    fiber_geometries: Dict[int, Dict[str, Any]] = {
        int(fid): bundle_geometry.get_fiber_geometry(fid) for fid in bundle_geometry.fiber_ids
    }
    fiber_ids = [int(fid) for fid in bundle_geometry.fiber_ids]
    stimulated_fiber_id_set = None if stimulated_fiber_ids is None else {int(fid) for fid in stimulated_fiber_ids}

    per_fiber_histories: Dict[int, CableHistoryBuffer] = {fid: CableHistoryBuffer() for fid in fiber_ids}
    per_fiber_feedback_histories: Dict[int, FeedbackHistoryBuffer] = {fid: FeedbackHistoryBuffer() for fid in fiber_ids}
    bundle_hist = BundleHistoryBuffer()

    knp_hist: Optional[KNPHistoryBuffer] = None
    knp_domain = None
    knp_concentrations = None
    knp_phi_e_mV = None
    knp_source_debug = {
        "max_abs_source_mM_per_ms": 0.0,
        "max_abs_source_per_fiber": {fid: 0.0 for fid in fiber_ids},
    }

    active_feedback_by_fiber = {
        fid: build_initial_feedback_snapshot(geometry=fiber_geometries[fid], params=params)
        for fid in fiber_ids
    }
    initial_phi_reference_by_fiber = {
        fid: np.asarray(active_feedback_by_fiber[fid].phi_e_cable_mV, dtype=float).copy()
        for fid in fiber_ids
    }

    if enable_knp:
        knp_hist = KNPHistoryBuffer()
        knp_domain = build_uniform_knp_domain_1d(
            length_um=float(params.knp.domain_length_um),
            dx_um=float(params.knp.dx_um),
        )
        knp_concentrations = initialize_species_concentrations_mM(knp_domain, params)
        knp_phi_e_mV = initialize_phi_e_mV(knp_domain)

    for fid in fiber_ids:
        per_fiber_feedback_histories[fid].append(0.0, active_feedback_by_fiber[fid])

    for step in range(n_fast_steps + 1):
        t_ms = step * dt_fast_ms

        # Record current fast state for every fiber before the step.
        bundle_hist.append(t_ms=t_ms, fiber_states=bundle_state.fiber_states)
        for fid in fiber_ids:
            fiber_state = bundle_state.fiber_states[fid]
            geometry = fiber_geometries[fid]
            zero_vec = np.zeros(int(geometry["n_compartments"]), dtype=float)
            append_kwargs = dict(
                t_ms=t_ms,
                V_m_mV=np.asarray(fiber_state["V_m_mV"], dtype=float),
                I_ion_uA_per_cm2=np.asarray(fiber_state.get("I_ion_uA_per_cm2", zero_vec), dtype=float),
                I_app_uA_per_cm2=np.asarray(fiber_state.get("I_app_uA_per_cm2", zero_vec), dtype=float),
                I_app_abs_uA=np.asarray(fiber_state.get("I_app_abs_uA", zero_vec), dtype=float),
                extra={
                    "phi_e_cable_mV": np.asarray(fiber_state.get("phi_e_cable_mV", zero_vec), dtype=float),
                    "V_membrane_effective_mV": np.asarray(
                        fiber_state.get("V_membrane_effective_mV", fiber_state["V_m_mV"]), dtype=float
                    ),
                },
            )
            per_fiber_histories[fid].append(**append_kwargs)

        # Advance every fiber on the fast clock.
        next_states: Dict[int, Dict[str, np.ndarray]] = {}
        for fid in fiber_ids:
            fiber_state = bundle_state.fiber_states[fid]
            geometry = fiber_geometries[fid]
            advance_kwargs = dict(
                state=fiber_state,
                geometry=geometry,
                t_ms=t_ms,
                dt_ms=dt_fast_ms,
                params=params,
                stimulated_index=None if stimulated_fiber_id_set is None or fid in stimulated_fiber_id_set else -1,
            )
            if cable_accepts_feedback:
                advance_kwargs["ecs_feedback"] = {
                    "phi_e_cable_mV": active_feedback_by_fiber[fid].phi_e_cable_mV,
                    "reversal_overrides_mV": active_feedback_by_fiber[fid].reversal_overrides_mV,
                } if feedback_enabled else None

            next_states[fid] = advance_cable_state(**advance_kwargs)

        bundle_state.fiber_states = next_states

        # Aggregate slow shared ECS loading from all fibers.
        if enable_knp and knp_domain is not None and knp_hist is not None and step % slow_stride == 0:
            per_fiber_payloads: List[Dict[str, Any]] = []

            for fid in fiber_ids:
                geometry = fiber_geometries[fid]
                zero_vec = np.zeros(int(geometry["n_compartments"]), dtype=float)

                I_ion_density = np.asarray(
                    bundle_state.fiber_states[fid].get("I_ion_uA_per_cm2", zero_vec),
                    dtype=float,
                )

                source_payload = project_cable_density_to_shared_source_payload(
                    I_membrane_density_uA_per_cm2=I_ion_density,
                    geometry=geometry,
                    knp_x_um=knp_domain.x_um,
                    species_current_density_uA_per_cm2=bundle_state.fiber_states[fid].get(
                        "species_current_density_uA_per_cm2", None
                    ),
                    fallback_species_name=knp_species,
                    scale_mM_per_ms_per_uA=knp_coupling_scale_mM_per_ms_per_uA,
                )
                per_fiber_payloads.append(source_payload)

                species_sources = source_payload["species_sources_mM_per_ms"]
                if knp_species in species_sources:
                    local_max = float(np.max(np.abs(species_sources[knp_species])))
                    knp_source_debug["max_abs_source_per_fiber"][fid] = max(
                        float(knp_source_debug["max_abs_source_per_fiber"][fid]),
                        local_max,
                    )
                    knp_source_debug["max_abs_source_mM_per_ms"] = max(
                        float(knp_source_debug["max_abs_source_mM_per_ms"]),
                        local_max,
                    )

            shared_payload = project_bundle_source_payloads_to_shared_terms(per_fiber_payloads)
            aggregated_source_terms = shared_payload["species_sources_mM_per_ms"]
            
            knp_updated = advance_knp_state_1d(
                concentrations_mM=knp_concentrations,
                phi_e_mV=knp_phi_e_mV,
                domain=knp_domain,
                dt_ms=dt_slow_ms,
                params=params,
                source_terms_mM_per_ms=aggregated_source_terms,
            )
            knp_concentrations = knp_updated["concentrations_mM"]
            knp_phi_e_mV = knp_updated["phi_e_mV"]

            knp_hist.append(
                t_ms=t_ms,
                phi_e_mV=knp_phi_e_mV,
                concentrations_mM=knp_concentrations,
            )

            bundle_state.shared_ecs_state = {
                "t_ms": float(t_ms),
                "x_um": np.asarray(knp_domain.x_um, dtype=float).copy(),
                "phi_e_mV": np.asarray(knp_phi_e_mV, dtype=float).copy(),
                "concentrations_mM": {k: np.asarray(v, dtype=float).copy() for k, v in knp_concentrations.items()},
                "source_terms_mM_per_ms": {k: np.asarray(v, dtype=float).copy() for k, v in aggregated_source_terms.items()},
                "total_membrane_source_uA": np.asarray(shared_payload["total_membrane_source_uA"], dtype=float).copy(),
                "capacitive_source_uA": np.asarray(shared_payload["capacitive_source_uA"], dtype=float).copy(),
                "vc_point_sources": list(shared_payload["vc_point_sources"]),
                "source_metadata": dict(shared_payload["source_metadata"]),
            }

            if feedback_enabled:
                for fid in fiber_ids:
                    geometry = fiber_geometries[fid]
                    active_feedback_by_fiber[fid] = build_feedback_snapshot(
                        knp_x_um=knp_domain.x_um,
                        concentrations_mM=knp_concentrations,
                        phi_e_mV=knp_phi_e_mV,
                        cable_x_um=np.asarray(geometry["x_um"], dtype=float),
                        params=params,
                        initial_phi_reference_mV=initial_phi_reference_by_fiber[fid],
                    )
                    per_fiber_feedback_histories[fid].append(t_ms=t_ms, snapshot=active_feedback_by_fiber[fid])

    per_fiber_cable_results: Dict[int, Dict[str, np.ndarray]] = {
        fid: per_fiber_histories[fid].finalize(fiber_geometries[fid]) for fid in fiber_ids
    }

    vc_result = None
    if enable_vc and electrode_points is not None and len(electrode_points) > 0:
        per_fiber_vc_results = []
        for fid in fiber_ids:
            geometry = fiber_geometries[fid]
            per_fiber_result = cable_to_vc_traces(
                cable_result=per_fiber_cable_results[fid],
                geometry=geometry,
                electrode_points=electrode_points,
                params=params,
                cable_y_um=float(geometry.get("fiber_center_y_um", 0.0)),
                cable_z_um=float(geometry.get("fiber_center_z_um", 0.0)),
                material_model=material_model
            )
            per_fiber_vc_results.append(per_fiber_result)
        vc_result = _sum_vc_results(per_fiber_vc_results)
        bundle_state.shared_vc_state = {
            "per_fiber_vc_result": {fid: v for fid, v in zip(fiber_ids, per_fiber_vc_results)},
            "summed_vc_result": vc_result,
        }

    knp_result = None
    if enable_knp and knp_hist is not None and knp_domain is not None:
        knp_result = knp_hist.finalize(x_um=knp_domain.x_um)
        knp_result["source_debug"] = knp_source_debug
        knp_result["coupling_mode"] = str(getattr(params.knp, "coupling_mode", "physiologic"))
        knp_result["coupling_scale_mM_per_ms_per_uA"] = float(knp_coupling_scale_mM_per_ms_per_uA)
        knp_result["fiber_count"] = int(len(fiber_ids))
        if enable_knp and "shared_payload" in locals():
            knp_result["source_payload_metadata"] = dict(shared_payload["source_metadata"])

    bundle_history = bundle_hist.finalize()
    bundle_feedback_history = {
        fid: per_fiber_feedback_histories[fid].finalize() for fid in fiber_ids
    }

    # Bundle-level diagnostic summary.
    bundle_state.diagnostics_state = {
        "fiber_ids": fiber_ids,
        "resting_vm_mean_mV": {
            fid: float(np.mean(per_fiber_cable_results[fid]["V_m_mV"][0])) for fid in fiber_ids
        },
        "peak_vm_mV": {
            fid: float(np.max(per_fiber_cable_results[fid]["V_m_mV"])) for fid in fiber_ids
        },
    }

    metadata = {
    "preset_name": getattr(preset, "active_preset_name", "unspecified"),
    "dt_fast_ms": dt_fast_ms,
    "dt_slow_ms": dt_slow_ms,
    "enable_vc": bool(enable_vc),
    "enable_knp": bool(enable_knp),
    "feedback_enabled": bool(feedback_enabled),
    "feedback_policy_enabled": bool(getattr(params.policy, "use_dynamic_ecs_feedback", False)),
    "feedback_runtime_override": enable_ecs_feedback,
    "feedback_cable_support": bool(cable_accepts_feedback),
    "knp_species": knp_species,
    "slow_stride": slow_stride,
    "knp_coupling_mode": str(getattr(params.knp, "coupling_mode", "physiologic")),
    "knp_coupling_scale_mM_per_ms_per_uA": float(knp_coupling_scale_mM_per_ms_per_uA),

    "bundle_id": bundle_geometry.metadata.get("bundle_id", "bundle_0"),
    "bundle_layout_name": getattr(bundle_geometry, "layout_name", "unspecified"),
    "fiber_count": int(len(fiber_ids)),
    "fiber_ids": [int(fid) for fid in fiber_ids],
    "bundle_radius_um": float(getattr(bundle_geometry, "bundle_radius_um", 0.0)),
    "y_extent_um": tuple(getattr(bundle_geometry, "y_extent_um", (0.0, 0.0))),
    "z_extent_um": tuple(getattr(bundle_geometry, "z_extent_um", (0.0, 0.0))),

    "shared_ecs_enabled": bool(enable_knp),
    "shared_vc_enabled": bool(enable_vc),
    "vc_material_model_enabled": bool(material_model is not None),
    "per_fiber_feedback_history": bundle_feedback_history,
    "feedback_last_reversals_mV_by_fiber": {
        fid: dict(active_feedback_by_fiber[fid].reversal_overrides_mV)
        for fid in fiber_ids
    },
    "feedback_last_phi_stats_mV_by_fiber": {
        fid: {
            "mean": float(np.mean(active_feedback_by_fiber[fid].phi_e_cable_mV)),
            "min": float(np.min(active_feedback_by_fiber[fid].phi_e_cable_mV)),
            "max": float(np.max(active_feedback_by_fiber[fid].phi_e_cable_mV)),
        }
        for fid in fiber_ids
    },
    }

    return BundleMultirateSimulationResult(
        per_fiber_cable_results=per_fiber_cable_results,
        bundle_history=bundle_history,
        vc_result=vc_result,
        knp_result=knp_result,
        metadata=metadata,
    )