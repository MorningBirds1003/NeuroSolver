"""
cable_solver.py

Semi-implicit 1D cable solver for the fast propagation layer.

This version supports optional slow ECS feedback:
- extracellular potentials can offset effective membrane voltage,
- reversal potentials can be overridden,
- extracellular potential gradients can enter the axial solve.

Callers that do not pass ECS feedback still get the baseline behavior.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np

from ..params import (
    DEFAULT_PARAMS,
    SimulationParameters,
    axial_resistance_ohm,
    cylinder_lateral_area_cm2,
    summarize_reversal_potentials_mV,
)
from .gating import m_inf_tau_m_ms, h_inf_tau_h_ms, n_inf_tau_n_ms, step_all_gates
from .ion_channels import current_breakdown_uA_per_cm2
from ..stimuli import PulseTrainSpec, pulse_train

def build_uniform_fiber_geometry(
    params: SimulationParameters = DEFAULT_PARAMS,
    region_type: str = "node",
) -> Dict[str, np.ndarray | int]:
    """Build a simple uniform 1D chain of identical compartments."""
    n_compartments = int(params.topology.node_count)
    dx_um = float(params.geometry.segment_length_um)
    diameter_um = float(params.geometry.axon_diameter_um)

    area_cm2 = cylinder_lateral_area_cm2(dx_um, diameter_um)
    ra_interface_ohm = axial_resistance_ohm(
        length_um=dx_um,
        diameter_um=diameter_um,
        rho_i_ohm_cm=params.membrane.axial_resistivity_ohm_cm,
    )

    x_um = (np.arange(n_compartments, dtype=float) + 0.5) * dx_um
    region_labels = np.array([region_type] * n_compartments, dtype=object)

    Ra_left = np.zeros(n_compartments, dtype=float)
    Ra_right = np.zeros(n_compartments, dtype=float)
    for i in range(n_compartments - 1):
        Ra_right[i] = ra_interface_ohm
        Ra_left[i + 1] = ra_interface_ohm

    return {
        "n_compartments": n_compartments,
        "x_um": x_um,
        "dx_um": np.full(n_compartments, dx_um, dtype=float),
        "diameter_um": np.full(n_compartments, diameter_um, dtype=float),
        "area_cm2": np.full(n_compartments, area_cm2, dtype=float),
        "Ra_left_ohm": Ra_left,
        "Ra_right_ohm": Ra_right,
        "region_type": region_labels,
    }

def _default_stimulated_index(geometry: Dict[str, np.ndarray | int]) -> int:
    """Prefer the first node as the default stimulus site."""
    region_type = np.asarray(geometry["region_type"], dtype=object)
    node_indices = np.where(region_type == "node")[0]
    if node_indices.size > 0:
        return int(node_indices[0])
    return 0

def initialize_cable_state(
    geometry: Dict[str, np.ndarray | int],
    params: SimulationParameters = DEFAULT_PARAMS,
) -> Dict[str, np.ndarray]:
    """Initialize Vm and gates at steady state around the initial voltage."""
    n_compartments = int(geometry["n_compartments"])
    V0 = float(params.membrane.initial_voltage_mV)
    m0, _ = m_inf_tau_m_ms(V0, params.kinetics)
    h0, _ = h_inf_tau_h_ms(V0, params.kinetics)
    n0, _ = n_inf_tau_n_ms(V0, params.kinetics)
    zeros = np.zeros(n_compartments, dtype=float)

    return {
        "V_m_mV": np.full(n_compartments, V0, dtype=float),
        "m": np.full(n_compartments, m0, dtype=float),
        "h": np.full(n_compartments, h0, dtype=float),
        "n": np.full(n_compartments, n0, dtype=float),
        "phi_e_cable_mV": zeros.copy(),
        "V_membrane_effective_mV": np.full(n_compartments, V0, dtype=float),
    }

def stimulus_vector_uA(
    t_ms: float,
    geometry: Dict[str, np.ndarray | int],
    params: SimulationParameters = DEFAULT_PARAMS,
    stimulated_index: Optional[int] = None,
) -> np.ndarray:
    """
    Return an absolute-current stimulus vector [uA].

    Only one compartment is stimulated in the current implementation.
    """
    n_compartments = int(geometry["n_compartments"])
    I_app_abs = np.zeros(n_compartments, dtype=float)

    if stimulated_index is None:
        stimulated_index = _default_stimulated_index(geometry)

    t0 = float(params.stimulus.pulse_start_ms)
    width = float(params.stimulus.pulse_width_ms)
    amp_uA = float(getattr(params.stimulus, "pulse_amplitude_uA", 0.0))
    pulse_count = int(getattr(params.stimulus, "pulse_count", 1))
    interval_ms = float(getattr(params.stimulus, "pulse_interval_ms", width))

    if pulse_count > 1:
        amp_now = pulse_train(
            t_ms=t_ms,
            spec=PulseTrainSpec(
                start_ms=t0,
                width_ms=width,
                amplitude=amp_uA,
                pulse_count=pulse_count,
                interval_ms=interval_ms,
            ),
        )
    else:
        amp_now = amp_uA if (t0 <= t_ms < t0 + width) else 0.0

    if 0 <= stimulated_index < n_compartments:
        I_app_abs[stimulated_index] = float(amp_now)

    return I_app_abs

def stimulus_vector_uA_per_cm2(
    t_ms: float,
    geometry: Dict[str, np.ndarray | int],
    params: SimulationParameters = DEFAULT_PARAMS,
    stimulated_index: Optional[int] = None,
) -> np.ndarray:
    """Return a current-density stimulus vector [uA/cm^2]."""
    n_compartments = int(geometry["n_compartments"])
    I_app = np.zeros(n_compartments, dtype=float)

    if stimulated_index is None:
        stimulated_index = _default_stimulated_index(geometry)

    t0 = float(params.stimulus.pulse_start_ms)
    width = float(params.stimulus.pulse_width_ms)
    amp = float(params.stimulus.pulse_amplitude_uA_per_cm2)
    pulse_count = int(getattr(params.stimulus, "pulse_count", 1))
    interval_ms = float(getattr(params.stimulus, "pulse_interval_ms", width))

    if pulse_count > 1:
        amp_now = pulse_train(
            t_ms=t_ms,
            spec=PulseTrainSpec(
                start_ms=t0,
                width_ms=width,
                amplitude=amp,
                pulse_count=pulse_count,
                interval_ms=interval_ms,
            ),
        )
    else:
        amp_now = amp if (t0 <= t_ms < t0 + width) else 0.0

    if 0 <= stimulated_index < n_compartments:
        I_app[stimulated_index] = float(amp_now)

    return I_app

def _select_region_conductances(region_type: str, params: SimulationParameters) -> Dict[str, float]:
    """Select conductance densities for a named region."""
    if region_type == "node":
        region = params.regional_conductance.node
    elif region_type == "internode":
        region = params.regional_conductance.internode
    elif region_type == "paranode":
        region = params.regional_conductance.paranode
    elif region_type == "juxtaparanode":
        region = params.regional_conductance.juxtaparanode
    else:
        raise ValueError(f"Unsupported region_type '{region_type}'.")

    return {
        "gbar_na_mS_per_cm2": float(region.gbar_na_mS_per_cm2),
        "gbar_k_mS_per_cm2": float(region.gbar_k_mS_per_cm2),
        "gbar_l_mS_per_cm2": float(region.gbar_l_mS_per_cm2),
    }

def _select_region_membrane_properties(region_type: str, params: SimulationParameters) -> Dict[str, float]:
    """Select membrane properties for a named region."""
    if region_type == "node":
        region = params.regional_membrane.node
    elif region_type == "internode":
        region = params.regional_membrane.internode
    elif region_type == "paranode":
        region = params.regional_membrane.paranode
    elif region_type == "juxtaparanode":
        region = params.regional_membrane.juxtaparanode
    else:
        raise ValueError(f"Unsupported region_type '{region_type}'.")

    return {
        "membrane_capacitance_uF_per_cm2": float(region.membrane_capacitance_uF_per_cm2),
        "specific_membrane_resistance_ohm_cm2": float(region.specific_membrane_resistance_ohm_cm2),
    }

def _coerce_vector(values: object, n_compartments: int, default: float = 0.0) -> np.ndarray:
    """Coerce scalar/None/vector input into a length-n_compartments vector."""
    if values is None:
        return np.full(n_compartments, default, dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(n_compartments, float(arr), dtype=float)
    if arr.size == n_compartments:
        return arr.astype(float, copy=False)
    if arr.size == 1:
        return np.full(n_compartments, float(arr.reshape(-1)[0]), dtype=float)
    raise ValueError(f"Cannot coerce array of size {arr.size} into length {n_compartments}.")

def _select_reversal_potentials(
    params: SimulationParameters,
    n_compartments: int,
    ecs_feedback: Optional[Mapping[str, object]] = None,
) -> Dict[str, np.ndarray]:
    """
    Build compartment-wise reversal vectors.

    Uses either fixed preset reversals or Nernst-derived reversals, then applies
    optional ECS overrides.
    """
    use_nernst = bool(
        getattr(getattr(params, "policy", object()), "use_nernst_reversal_from_ion_pools", False)
    )

    if use_nernst:
        nernst = summarize_reversal_potentials_mV(params)
        reversals = {
            "E_na_mV": np.full(n_compartments, float(nernst["E_Na_mV"]), dtype=float),
            "E_k_mV": np.full(n_compartments, float(nernst["E_K_mV"]), dtype=float),
            "E_l_mV": np.full(n_compartments, float(params.membrane.leak_reversal_mV), dtype=float),
        }
    else:
        reversals = {
            "E_na_mV": np.full(n_compartments, float(params.membrane.sodium_reversal_mV), dtype=float),
            "E_k_mV": np.full(n_compartments, float(params.membrane.potassium_reversal_mV), dtype=float),
            "E_l_mV": np.full(n_compartments, float(params.membrane.leak_reversal_mV), dtype=float),
        }

    if ecs_feedback is None:
        return reversals

    overrides = ecs_feedback.get("reversal_overrides_mV", None)
    if isinstance(overrides, Mapping):
        for key in ("E_na_mV", "E_k_mV", "E_l_mV"):
            if key in overrides:
                reversals[key] = _coerce_vector(overrides[key], n_compartments, default=float(reversals[key][0]))
    return reversals

def _thomas_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal linear system with the Thomas algorithm."""
    n = diag.size
    if n == 0:
        return np.array([], dtype=float)

    c_prime = np.zeros(max(n - 1, 0), dtype=float)
    d_prime = np.zeros(n, dtype=float)

    denom = float(diag[0])
    if abs(denom) < 1.0e-15:
        raise ZeroDivisionError("Singular tridiagonal system at first diagonal entry.")

    if n > 1:
        c_prime[0] = float(upper[0]) / denom
    d_prime[0] = float(rhs[0]) / denom

    for i in range(1, n):
        denom = float(diag[i]) - float(lower[i - 1]) * c_prime[i - 1]
        if abs(denom) < 1.0e-15:
            raise ZeroDivisionError(f"Singular tridiagonal system at index {i}.")
        if i < n - 1:
            c_prime[i] = float(upper[i]) / denom
        d_prime[i] = (float(rhs[i]) - float(lower[i - 1]) * d_prime[i - 1]) / denom

    x = np.zeros(n, dtype=float)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x

def _compute_explicit_terms(
    state: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray | int],
    t_ms: float,
    dt_ms: float,
    params: SimulationParameters,
    stimulated_index: Optional[int] = None,
    ecs_feedback: Optional[Mapping[str, object]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute explicit terms for one time step.

    This updates gates, builds current terms, and resolves the effective
    membrane voltage seen by the channel kinetics.
    """
    V = np.asarray(state["V_m_mV"], dtype=float)
    m = np.asarray(state["m"], dtype=float)
    h = np.asarray(state["h"], dtype=float)
    n_gate = np.asarray(state["n"], dtype=float)
    region_type = np.asarray(geometry["region_type"], dtype=object)
    area_cm2 = np.asarray(geometry["area_cm2"], dtype=float)

    n_compartments = V.size
    phi_e_cable_mV = _coerce_vector(
        None if ecs_feedback is None else ecs_feedback.get("phi_e_cable_mV", None),
        n_compartments,
        default=0.0,
    )
    V_eff = V - phi_e_cable_mV
    reversals = _select_reversal_potentials(params, n_compartments=n_compartments, ecs_feedback=ecs_feedback)

    m_next = np.empty_like(m)
    h_next = np.empty_like(h)
    n_next = np.empty_like(n_gate)
    I_ion_density = np.empty_like(V)
    C_density = np.empty_like(V)
    I_na_density = np.empty_like(V)
    I_k_density = np.empty_like(V)
    I_l_density = np.empty_like(V)

    # Prefer absolute-current stimulation if it is active; otherwise fall back to density form.
    I_app_abs_direct = stimulus_vector_uA(
        t_ms=t_ms,
        geometry=geometry,
        params=params,
        stimulated_index=stimulated_index,
    )
    use_abs_stim = bool(np.any(np.abs(I_app_abs_direct) > 0.0))

    if use_abs_stim:
        I_app_abs_uA = I_app_abs_direct
        I_app_density = np.zeros(n_compartments, dtype=float)
        nonzero = area_cm2 > 0.0
        I_app_density[nonzero] = I_app_abs_uA[nonzero] / area_cm2[nonzero]
    else:
        I_app_density = stimulus_vector_uA_per_cm2(
            t_ms=t_ms,
            geometry=geometry,
            params=params,
            stimulated_index=stimulated_index,
        )
        I_app_abs_uA = I_app_density * area_cm2

    for i in range(n_compartments):
        label = str(region_type[i])
        gate_update = step_all_gates(
            V_m_mV=float(V_eff[i]),
            m=float(m[i]),
            h=float(h[i]),
            n=float(n_gate[i]),
            dt_ms=float(dt_ms),
            settings=params.kinetics,
        )
        m_next[i] = float(gate_update["m"])
        h_next[i] = float(gate_update["h"])
        n_next[i] = float(gate_update["n"])

        mem = _select_region_membrane_properties(label, params)
        gbar = _select_region_conductances(label, params)
        C_density[i] = mem["membrane_capacitance_uF_per_cm2"]

        currents = current_breakdown_uA_per_cm2(
            V_m_mV=float(V_eff[i]),
            m=float(m[i]),
            h=float(h[i]),
            n=float(n_gate[i]),
            E_na_mV=float(reversals["E_na_mV"][i]),
            E_k_mV=float(reversals["E_k_mV"][i]),
            E_l_mV=float(reversals["E_l_mV"][i]),
            gbar_na_mS_per_cm2=gbar["gbar_na_mS_per_cm2"],
            gbar_k_mS_per_cm2=gbar["gbar_k_mS_per_cm2"],
            gbar_l_mS_per_cm2=gbar["gbar_l_mS_per_cm2"],
        )
        I_na_density[i] = float(currents["I_Na_uA_per_cm2"])
        I_k_density[i] = float(currents["I_K_uA_per_cm2"])
        I_l_density[i] = float(currents["I_L_uA_per_cm2"])
        I_ion_density[i] = float(currents["I_total_uA_per_cm2"])

    C_abs_uF = C_density * area_cm2
    I_ion_abs_uA = I_ion_density * area_cm2

    return {
        "m_next": m_next,
        "h_next": h_next,
        "n_next": n_next,
        "C_uF_per_cm2": C_density,
        "C_abs_uF": C_abs_uF,
        "I_ion_uA_per_cm2": I_ion_density,
        "I_ion_abs_uA": I_ion_abs_uA,
        "I_app_uA_per_cm2": I_app_density,
        "I_app_abs_uA": I_app_abs_uA,
        "phi_e_cable_mV": phi_e_cable_mV,
        "V_membrane_effective_mV": V_eff,
        "E_na_mV": reversals["E_na_mV"],
        "E_k_mV": reversals["E_k_mV"],
        "E_l_mV": reversals["E_l_mV"],
        "species_current_density_uA_per_cm2": {
            "Na": I_na_density,
            "K": I_k_density,
        },
        "nonspecific_current_density_uA_per_cm2": {
            "L": I_l_density,
        },
    }

def _build_implicit_voltage_system(
    V_old_mV: np.ndarray,
    geometry: Dict[str, np.ndarray | int],
    C_abs_uF: np.ndarray,
    I_ion_abs_uA: np.ndarray,
    I_app_abs_uA: np.ndarray,
    dt_ms: float,
    phi_e_cable_mV: Optional[np.ndarray] = None,
    sealed_ends: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the tridiagonal system for the semi-implicit voltage update.

    Extracellular potential differences enter the right-hand side as an axial
    driving term.
    """
    V_old_mV = np.asarray(V_old_mV, dtype=float)
    C_abs_uF = np.asarray(C_abs_uF, dtype=float)
    I_ion_abs_uA = np.asarray(I_ion_abs_uA, dtype=float)
    I_app_abs_uA = np.asarray(I_app_abs_uA, dtype=float)
    Ra_left = np.asarray(geometry["Ra_left_ohm"], dtype=float)
    Ra_right = np.asarray(geometry["Ra_right_ohm"], dtype=float)

    n = V_old_mV.size
    phi = np.zeros(n, dtype=float) if phi_e_cable_mV is None else np.asarray(phi_e_cable_mV, dtype=float)

    lower = np.zeros(n - 1, dtype=float)
    diag = np.zeros(n, dtype=float)
    upper = np.zeros(n - 1, dtype=float)
    rhs = np.zeros(n, dtype=float)

    for i in range(n):
        C_i = float(C_abs_uF[i])
        if C_i <= 0.0:
            raise ValueError(f"Non-positive absolute membrane capacitance at compartment {i}.")

        center_coeff = C_i / float(dt_ms)
        rhs_i = (C_i / float(dt_ms)) * float(V_old_mV[i]) + float(I_app_abs_uA[i]) - float(I_ion_abs_uA[i])

        if i > 0:
            if Ra_left[i] <= 0.0:
                raise ValueError(f"Non-positive left axial resistance at compartment {i}.")
            g_left = 1000.0 / float(Ra_left[i])
            center_coeff += g_left
            lower[i - 1] = -g_left
            rhs_i += g_left * (float(phi[i]) - float(phi[i - 1]))

        if i < n - 1:
            if Ra_right[i] <= 0.0:
                raise ValueError(f"Non-positive right axial resistance at compartment {i}.")
            g_right = 1000.0 / float(Ra_right[i])
            center_coeff += g_right
            upper[i] = -g_right
            rhs_i += g_right * (float(phi[i]) - float(phi[i + 1]))

        diag[i] = center_coeff
        rhs[i] = rhs_i

    return lower, diag, upper, rhs

def advance_cable_state(
    state: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray | int],
    t_ms: float,
    dt_ms: float,
    params: SimulationParameters = DEFAULT_PARAMS,
    stimulated_index: Optional[int] = None,
    ecs_feedback: Optional[Mapping[str, object]] = None,
) -> Dict[str, np.ndarray]:
    """
    Advance the cable by one fast time step.

    Gates are updated explicitly; Vm is advanced with the semi-implicit axial solve.
    """
    explicit = _compute_explicit_terms(
        state=state,
        geometry=geometry,
        t_ms=t_ms,
        dt_ms=dt_ms,
        params=params,
        stimulated_index=stimulated_index,
        ecs_feedback=ecs_feedback,
    )

    lower, diag, upper, rhs = _build_implicit_voltage_system(
        V_old_mV=np.asarray(state["V_m_mV"], dtype=float),
        geometry=geometry,
        C_abs_uF=explicit["C_abs_uF"],
        I_ion_abs_uA=explicit["I_ion_abs_uA"],
        I_app_abs_uA=explicit["I_app_abs_uA"],
        dt_ms=float(dt_ms),
        phi_e_cable_mV=explicit.get("phi_e_cable_mV", None),
        sealed_ends=bool(params.topology.sealed_end_boundary),
    )

    V_next = _thomas_solve(lower, diag, upper, rhs)
    phi_e = np.asarray(explicit.get("phi_e_cable_mV", np.zeros_like(V_next)), dtype=float)
    V_eff_next = V_next - phi_e

    return {
        "V_m_mV": V_next,
        "m": explicit["m_next"],
        "h": explicit["h_next"],
        "n": explicit["n_next"],
        "I_ion_uA_per_cm2": explicit["I_ion_uA_per_cm2"],
        "I_app_uA_per_cm2": explicit["I_app_uA_per_cm2"],
        "I_app_abs_uA": explicit["I_app_abs_uA"],
        "phi_e_cable_mV": phi_e,
        "V_membrane_effective_mV": V_eff_next,
        "E_na_mV": explicit["E_na_mV"],
        "E_k_mV": explicit["E_k_mV"],
        "E_l_mV": explicit["E_l_mV"],
        "species_current_density_uA_per_cm2": explicit["species_current_density_uA_per_cm2"],
        "nonspecific_current_density_uA_per_cm2": explicit["nonspecific_current_density_uA_per_cm2"],
    }

def run_cable_pulse_test(
    params: SimulationParameters = DEFAULT_PARAMS,
    geometry: Optional[Dict[str, np.ndarray | int]] = None,
    stimulated_index: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Run a baseline cable pulse test and return full time histories."""
    if geometry is None:
        geometry = build_uniform_fiber_geometry(params=params, region_type="node")

    dt_ms = float(params.solver.dt_fast_ms)
    t_stop_ms = float(params.solver.t_stop_ms)
    n_steps = int(round(t_stop_ms / dt_ms))

    state = initialize_cable_state(geometry, params)

    n_compartments = int(geometry["n_compartments"])
    zero_vec = np.zeros(n_compartments, dtype=float)

    t_history: List[float] = []
    V_history: List[np.ndarray] = []
    V_eff_history: List[np.ndarray] = []
    phi_history: List[np.ndarray] = []
    I_ion_history: List[np.ndarray] = []
    I_app_history: List[np.ndarray] = []
    I_app_abs_history: List[np.ndarray] = []

    for step in range(n_steps + 1):
        t_ms = step * dt_ms

        # Record state before the next advance.
        t_history.append(t_ms)
        V_history.append(np.asarray(state["V_m_mV"], dtype=float).copy())
        V_eff_history.append(np.asarray(state.get("V_membrane_effective_mV", state["V_m_mV"]), dtype=float).copy())
        phi_history.append(np.asarray(state.get("phi_e_cable_mV", zero_vec), dtype=float).copy())
        I_ion_history.append(np.asarray(state.get("I_ion_uA_per_cm2", zero_vec), dtype=float).copy())
        I_app_history.append(np.asarray(state.get("I_app_uA_per_cm2", zero_vec), dtype=float).copy())
        I_app_abs_history.append(np.asarray(state.get("I_app_abs_uA", zero_vec), dtype=float).copy())

        state = advance_cable_state(
            state=state,
            geometry=geometry,
            t_ms=t_ms,
            dt_ms=dt_ms,
            params=params,
            stimulated_index=stimulated_index,
            ecs_feedback=None,
        )

    return {
        "t_ms": np.asarray(t_history, dtype=float),
        "V_m_mV": np.asarray(V_history, dtype=float),
        "V_membrane_effective_mV": np.asarray(V_eff_history, dtype=float),
        "phi_e_cable_mV": np.asarray(phi_history, dtype=float),
        "x_um": np.asarray(geometry["x_um"], dtype=float).copy(),
        "I_ion_uA_per_cm2": np.asarray(I_ion_history, dtype=float),
        "I_app_uA_per_cm2": np.asarray(I_app_history, dtype=float),
        "I_app_abs_uA": np.asarray(I_app_abs_history, dtype=float),
        "region_type": np.asarray(geometry["region_type"], dtype=object).copy(),
    }
