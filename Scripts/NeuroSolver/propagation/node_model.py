"""
node_model.py

Single-compartment active membrane model for the fast membrane layer.

This keeps the ECS-feedback hook intact and uses preset-driven region property
selection for conductances and capacitance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from ..params import DEFAULT_PARAMS, SimulationParameters, summarize_reversal_potentials_mV
from .gating import (
    m_inf_tau_m_ms,
    h_inf_tau_h_ms,
    n_inf_tau_n_ms,
    step_all_gates,
)
from .ion_channels import (
    current_breakdown_uA_per_cm2,
    dVdt_mV_per_ms,
)


@dataclass
class NodeState:
    """State of one active compartment."""
    V_m_mV: float
    m: float
    h: float
    n: float


def _select_reversal_potentials(
    params: SimulationParameters,
    ecs_feedback: Optional[Mapping[str, object]] = None,
) -> Dict[str, float]:
    """Resolve reversal potentials, then apply optional ECS overrides."""
    if bool(getattr(params.policy, "use_nernst_reversal_from_ion_pools", False)):
        nernst = summarize_reversal_potentials_mV(params)
        reversals = {
            "E_na_mV": float(nernst["E_Na_mV"]),
            "E_k_mV": float(nernst["E_K_mV"]),
            "E_l_mV": float(params.membrane.leak_reversal_mV),
        }
    else:
        reversals = {
            "E_na_mV": float(params.membrane.sodium_reversal_mV),
            "E_k_mV": float(params.membrane.potassium_reversal_mV),
            "E_l_mV": float(params.membrane.leak_reversal_mV),
        }

    if ecs_feedback is None:
        return reversals

    overrides = ecs_feedback.get("reversal_overrides_mV", None)
    if isinstance(overrides, Mapping):
        for key in ("E_na_mV", "E_k_mV", "E_l_mV"):
            if key in overrides:
                reversals[key] = float(overrides[key])

    return reversals


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


def initialize_node_state(
    params: SimulationParameters = DEFAULT_PARAMS,
    V_init_mV: Optional[float] = None,
) -> NodeState:
    """Initialize the node at steady-state gate values for the chosen voltage."""
    V0 = params.membrane.initial_voltage_mV if V_init_mV is None else float(V_init_mV)
    m_inf, _ = m_inf_tau_m_ms(V0, params.kinetics)
    h_inf, _ = h_inf_tau_h_ms(V0, params.kinetics)
    n_inf, _ = n_inf_tau_n_ms(V0, params.kinetics)
    return NodeState(V_m_mV=V0, m=m_inf, h=h_inf, n=n_inf)


def default_injected_current_uA_per_cm2(
    t_ms: float,
    params: SimulationParameters = DEFAULT_PARAMS,
) -> float:
    """Default rectangular injected-current pulse."""
    t0 = params.stimulus.pulse_start_ms
    width = params.stimulus.pulse_width_ms
    amp = params.stimulus.pulse_amplitude_uA_per_cm2
    return amp if (t0 <= t_ms < t0 + width) else 0.0


def compute_node_current_breakdown(
    state: NodeState,
    region_type: str = "node",
    params: SimulationParameters = DEFAULT_PARAMS,
    ecs_feedback: Optional[Mapping[str, object]] = None,
) -> Dict[str, float]:
    """
    Compute current components for one node state.

    ECS feedback enters as an extracellular potential offset and/or reversal override.
    """
    phi_e_mV = float((ecs_feedback or {}).get("phi_e_mV", 0.0) or 0.0)
    V_eff_mV = float(state.V_m_mV) - phi_e_mV

    reversals = _select_reversal_potentials(params, ecs_feedback=ecs_feedback)
    gbar = _select_region_conductances(region_type, params)

    return current_breakdown_uA_per_cm2(
        V_m_mV=V_eff_mV,
        m=state.m,
        h=state.h,
        n=state.n,
        E_na_mV=reversals["E_na_mV"],
        E_k_mV=reversals["E_k_mV"],
        E_l_mV=reversals["E_l_mV"],
        gbar_na_mS_per_cm2=gbar["gbar_na_mS_per_cm2"],
        gbar_k_mS_per_cm2=gbar["gbar_k_mS_per_cm2"],
        gbar_l_mS_per_cm2=gbar["gbar_l_mS_per_cm2"],
    )


def advance_node_state(
    state: NodeState,
    dt_ms: float,
    I_applied_uA_per_cm2: float = 0.0,
    I_axial_uA_per_cm2: float = 0.0,
    I_extracellular_uA_per_cm2: float = 0.0,
    region_type: str = "node",
    params: SimulationParameters = DEFAULT_PARAMS,
    ecs_feedback: Optional[Mapping[str, object]] = None,
) -> NodeState:
    """
    Advance one node state by one fast time step.

    Gates are updated first, then Vm is advanced using the updated gate values.
    """
    phi_e_mV = float((ecs_feedback or {}).get("phi_e_mV", 0.0) or 0.0)
    V_eff_mV = float(state.V_m_mV) - phi_e_mV

    reversals = _select_reversal_potentials(params, ecs_feedback=ecs_feedback)
    gate_update = step_all_gates(
        V_m_mV=V_eff_mV,
        m=float(state.m),
        h=float(state.h),
        n=float(state.n),
        dt_ms=float(dt_ms),
        settings=params.kinetics,
    )

    mem = _select_region_membrane_properties(region_type, params)
    gbar = _select_region_conductances(region_type, params)

    dVdt = dVdt_mV_per_ms(
        V_m_mV=V_eff_mV,
        m=float(gate_update["m"]),
        h=float(gate_update["h"]),
        n=float(gate_update["n"]),
        I_applied_uA_per_cm2=float(I_applied_uA_per_cm2),
        I_axial_uA_per_cm2=float(I_axial_uA_per_cm2),
        I_extracellular_uA_per_cm2=float(I_extracellular_uA_per_cm2),
        membrane_capacitance_uF_per_cm2=float(mem["membrane_capacitance_uF_per_cm2"]),
        E_na_mV=float(reversals["E_na_mV"]),
        E_k_mV=float(reversals["E_k_mV"]),
        E_l_mV=float(reversals["E_l_mV"]),
        gbar_na_mS_per_cm2=float(gbar["gbar_na_mS_per_cm2"]),
        gbar_k_mS_per_cm2=float(gbar["gbar_k_mS_per_cm2"]),
        gbar_l_mS_per_cm2=float(gbar["gbar_l_mS_per_cm2"]),
    )

    return NodeState(
        V_m_mV=float(state.V_m_mV) + float(dt_ms) * float(dVdt),
        m=float(gate_update["m"]),
        h=float(gate_update["h"]),
        n=float(gate_update["n"]),
    )


def run_single_node_pulse_test(
    params: SimulationParameters = DEFAULT_PARAMS,
    region_type: str = "node",
) -> List[Dict[str, float]]:
    """Run a simple single-node pulse test and return the time trace."""
    dt_ms = params.solver.dt_fast_ms
    t_stop_ms = params.solver.t_stop_ms
    n_steps = int(round(t_stop_ms / dt_ms))

    state = initialize_node_state(params)
    trace: List[Dict[str, float]] = []

    for step in range(n_steps + 1):
        t_ms = step * dt_ms
        I_app = default_injected_current_uA_per_cm2(t_ms, params)
        currents = compute_node_current_breakdown(state, region_type, params)

        trace.append(
            {
                "t_ms": t_ms,
                "V_m_mV": state.V_m_mV,
                "m": state.m,
                "h": state.h,
                "n": state.n,
                "I_app_uA_per_cm2": I_app,
                "I_total_uA_per_cm2": currents["I_total_uA_per_cm2"],
            }
        )

        state = advance_node_state(
            state=state,
            dt_ms=dt_ms,
            I_applied_uA_per_cm2=I_app,
            region_type=region_type,
            params=params,
        )

    return trace
