"""
ion_channels.py

Ionic current equations for the fast membrane solver.

Separation of roles:
- gating.py handles gate dynamics,
- this file turns gates into conductances and currents,
- node_model.py / cable_solver.py assemble those currents into Vm updates.
"""

from __future__ import annotations

from typing import Dict

from ..params import DEFAULT_PARAMS


def sodium_conductance_mS_per_cm2(
    m: float,
    h: float,
    gbar_na_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_na_mS_per_cm2,
) -> float:
    """Instantaneous sodium conductance density."""
    return gbar_na_mS_per_cm2 * (m ** 3) * h


def potassium_conductance_mS_per_cm2(
    n: float,
    gbar_k_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_k_mS_per_cm2,
) -> float:
    """Instantaneous delayed-rectifier potassium conductance density."""
    return gbar_k_mS_per_cm2 * (n ** 4)


def leak_conductance_mS_per_cm2(
    gbar_l_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_l_mS_per_cm2,
) -> float:
    """Constant leak conductance density."""
    return gbar_l_mS_per_cm2


def sodium_current_uA_per_cm2(
    V_m_mV: float,
    m: float,
    h: float,
    E_na_mV: float = DEFAULT_PARAMS.membrane.sodium_reversal_mV,
    gbar_na_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_na_mS_per_cm2,
) -> float:
    """Sodium current density [uA/cm^2]."""
    g_na = sodium_conductance_mS_per_cm2(m=m, h=h, gbar_na_mS_per_cm2=gbar_na_mS_per_cm2)
    return g_na * (V_m_mV - E_na_mV)


def potassium_current_uA_per_cm2(
    V_m_mV: float,
    n: float,
    E_k_mV: float = DEFAULT_PARAMS.membrane.potassium_reversal_mV,
    gbar_k_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_k_mS_per_cm2,
) -> float:
    """Potassium current density [uA/cm^2]."""
    g_k = potassium_conductance_mS_per_cm2(n=n, gbar_k_mS_per_cm2=gbar_k_mS_per_cm2)
    return g_k * (V_m_mV - E_k_mV)


def leak_current_uA_per_cm2(
    V_m_mV: float,
    E_l_mV: float = DEFAULT_PARAMS.membrane.leak_reversal_mV,
    gbar_l_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_l_mS_per_cm2,
) -> float:
    """Leak current density [uA/cm^2]."""
    g_l = leak_conductance_mS_per_cm2(gbar_l_mS_per_cm2=gbar_l_mS_per_cm2)
    return g_l * (V_m_mV - E_l_mV)


def total_ionic_current_uA_per_cm2(
    V_m_mV: float,
    m: float,
    h: float,
    n: float,
    E_na_mV: float = DEFAULT_PARAMS.membrane.sodium_reversal_mV,
    E_k_mV: float = DEFAULT_PARAMS.membrane.potassium_reversal_mV,
    E_l_mV: float = DEFAULT_PARAMS.membrane.leak_reversal_mV,
    gbar_na_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_na_mS_per_cm2,
    gbar_k_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_k_mS_per_cm2,
    gbar_l_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_l_mS_per_cm2,
) -> float:
    """Sum sodium, potassium, and leak current densities."""
    i_na = sodium_current_uA_per_cm2(V_m_mV, m, h, E_na_mV, gbar_na_mS_per_cm2)
    i_k = potassium_current_uA_per_cm2(V_m_mV, n, E_k_mV, gbar_k_mS_per_cm2)
    i_l = leak_current_uA_per_cm2(V_m_mV, E_l_mV, gbar_l_mS_per_cm2)
    return i_na + i_k + i_l


def current_breakdown_uA_per_cm2(
    V_m_mV: float,
    m: float,
    h: float,
    n: float,
    E_na_mV: float = DEFAULT_PARAMS.membrane.sodium_reversal_mV,
    E_k_mV: float = DEFAULT_PARAMS.membrane.potassium_reversal_mV,
    E_l_mV: float = DEFAULT_PARAMS.membrane.leak_reversal_mV,
    gbar_na_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_na_mS_per_cm2,
    gbar_k_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_k_mS_per_cm2,
    gbar_l_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_l_mS_per_cm2,
) -> Dict[str, float | Dict[str, float]]:
    """
    Return labeled current components.

    Na and K are tracked as species currents. Leak stays separate because it is
    not currently tied to a specific ionic channel species.
    """
    i_na = sodium_current_uA_per_cm2(V_m_mV, m, h, E_na_mV, gbar_na_mS_per_cm2)
    i_k = potassium_current_uA_per_cm2(V_m_mV, n, E_k_mV, gbar_k_mS_per_cm2)
    i_l = leak_current_uA_per_cm2(V_m_mV, E_l_mV, gbar_l_mS_per_cm2)
    i_total = i_na + i_k + i_l

    return {
        "I_Na_uA_per_cm2": i_na,
        "I_K_uA_per_cm2": i_k,
        "I_L_uA_per_cm2": i_l,
        "I_total_uA_per_cm2": i_total,
        "species_current_density_uA_per_cm2": {
            "Na": i_na,
            "K": i_k,
        },
        "nonspecific_current_density_uA_per_cm2": {
            "L": i_l,
        },
    }


def dVdt_mV_per_ms(
    V_m_mV: float,
    m: float,
    h: float,
    n: float,
    I_applied_uA_per_cm2: float = 0.0,
    I_axial_uA_per_cm2: float = 0.0,
    I_extracellular_uA_per_cm2: float = 0.0,
    membrane_capacitance_uF_per_cm2: float = DEFAULT_PARAMS.membrane.membrane_capacitance_uF_per_cm2,
    E_na_mV: float = DEFAULT_PARAMS.membrane.sodium_reversal_mV,
    E_k_mV: float = DEFAULT_PARAMS.membrane.potassium_reversal_mV,
    E_l_mV: float = DEFAULT_PARAMS.membrane.leak_reversal_mV,
    gbar_na_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_na_mS_per_cm2,
    gbar_k_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_k_mS_per_cm2,
    gbar_l_mS_per_cm2: float = DEFAULT_PARAMS.conductance.gbar_l_mS_per_cm2,
) -> float:
    """
    Compute dV/dt for one active compartment.

    Sign convention:
    positive ionic current is outward; applied/axial/extracellular inputs enter
    as inward drive in the membrane-balance equation.
    """
    I_ion = total_ionic_current_uA_per_cm2(
        V_m_mV=V_m_mV,
        m=m,
        h=h,
        n=n,
        E_na_mV=E_na_mV,
        E_k_mV=E_k_mV,
        E_l_mV=E_l_mV,
        gbar_na_mS_per_cm2=gbar_na_mS_per_cm2,
        gbar_k_mS_per_cm2=gbar_k_mS_per_cm2,
        gbar_l_mS_per_cm2=gbar_l_mS_per_cm2,
    )

    net_inward_drive_uA_per_cm2 = I_applied_uA_per_cm2 + I_axial_uA_per_cm2 + I_extracellular_uA_per_cm2
    return (-I_ion + net_inward_drive_uA_per_cm2) / membrane_capacitance_uF_per_cm2
