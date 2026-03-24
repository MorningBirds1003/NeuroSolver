"""
gating.py

Voltage-gated state-variable kinetics for the fast membrane solver.

This module only handles gate mathematics and gate updates.
Current formulas live in ion_channels.py.
"""

from __future__ import annotations

from typing import Dict, Tuple
import math

from ..params import DEFAULT_PARAMS, GateKineticsSettings, q10_scale_factor


def _safe_exp(x: float) -> float:
    """Exponent with clipping to avoid floating-point overflow."""
    x_clipped = max(min(float(x), 700.0), -700.0)
    return math.exp(x_clipped)


def _clip_gate(value: float) -> float:
    """Clamp a gate value to the physical range [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _clamp_voltage_for_rates(V_m_mV: float, settings: GateKineticsSettings) -> float:
    """
    Apply voltage shift and clamp the voltage used in the rate laws.

    The clamp is intentionally broad: physiological values pass unchanged, but
    pathological inputs do not destabilize exp-based rate expressions.
    """
    V = float(V_m_mV) + float(settings.voltage_shift_mV)
    return max(min(V, 150.0), -200.0)


def _vtrap(numerator: float, denominator_argument: float) -> float:
    """
    Stable evaluation of numerator / (exp(x) - 1).

    This avoids cancellation near x=0 and poor behavior at extreme x.
    """
    x = float(denominator_argument)
    num = float(numerator)

    if abs(x) < 1.0e-6:
        return num if x == 0.0 else num / x
    if x > 50.0:
        return 0.0
    if x < -50.0:
        return -num

    return num / (_safe_exp(x) - 1.0)


def _temperature_rate_scale(settings: GateKineticsSettings) -> float:
    """Return the Q10-based rate scaling factor, or 1 if disabled."""
    if not bool(getattr(settings, "use_temperature_scaling", False)):
        return 1.0
    temperature_celsius = float(getattr(settings, "temperature_celsius", 37.0))
    reference_celsius = float(getattr(settings, "reference_celsius", 20.0))
    q10 = float(getattr(settings, "q10", 2.5))
    return q10_scale_factor(
        temperature_celsius=temperature_celsius,
        reference_celsius=reference_celsius,
        q10=q10,
    )


def alpha_m_per_ms(
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> float:
    """Sodium activation alpha_m rate [1/ms]."""
    V = _clamp_voltage_for_rates(V_m_mV, settings)
    return _temperature_rate_scale(settings) * 0.1 * _vtrap(-(V + 40.0), -(V + 40.0) / 10.0)


def beta_m_per_ms(
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> float:
    """Sodium activation beta_m rate [1/ms]."""
    V = _clamp_voltage_for_rates(V_m_mV, settings)
    return _temperature_rate_scale(settings) * 4.0 * _safe_exp(-(V + 65.0) / 18.0)


def alpha_h_per_ms(
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> float:
    """Sodium inactivation alpha_h rate [1/ms]."""
    V = _clamp_voltage_for_rates(V_m_mV, settings)
    return _temperature_rate_scale(settings) * 0.07 * _safe_exp(-(V + 65.0) / 20.0)


def beta_h_per_ms(
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> float:
    """Sodium inactivation beta_h rate [1/ms]."""
    V = _clamp_voltage_for_rates(V_m_mV, settings)
    return _temperature_rate_scale(settings) * (1.0 / (1.0 + _safe_exp(-(V + 35.0) / 10.0)))


def alpha_n_per_ms(
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> float:
    """Potassium activation alpha_n rate [1/ms]."""
    V = _clamp_voltage_for_rates(V_m_mV, settings)
    return _temperature_rate_scale(settings) * 0.01 * _vtrap(-(V + 55.0), -(V + 55.0) / 10.0)


def beta_n_per_ms(
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> float:
    """Potassium activation beta_n rate [1/ms]."""
    V = _clamp_voltage_for_rates(V_m_mV, settings)
    return _temperature_rate_scale(settings) * 0.125 * _safe_exp(-(V + 65.0) / 80.0)


def gate_inf_and_tau(
    alpha_per_ms: float,
    beta_per_ms: float,
    minimum_time_constant_ms: float = DEFAULT_PARAMS.kinetics.minimum_time_constant_ms,
) -> Tuple[float, float]:
    """Convert alpha/beta rates into (x_inf, tau_x)."""
    rate_sum = float(alpha_per_ms) + float(beta_per_ms)
    if rate_sum <= 0.0:
        return 0.0, max(float(minimum_time_constant_ms), 1.0)

    x_inf = float(alpha_per_ms) / rate_sum
    tau_ms = max(1.0 / rate_sum, float(minimum_time_constant_ms))
    return x_inf, tau_ms


def m_inf_tau_m_ms(
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> Tuple[float, float]:
    """Return sodium activation steady state and time constant."""
    return gate_inf_and_tau(
        alpha_m_per_ms(V_m_mV, settings),
        beta_m_per_ms(V_m_mV, settings),
        settings.minimum_time_constant_ms,
    )


def h_inf_tau_h_ms(
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> Tuple[float, float]:
    """Return sodium inactivation steady state and time constant."""
    return gate_inf_and_tau(
        alpha_h_per_ms(V_m_mV, settings),
        beta_h_per_ms(V_m_mV, settings),
        settings.minimum_time_constant_ms,
    )


def n_inf_tau_n_ms(
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> Tuple[float, float]:
    """Return potassium activation steady state and time constant."""
    return gate_inf_and_tau(
        alpha_n_per_ms(V_m_mV, settings),
        beta_n_per_ms(V_m_mV, settings),
        settings.minimum_time_constant_ms,
    )


def dm_dt_per_ms(
    m: float,
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> float:
    """Differential-form derivative for m."""
    a = alpha_m_per_ms(V_m_mV, settings)
    b = beta_m_per_ms(V_m_mV, settings)
    return a * (1.0 - float(m)) - b * float(m)


def dh_dt_per_ms(
    h: float,
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> float:
    """Differential-form derivative for h."""
    a = alpha_h_per_ms(V_m_mV, settings)
    b = beta_h_per_ms(V_m_mV, settings)
    return a * (1.0 - float(h)) - b * float(h)


def dn_dt_per_ms(
    n: float,
    V_m_mV: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> float:
    """Differential-form derivative for n."""
    a = alpha_n_per_ms(V_m_mV, settings)
    b = beta_n_per_ms(V_m_mV, settings)
    return a * (1.0 - float(n)) - b * float(n)


def euler_step_gate(
    gate_value: float,
    dgate_dt_per_ms: float,
    dt_ms: float,
    clamp_to_unit_interval: bool = True,
) -> float:
    """Explicit Euler update for a gate variable."""
    updated = float(gate_value) + float(dt_ms) * float(dgate_dt_per_ms)
    return _clip_gate(updated) if clamp_to_unit_interval else updated


def rush_larsen_step_gate(
    gate_value: float,
    x_inf: float,
    tau_ms: float,
    dt_ms: float,
    clamp_to_unit_interval: bool = True,
) -> float:
    """Rush-Larsen update for first-order gate kinetics."""
    if tau_ms <= 0.0:
        updated = float(x_inf)
    else:
        updated = float(x_inf) + (float(gate_value) - float(x_inf)) * _safe_exp(-float(dt_ms) / float(tau_ms))
    return _clip_gate(updated) if clamp_to_unit_interval else updated


def step_all_gates(
    V_m_mV: float,
    m: float,
    h: float,
    n: float,
    dt_ms: float,
    settings: GateKineticsSettings = DEFAULT_PARAMS.kinetics,
) -> Dict[str, float]:
    """
    Advance m, h, and n for one membrane update.

    Rush-Larsen is preferred for stability; Euler remains available for simpler
    debugging or baseline tests.
    """
    if settings.use_rush_larsen:
        m_inf, tau_m_ms = m_inf_tau_m_ms(V_m_mV, settings)
        h_inf, tau_h_ms = h_inf_tau_h_ms(V_m_mV, settings)
        n_inf, tau_n_ms = n_inf_tau_n_ms(V_m_mV, settings)

        m_new = rush_larsen_step_gate(m, m_inf, tau_m_ms, dt_ms, settings.clamp_gate_range)
        h_new = rush_larsen_step_gate(h, h_inf, tau_h_ms, dt_ms, settings.clamp_gate_range)
        n_new = rush_larsen_step_gate(n, n_inf, tau_n_ms, dt_ms, settings.clamp_gate_range)
    else:
        m_new = euler_step_gate(m, dm_dt_per_ms(m, V_m_mV, settings), dt_ms, settings.clamp_gate_range)
        h_new = euler_step_gate(h, dh_dt_per_ms(h, V_m_mV, settings), dt_ms, settings.clamp_gate_range)
        n_new = euler_step_gate(n, dn_dt_per_ms(n, V_m_mV, settings), dt_ms, settings.clamp_gate_range)

    return {"m": m_new, "h": h_new, "n": n_new}
