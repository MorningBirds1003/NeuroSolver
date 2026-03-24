"""
knp_solver.py

Minimal 1D electroneutral KNP-style extracellular solver for NeuroSolver.

Purpose
-------
This is a compact first-pass KNP implementation for:
- 1D extracellular domains
- slow concentration evolution
- simple electroneutral / diffusion-potential behavior
- debugging cable -> ECS source coupling

This version adds explicit debug visibility:
- stores source-term history
- reports concentration deviation from baseline
- reports phi deviation from initial
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from Scripts.NeuroSolver.params import DEFAULT_PARAMS, SimulationParameters, um_to_m

@dataclass(frozen=True)
class KNPDomain1D:
    """
    One-dimensional extracellular domain.
    """
    x_um: np.ndarray
    dx_um: float

# Build uniform knp domain 1d.
def build_uniform_knp_domain_1d(
    length_um: float,
    dx_um: float,
) -> KNPDomain1D:
    """
    Build a uniform 1D domain.
    """
    if dx_um <= 0.0:
        raise ValueError("dx_um must be positive.")
    n = int(round(length_um / dx_um)) + 1
    x_um = np.linspace(0.0, float(length_um), n, dtype=float)
    return KNPDomain1D(x_um=x_um, dx_um=float(dx_um))

# Initialize species concentrations mm.
def initialize_species_concentrations_mM(
    domain: KNPDomain1D,
    params: SimulationParameters = DEFAULT_PARAMS,
) -> Dict[str, np.ndarray]:
    """
    Initialize extracellular concentrations from params.
    """
    n = domain.x_um.size
    return {
        "Na": np.full(n, float(params.ions.sodium.extracellular_mM), dtype=float),
        "K": np.full(n, float(params.ions.potassium.extracellular_mM), dtype=float),
        "Cl": np.full(n, float(params.ions.chloride.extracellular_mM), dtype=float),
        "Ca": np.full(n, float(params.ions.calcium.extracellular_mM), dtype=float),
    }

# Initialize phi e mv.
def initialize_phi_e_mV(
    domain: KNPDomain1D,
) -> np.ndarray:
    """
    Initialize extracellular potential to zero.
    """
    return np.zeros(domain.x_um.size, dtype=float)

# Baseline species concentrations mm.
def baseline_species_concentrations_mM(
    domain: KNPDomain1D,
    params: SimulationParameters = DEFAULT_PARAMS,
) -> Dict[str, np.ndarray]:
    """
    Return baseline extracellular concentrations for delta calculations.
    """
    return initialize_species_concentrations_mM(domain, params)

# Species metadata.
def _species_metadata(params: SimulationParameters = DEFAULT_PARAMS) -> Dict[str, Dict[str, float]]:
    """
    Return species valence and diffusion metadata.
    """
    return {
        "Na": {
            "z": float(params.ions.sodium.valence),
            "D_m2_per_s": float(params.ions.sodium.diffusion_m2_per_s),
        },
        "K": {
            "z": float(params.ions.potassium.valence),
            "D_m2_per_s": float(params.ions.potassium.diffusion_m2_per_s),
        },
        "Cl": {
            "z": float(params.ions.chloride.valence),
            "D_m2_per_s": float(params.ions.chloride.diffusion_m2_per_s),
        },
        "Ca": {
            "z": float(params.ions.calcium.valence),
            "D_m2_per_s": float(params.ions.calcium.diffusion_m2_per_s),
        },
    }

# Compute effective diffusion m2 per s.
def compute_effective_diffusion_m2_per_s(
    D_free_m2_per_s: float,
    params: SimulationParameters = DEFAULT_PARAMS,
) -> float:
    """
    Apply ECS tortuosity scaling to diffusion coefficient.
    """
    return float(D_free_m2_per_s) * float(params.ecs.diffusivity_scale_factor)

# Compute sigma from concentrations s per m.
def compute_sigma_from_concentrations_S_per_m(
    concentrations_mM: Dict[str, np.ndarray],
    params: SimulationParameters = DEFAULT_PARAMS,
) -> np.ndarray:
    """
    Approximate local conductivity from concentrations.

    Uses:
        sigma ~ (F^2 / RT) * sum(z_i^2 D_i c_i)

    with mM treated as mol/m^3.
    """
    meta = _species_metadata(params)
    F = float(params.physical.faraday_C_per_mol)
    R = float(params.physical.gas_constant_J_per_molK)
    T = float(params.temperature.kelvin)

    first_key = next(iter(concentrations_mM))
    n = np.asarray(concentrations_mM[first_key], dtype=float).size
    sigma = np.zeros(n, dtype=float)

    for name, c_mM in concentrations_mM.items():
        c = np.asarray(c_mM, dtype=float)
        z = float(meta[name]["z"])
        D_eff = compute_effective_diffusion_m2_per_s(meta[name]["D_m2_per_s"], params)
        sigma += (z * z) * D_eff * c

    sigma *= (F * F) / (R * T)
    return sigma

# Laplacian 1d.
def _laplacian_1d(y: np.ndarray, dx_m: float) -> np.ndarray:
    """
    Second derivative with zero-flux edge handling.
    """
    y = np.asarray(y, dtype=float)
    lap = np.zeros_like(y)

    lap[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (dx_m * dx_m)
    lap[0] = (y[1] - y[0]) / (dx_m * dx_m)
    lap[-1] = (y[-2] - y[-1]) / (dx_m * dx_m)

    return lap

# Gradient 1d.
def _gradient_1d(y: np.ndarray, dx_m: float) -> np.ndarray:
    """
    First derivative with one-sided edges.
    """
    y = np.asarray(y, dtype=float)
    grad = np.zeros_like(y)

    grad[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx_m)
    grad[0] = (y[1] - y[0]) / dx_m
    grad[-1] = (y[-1] - y[-2]) / dx_m

    return grad

# Compute phi diffusive mv.
def compute_phi_diffusive_mV(
    concentrations_mM: Dict[str, np.ndarray],
    params: SimulationParameters = DEFAULT_PARAMS,
) -> np.ndarray:
    """
    Compute a baseline-referenced diffusion-potential-like correction.

    This remains a pragmatic approximation, not a full elliptic KNP solve.
    The key idea here is to make phi_diff respond to spatial gradients of
    concentration deviations from baseline, rather than only to local
    pointwise ratio cancellation.

    Returns
    -------
    np.ndarray
        Diffusive extracellular potential [mV].
    """
    meta = _species_metadata(params)
    R = float(params.physical.gas_constant_J_per_molK)
    T = float(params.temperature.kelvin)
    F = float(params.physical.faraday_C_per_mol)

    baseline_map = {
        "Na": float(params.ions.sodium.extracellular_mM),
        "K": float(params.ions.potassium.extracellular_mM),
        "Cl": float(params.ions.chloride.extracellular_mM),
        "Ca": float(params.ions.calcium.extracellular_mM),
    }

    first_key = next(iter(concentrations_mM))
    c0 = np.asarray(concentrations_mM[first_key], dtype=float)
    n = c0.size

    dx_m = float(params.knp.dx_um) * 1.0e-6

    denom = np.zeros(n, dtype=float)
    numer_grad = np.zeros(n, dtype=float)

    for name, c_mM in concentrations_mM.items():
        c = np.maximum(np.asarray(c_mM, dtype=float), 1.0e-12)
        c_ref = max(float(baseline_map[name]), 1.0e-12)

        z = float(meta[name]["z"])
        D_eff = compute_effective_diffusion_m2_per_s(meta[name]["D_m2_per_s"], params)

        # Baseline-referenced log concentration ratio
        log_ratio = np.log(c / c_ref)

        # Make phi_diff respond to spatial structure, not just local cancellation
        dlog_ratio_dx = _gradient_1d(log_ratio, dx_m)

        denom += (z * z) * D_eff * c
        numer_grad += z * D_eff * dlog_ratio_dx

    phi_grad_V_per_m = np.zeros(n, dtype=float)
    valid = denom > 1.0e-18
    phi_grad_V_per_m[valid] = -(R * T / F) * (numer_grad[valid] / denom[valid])

    # Integrate field -> potential, anchored to zero mean to avoid arbitrary drift
    phi_diff_V = np.cumsum(phi_grad_V_per_m) * dx_m
    phi_diff_V -= np.mean(phi_diff_V)

    return 1000.0 * phi_diff_V

# Advance knp state 1d.
def advance_knp_state_1d(
    concentrations_mM: Dict[str, np.ndarray],
    phi_e_mV: np.ndarray,
    domain: KNPDomain1D,
    dt_ms: float,
    params: SimulationParameters = DEFAULT_PARAMS,
    source_terms_mM_per_ms: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, object]:
    """
    Advance 1D extracellular concentrations and potential by one slow step.

    Returns
    -------
    dict
        {
            "concentrations_mM": {...},
            "phi_e_mV": phi_next,
            "sigma_S_per_m": sigma,
            "debug": {...}
        }
    """
    dt_s = float(dt_ms) * 1.0e-3
    dx_m = um_to_m(float(domain.dx_um))

    meta = _species_metadata(params)
    F = float(params.physical.faraday_C_per_mol)
    R = float(params.physical.gas_constant_J_per_molK)
    T = float(params.temperature.kelvin)

    floor_conc = float(getattr(params.knp, "floor_concentration_mM", 1.0e-9))

    phi_V = np.asarray(phi_e_mV, dtype=float) * 1.0e-3
    dphi_dx = _gradient_1d(phi_V, dx_m)

    new_conc: Dict[str, np.ndarray] = {}
    used_sources: Dict[str, np.ndarray] = {}

    for name, c_old_mM in concentrations_mM.items():
        c_old = np.maximum(np.asarray(c_old_mM, dtype=float), floor_conc)
        z = float(meta[name]["z"])
        D_eff = compute_effective_diffusion_m2_per_s(meta[name]["D_m2_per_s"], params)

        d2c_dx2 = _laplacian_1d(c_old, dx_m)

        drift_flux = (D_eff * z * F / (R * T)) * c_old * dphi_dx
        d_drift_flux_dx = _gradient_1d(drift_flux, dx_m)

        source = np.zeros_like(c_old)
        if source_terms_mM_per_ms is not None and name in source_terms_mM_per_ms:
            source = np.asarray(source_terms_mM_per_ms[name], dtype=float)

        if bool(getattr(params.knp, "source_volume_fraction_scaling", False)):
            alpha = max(float(getattr(params.ecs, "volume_fraction", 1.0)), 1.0e-9)
            source = source / alpha

        baseline = baseline_species_concentrations_mM(domain, params)[name]

        clearance = np.zeros_like(c_old)
        if bool(getattr(params.knp, "clearance_enabled", False)):
            tau_ms_map = {
                "K": float(getattr(params.knp, "clearance_tau_K_ms", 500.0)),
                "Na": float(getattr(params.knp, "clearance_tau_Na_ms", 1000.0)),
                "Cl": float(getattr(params.knp, "clearance_tau_Cl_ms", 1000.0)),
                "Ca": float(getattr(params.knp, "clearance_tau_Ca_ms", 1000.0)),
            }
            tau_ms = max(float(tau_ms_map.get(name, 1000.0)), 1.0e-9)
            clearance = -(c_old - baseline) / tau_ms

        c_next = (
            c_old
            + dt_s * D_eff * d2c_dx2
            - dt_s * d_drift_flux_dx
            + float(dt_ms) * source
            + float(dt_ms) * clearance
        )
        c_next = np.maximum(c_next, floor_conc)

        new_conc[name] = c_next
        used_sources[name] = source

    phi_next_mV = compute_phi_diffusive_mV(new_conc, params)
    sigma_next = compute_sigma_from_concentrations_S_per_m(new_conc, params)

    debug = {
        "max_abs_source_mM_per_ms": max(
            float(np.max(np.abs(src))) for src in used_sources.values()
        ) if used_sources else 0.0,
        "max_abs_phi_change_mV": float(np.max(np.abs(phi_next_mV - np.asarray(phi_e_mV, dtype=float)))),
    }

    return {
        "concentrations_mM": new_conc,
        "phi_e_mV": phi_next_mV,
        "sigma_S_per_m": sigma_next,
        "debug": debug,
    }

# Run knp toy diffusion test.
def run_knp_toy_diffusion_test(
    params: SimulationParameters = DEFAULT_PARAMS,
    length_um: float = 1000.0,
    dx_um: float = 10.0,
    dt_ms: float = 0.1,
    t_stop_ms: float = 50.0,
    perturb_species: str = "K",
    perturb_center_um: float = 500.0,
    perturb_width_um: float = 50.0,
    perturb_amplitude_mM: float = 2.0,
) -> Dict[str, object]:
    """
    Run a standalone toy KNP diffusion test.
    """
    domain = build_uniform_knp_domain_1d(length_um=length_um, dx_um=dx_um)
    concentrations = initialize_species_concentrations_mM(domain, params)
    baseline = baseline_species_concentrations_mM(domain, params)
    phi_e_mV = initialize_phi_e_mV(domain)

    x = domain.x_um
    gauss = float(perturb_amplitude_mM) * np.exp(
        -0.5 * ((x - float(perturb_center_um)) / float(perturb_width_um)) ** 2
    )
    concentrations[perturb_species] = concentrations[perturb_species] + gauss

    n_steps = int(round(float(t_stop_ms) / float(dt_ms)))

    t_history: List[float] = []
    phi_history: List[np.ndarray] = []
    sigma_history: List[np.ndarray] = []
    species_history: Dict[str, List[np.ndarray]] = {k: [] for k in concentrations.keys()}

    for step in range(n_steps + 1):
        t_ms = step * float(dt_ms)
        t_history.append(t_ms)
        phi_history.append(phi_e_mV.copy())
        sigma_history.append(compute_sigma_from_concentrations_S_per_m(concentrations, params))
        for name in concentrations:
            species_history[name].append(concentrations[name].copy())

        updated = advance_knp_state_1d(
            concentrations_mM=concentrations,
            phi_e_mV=phi_e_mV,
            domain=domain,
            dt_ms=float(dt_ms),
            params=params,
            source_terms_mM_per_ms=None,
        )
        concentrations = updated["concentrations_mM"]
        phi_e_mV = updated["phi_e_mV"]

    baseline_arrays = {k: np.asarray(v, dtype=float) for k, v in baseline.items()}

    return {
        "t_ms": np.asarray(t_history, dtype=float),
        "x_um": domain.x_um.copy(),
        "phi_e_mV": np.asarray(phi_history, dtype=float),
        "sigma_S_per_m": np.asarray(sigma_history, dtype=float),
        "species_mM": {k: np.asarray(v, dtype=float) for k, v in species_history.items()},
        "baseline_species_mM": baseline_arrays,
    }
