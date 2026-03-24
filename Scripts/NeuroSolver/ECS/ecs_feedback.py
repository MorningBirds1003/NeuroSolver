"""
ecs_feedback.py

Slow membrane/ECS feedback interface for NeuroSolver.

Purpose
-------
Translate the slow 1D KNP state into quantities the fast membrane model can use
without forcing the cable solver to solve a fully coupled electrodiffusive
problem on every fast step.

This module implements the exact bridge called out in the updated design
blueprint:
- sample slow extracellular concentrations onto the cable geometry
- update Nernst reversals on a slow cadence
- expose slow extracellular potential offsets back to the fast solver

Design assumptions
------------------
1. The KNP state evolves slowly compared to the cable state.
2. Feedback should therefore be piecewise constant over one slow interval.
3. The current KNP implementation is 1D, so cable coupling is performed by
   sampling along x only.
4. This interface is intentionally conservative and modular. It does not try to
   force a full EMI formulation into the present code base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from Scripts.NeuroSolver.params import DEFAULT_PARAMS, SimulationParameters, nernst_potential_mV

@dataclass
class ECSFeedbackSnapshot:
    """
    Slow feedback state held constant across fast cable steps.

    Attributes
    ----------
    phi_e_cable_mV
        Extracellular potential sampled onto each cable compartment.
    reversal_overrides_mV
        Slow Nernst updates keyed by membrane-current field names expected by the
        fast membrane model, e.g. ``E_k_mV``.
    extracellular_species_cable_mM
        Species concentrations sampled onto the cable compartments.
    metadata
        Debug / provenance fields.
    """

    phi_e_cable_mV: np.ndarray
    reversal_overrides_mV: Dict[str, float]
    extracellular_species_cable_mM: Dict[str, np.ndarray]
    metadata: Dict[str, object]

@dataclass(frozen=True)
class ECSFeedbackConfig:
    """Configuration derived from params with sensible fallbacks."""

    enabled: bool = False
    apply_phi_offset: bool = True
    apply_reversal_feedback: bool = True
    subtract_initial_phi_reference: bool = True
    phi_gain: float = 1.0
    phi_clamp_abs_mV: float = 100.0
    sample_mode: str = "linear"
    coordinate_mode: str = "auto"
    reversal_reduction: str = "mean"
    reversal_clamp_abs_shift_mV: float = 50.0

# Resolve feedback config.
def resolve_feedback_config(params: SimulationParameters = DEFAULT_PARAMS) -> ECSFeedbackConfig:
    """
    Resolve optional slow-feedback settings from the current parameter tree.

    This function is intentionally tolerant of missing attributes so the new
    module can be dropped into the present code base before params.py is fully
    extended.
    """
    policy_enabled = bool(getattr(params.policy, "use_dynamic_ecs_feedback", False))
    knp_enabled = bool(getattr(params.knp, "enabled", False))

    enabled = policy_enabled and knp_enabled
    apply_phi_offset = bool(getattr(params.knp, "feedback_apply_phi_offset", True))
    apply_reversal_feedback = bool(getattr(params.knp, "feedback_apply_reversal_updates", True))
    subtract_initial_phi_reference = bool(
        getattr(params.knp, "feedback_subtract_initial_phi_reference", True)
    )
    phi_gain = float(getattr(params.knp, "feedback_phi_gain", 1.0))
    phi_clamp_abs_mV = float(getattr(params.knp, "feedback_phi_clamp_abs_mV", 100.0))
    sample_mode = str(getattr(params.knp, "feedback_sampling_mode", "linear")).strip().lower()
    coordinate_mode = str
    (
        getattr(params.knp, "feedback_coordinate_mode", "auto")
    ).strip().lower()
    reversal_reduction = str
    (
        getattr(params.knp, "feedback_reversal_reduction", "mean")
    ).strip().lower()
    reversal_clamp_abs_shift_mV = float
    (
        getattr(params.knp, "feedback_reversal_clamp_abs_shift_mV", 50.0)
    )

    return ECSFeedbackConfig(
        enabled=enabled,
        apply_phi_offset=apply_phi_offset,
        apply_reversal_feedback=apply_reversal_feedback,
        subtract_initial_phi_reference=subtract_initial_phi_reference,
        phi_gain=phi_gain,
        phi_clamp_abs_mV=phi_clamp_abs_mV,
        sample_mode=sample_mode,
        coordinate_mode=coordinate_mode,
        reversal_reduction=reversal_reduction,
    )

# Sample 1d field to positions.
def _sample_1d_field_to_positions(
    field_values: Sequence[float],
    field_x_um: Sequence[float],
    sample_x_um: Sequence[float],
    mode: str = "linear",
) -> np.ndarray:
    """Sample a 1D field onto new positions."""
    field_values = np.asarray(field_values, dtype=float)
    field_x_um = np.asarray(field_x_um, dtype=float)
    sample_x_um = np.asarray(sample_x_um, dtype=float)

    if field_values.ndim != 1 or field_x_um.ndim != 1:
        raise ValueError("field_values and field_x_um must be 1D arrays.")
    if field_values.size != field_x_um.size:
        raise ValueError("field_values and field_x_um must have identical lengths.")
    if field_values.size == 0:
        return np.zeros_like(sample_x_um, dtype=float)

    mode = str(mode).strip().lower()
    if mode == "nearest":
        idx = np.abs(field_x_um[:, None] - sample_x_um[None, :]).argmin(axis=0)
        return field_values[idx]

    # default: linear interpolation with clamped end values
    return np.interp(sample_x_um, field_x_um, field_values, left=field_values[0], right=field_values[-1])

# Normalize sample positions to field span.
def _normalize_sample_positions_to_field_span(
    field_x_um: Sequence[float],
    sample_x_um: Sequence[float],
    coordinate_mode: str = "auto",
) -> tuple[np.ndarray, str]:
    """Map cable positions onto the KNP span when the two domains differ strongly."""
    field_x_um = np.asarray(field_x_um, dtype=float)
    sample_x_um = np.asarray(sample_x_um, dtype=float)

    if field_x_um.size == 0 or sample_x_um.size == 0:
        return sample_x_um.copy(), "direct"

    field_min = float(np.min(field_x_um))
    field_max = float(np.max(field_x_um))
    sample_min = float(np.min(sample_x_um))
    sample_max = float(np.max(sample_x_um))

    field_span = max(field_max - field_min, 1.0e-12)
    sample_span = max(sample_max - sample_min, 1.0e-12)

    mode = str(coordinate_mode).strip().lower()
    if mode == "direct":
        return sample_x_um.copy(), "direct"
    if mode == "normalize":
        normalized = field_min + ((sample_x_um - sample_min) / sample_span) * field_span
        return normalized, "normalized"

    # auto: normalize when spans differ substantially
    span_ratio = max(sample_span / field_span, field_span / sample_span)
    if span_ratio > 2.0:
        normalized = field_min + ((sample_x_um - sample_min) / sample_span) * field_span
        return normalized, "normalized"
    return sample_x_um.copy(), "direct"

# Sample knp state onto cable.
def sample_knp_state_onto_cable(
    knp_x_um: Sequence[float],
    concentrations_mM: Mapping[str, Sequence[float]],
    phi_e_mV: Sequence[float],
    cable_x_um: Sequence[float],
    params: SimulationParameters = DEFAULT_PARAMS,
    mode: str = "linear",
    coordinate_mode: str = "auto",
) -> Dict[str, object]:
    """
    Sample the slow KNP state onto cable compartment centers.
    """
    sample_x_mapped_um, applied_coordinate_mode = _normalize_sample_positions_to_field_span(
        field_x_um=knp_x_um,
        sample_x_um=cable_x_um,
        coordinate_mode=coordinate_mode,
    )

    sampled_species: Dict[str, np.ndarray] = {}
    for species_name, species_field in concentrations_mM.items():
        sampled_species[str(species_name)] = _sample_1d_field_to_positions(
            field_values=species_field,
            field_x_um=knp_x_um,
            sample_x_um=sample_x_mapped_um,
            mode=mode,
        )

    sampled_phi = _sample_1d_field_to_positions(
        field_values=phi_e_mV,
        field_x_um=knp_x_um,
        sample_x_um=sample_x_mapped_um,
        mode=mode,
    )

    return {
        "species_mM": sampled_species,
        "phi_e_cable_mV": sampled_phi,
        "sigma_reference_S_per_m": float(getattr(params.ecs, "conductivity_S_per_m", 0.30)),
        "sample_x_mapped_um": sample_x_mapped_um,
        "coordinate_mode_applied": applied_coordinate_mode,
    }

# Compute reversal overrides mv.
def compute_reversal_overrides_mV(
    extracellular_species_cable_mM: Mapping[str, Sequence[float]],
    params: SimulationParameters = DEFAULT_PARAMS,
    reduction: str = "mean",
) -> Dict[str, float]:
    """
    Convert sampled extracellular concentrations into slow Nernst reversals.

    The fast membrane model currently expects scalar reversal values. For now we
    therefore reduce the sampled cable concentrations to one representative ECS
    value per species. Mean is the safest default for the present solver.
    """
    reduction = str(reduction).strip().lower()

    def _reduce(arr: Sequence[float]) -> float:
        vec = np.asarray(arr, dtype=float)
        if vec.size == 0:
            raise ValueError("Cannot reduce an empty concentration vector.")
        if reduction == "median":
            return float(np.median(vec))
        if reduction == "center":
            return float(vec[vec.size // 2])
        return float(np.mean(vec))

    overrides: Dict[str, float] = {}

    if "Na" in extracellular_species_cable_mM:
        overrides["E_na_mV"] = nernst_potential_mV(
            z=int(params.ions.sodium.valence),
            c_out_mM=max(_reduce(extracellular_species_cable_mM["Na"]), 1.0e-12),
            c_in_mM=float(params.ions.sodium.intracellular_mM),
            temperature_celsius=float(params.temperature.celsius),
        )

    if "K" in extracellular_species_cable_mM:
        overrides["E_k_mV"] = nernst_potential_mV(
            z=int(params.ions.potassium.valence),
            c_out_mM=max(_reduce(extracellular_species_cable_mM["K"]), 1.0e-12),
            c_in_mM=float(params.ions.potassium.intracellular_mM),
            temperature_celsius=float(params.temperature.celsius),
        )

    if "Cl" in extracellular_species_cable_mM:
        overrides["E_cl_mV"] = nernst_potential_mV(
            z=int(params.ions.chloride.valence),
            c_out_mM=max(_reduce(extracellular_species_cable_mM["Cl"]), 1.0e-12),
            c_in_mM=float(params.ions.chloride.intracellular_mM),
            temperature_celsius=float(params.temperature.celsius),
        )

    if "Ca" in extracellular_species_cable_mM:
        overrides["E_ca_mV"] = nernst_potential_mV(
            z=int(params.ions.calcium.valence),
            c_out_mM=max(_reduce(extracellular_species_cable_mM["Ca"]), 1.0e-12),
            c_in_mM=float(params.ions.calcium.intracellular_mM),
            temperature_celsius=float(params.temperature.celsius),
        )

    return overrides

# Compute phi feedback vector mv.
def compute_phi_feedback_vector_mV(
    sampled_phi_e_mV: Sequence[float],
    config: ECSFeedbackConfig,
    initial_phi_reference_mV: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Compute the slow extracellular-potential vector applied back to the cable.
    """
    phi = np.asarray(sampled_phi_e_mV, dtype=float).copy()

    if config.subtract_initial_phi_reference and initial_phi_reference_mV is not None:
        phi = phi - np.asarray(initial_phi_reference_mV, dtype=float)

    phi *= float(config.phi_gain)
    phi = np.clip(phi, -float(config.phi_clamp_abs_mV), float(config.phi_clamp_abs_mV))
    return phi

# Build feedback snapshot.
def build_feedback_snapshot(
    knp_x_um: Sequence[float],
    concentrations_mM: Mapping[str, Sequence[float]],
    phi_e_mV: Sequence[float],
    cable_x_um: Sequence[float],
    params: SimulationParameters = DEFAULT_PARAMS,
    initial_phi_reference_mV: Optional[Sequence[float]] = None,
) -> ECSFeedbackSnapshot:
    """
    Build one slow feedback snapshot from the current KNP state.
    """
    config = resolve_feedback_config(params)
    sampled = sample_knp_state_onto_cable(
        knp_x_um=knp_x_um,
        concentrations_mM=concentrations_mM,
        phi_e_mV=phi_e_mV,
        cable_x_um=cable_x_um,
        params=params,
        mode=config.sample_mode,
        coordinate_mode=config.coordinate_mode,
    )

    phi_feedback = compute_phi_feedback_vector_mV(
        sampled_phi_e_mV=sampled["phi_e_cable_mV"],
        config=config,
        initial_phi_reference_mV=initial_phi_reference_mV,
    ) if config.apply_phi_offset else np.zeros_like(np.asarray(cable_x_um, dtype=float), dtype=float)

    if config.apply_reversal_feedback:
        reversal_overrides = compute_reversal_overrides_mV(
            extracellular_species_cable_mM=sampled["species_mM"],
            params=params,
            reduction=config.reversal_reduction,
        )

        baseline_reversals = {
            "E_na_mV": float(getattr(params.membrane, "sodium_reversal_mV", reversal_overrides.get("E_na_mV", 0.0))),
            "E_k_mV": float(getattr(params.membrane, "potassium_reversal_mV", reversal_overrides.get("E_k_mV", 0.0))),
            "E_cl_mV": float(getattr(params.membrane, "chloride_reversal_mV", reversal_overrides.get("E_cl_mV", 0.0))),
            "E_ca_mV": float(getattr(params.membrane, "calcium_reversal_mV", reversal_overrides.get("E_ca_mV", 0.0))),
        }

        for key, base in baseline_reversals.items():
            if key in reversal_overrides:
                lo = base - float(config.reversal_clamp_abs_shift_mV)
                hi = base + float(config.reversal_clamp_abs_shift_mV)
                reversal_overrides[key] = float(np.clip(reversal_overrides[key], lo, hi))
    else:
        reversal_overrides = {}

    return ECSFeedbackSnapshot(
        phi_e_cable_mV=phi_feedback,
        reversal_overrides_mV=reversal_overrides,
        extracellular_species_cable_mM=sampled["species_mM"],
        metadata={
            "enabled": bool(config.enabled),
            "sample_mode": config.sample_mode,
            "coordinate_mode": config.coordinate_mode,
            "coordinate_mode_applied": sampled.get("coordinate_mode_applied", "direct"),
            "phi_gain": float(config.phi_gain),
            "phi_clamp_abs_mV": float(config.phi_clamp_abs_mV),
            "apply_phi_offset": bool(config.apply_phi_offset),
            "apply_reversal_feedback": bool(config.apply_reversal_feedback),
            "reversal_reduction": config.reversal_reduction,
            "species_summary": {
                name: {
                    "mean_mM": float(np.mean(arr)),
                    "min_mM": float(np.min(arr)),
                    "max_mM": float(np.max(arr)),
                }
                for name, arr in sampled["species_mM"].items()
            },
        }
    )

# Build initial feedback snapshot.
def build_initial_feedback_snapshot(
    geometry: Mapping[str, Sequence[float]],
    params: SimulationParameters = DEFAULT_PARAMS,
) -> ECSFeedbackSnapshot:
    """
    Build a neutral snapshot used before the first slow KNP update arrives.
    """
    cable_x_um = np.asarray(geometry["x_um"], dtype=float)

    baseline_species = {
        "Na": np.full_like(cable_x_um, float(params.ions.sodium.extracellular_mM), dtype=float),
        "K": np.full_like(cable_x_um, float(params.ions.potassium.extracellular_mM), dtype=float),
        "Cl": np.full_like(cable_x_um, float(params.ions.chloride.extracellular_mM), dtype=float),
        "Ca": np.full_like(cable_x_um, float(params.ions.calcium.extracellular_mM), dtype=float),
    }

    return ECSFeedbackSnapshot(
        phi_e_cable_mV=np.zeros_like(cable_x_um, dtype=float),
        reversal_overrides_mV=compute_reversal_overrides_mV(baseline_species, params=params),
        extracellular_species_cable_mM=baseline_species,
        metadata={
            "enabled": False,
            "is_initial_snapshot": True,
        },
    )
