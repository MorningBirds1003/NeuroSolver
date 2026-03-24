"""
coupling.py

Glue code between cable, VC, and KNP layers.

Purpose
-------
This file centralizes data exchange between:
- fast cable solver
- fast VC forward model
- slow KNP ECS solver

Phase 3 note
------------
The earlier version of this file used a one-species, nearest-grid-point KNP
projection that was sufficient for architecture testing but not for
physiological source bookkeeping.

This revision keeps the old API for backward compatibility, but adds a more
physiological source interface:
- linear deposition onto the 1D KNP grid
- optional multi-species current-density inputs
- explicit shared-source payloads for KNP and VC
- explicit source metadata for later bundle-aware diagnostics
"""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Any, List

import numpy as np

from Scripts.NeuroSolver.params import DEFAULT_PARAMS, SimulationParameters
from Scripts.NeuroSolver.ECS.vc_solver import (
    ElectrodeSamplePoint,
    sample_virtual_electrodes_from_cable,
)

# -----------------------------------------------------------------------------
# VC wrapper
# -----------------------------------------------------------------------------
def cable_to_vc_traces(
    cable_result: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray],
    electrode_points: Sequence[ElectrodeSamplePoint],
    params: SimulationParameters = DEFAULT_PARAMS,
    cable_y_um: float = 0.0,
    cable_z_um: float = 0.0,
    material_model: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert cable result into virtual-electrode VC traces.

    This remains a thin wrapper around vc_solver.sample_virtual_electrodes_from_cable.
    """
    return sample_virtual_electrodes_from_cable(
        cable_result=cable_result,
        geometry=geometry,
        electrode_points=electrode_points,
        params=params,
        cable_y_um=cable_y_um,
        cable_z_um=cable_z_um,
        material_model=material_model,
    )

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _validate_cable_projection_inputs(
    I_membrane_density_uA_per_cm2: np.ndarray,
    geometry: Dict[str, np.ndarray],
    knp_x_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    I_membrane_density_uA_per_cm2 = np.asarray(I_membrane_density_uA_per_cm2, dtype=float)
    area_cm2 = np.asarray(geometry["area_cm2"], dtype=float)
    x_cable_um = np.asarray(geometry["x_um"], dtype=float)
    knp_x_um = np.asarray(knp_x_um, dtype=float)

    if I_membrane_density_uA_per_cm2.ndim != 1:
        raise ValueError("I_membrane_density_uA_per_cm2 must be 1D.")
    if area_cm2.ndim != 1 or x_cable_um.ndim != 1:
        raise ValueError("geometry['area_cm2'] and geometry['x_um'] must be 1D arrays.")
    if I_membrane_density_uA_per_cm2.shape[0] != area_cm2.shape[0]:
        raise ValueError("Current-density and cable area arrays must have the same length.")
    if x_cable_um.shape[0] != area_cm2.shape[0]:
        raise ValueError("Cable x-position and area arrays must have the same length.")
    if knp_x_um.ndim != 1 or knp_x_um.size < 2:
        raise ValueError("knp_x_um must be a 1D grid with at least 2 points.")

    return I_membrane_density_uA_per_cm2, area_cm2, x_cable_um

# Validate species density map.
def _validate_species_density_map(
    species_current_density_uA_per_cm2: Optional[Mapping[str, np.ndarray]],
    n_compartments: int,
) -> Dict[str, np.ndarray]:
    if species_current_density_uA_per_cm2 is None:
        return {}

    out: Dict[str, np.ndarray] = {}
    for species, arr in species_current_density_uA_per_cm2.items():
        arr_f = np.asarray(arr, dtype=float)
        if arr_f.ndim != 1 or arr_f.shape[0] != n_compartments:
            raise ValueError(
                f"species_current_density_uA_per_cm2['{species}'] must have shape ({n_compartments},)"
            )
        out[str(species)] = arr_f
    return out

# Deposit linearly onto 1d grid.
def _deposit_linearly_onto_1d_grid(
    x_src_um: float,
    value: float,
    grid_x_um: np.ndarray,
    target: np.ndarray,
) -> None:
    """
    Linearly deposit a scalar source value onto the two nearest 1D grid points.

    This is a direct improvement over nearest-grid-point projection because it
    avoids artificially pinning each compartment source to a single KNP voxel.
    """
    x = float(x_src_um)
    v = float(value)

    if x <= float(grid_x_um[0]):
        target[0] += v
        return
    if x >= float(grid_x_um[-1]):
        target[-1] += v
        return

    j_right = int(np.searchsorted(grid_x_um, x, side="left"))
    j_left = j_right - 1

    x_left = float(grid_x_um[j_left])
    x_right = float(grid_x_um[j_right])

    dx = x_right - x_left
    if dx <= 0.0:
        target[j_left] += v
        return

    w_right = (x - x_left) / dx
    w_left = 1.0 - w_right

    target[j_left] += w_left * v
    target[j_right] += w_right * v

# Compartment coordinates from geometry.
def _compartment_coordinates_from_geometry(
    geometry: Dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_um = np.asarray(geometry["x_um"], dtype=float)
    y_um = np.asarray(
        geometry.get("y_um", np.full_like(x_um, float(geometry.get("fiber_center_y_um", 0.0)))),
        dtype=float,
    )
    z_um = np.asarray(
        geometry.get("z_um", np.full_like(x_um, float(geometry.get("fiber_center_z_um", 0.0)))),
        dtype=float,
    )
    return x_um, y_um, z_um

# -----------------------------------------------------------------------------
# Phase 3 physiological source payload
# -----------------------------------------------------------------------------
def project_cable_density_to_shared_source_payload(
    I_membrane_density_uA_per_cm2: np.ndarray,
    geometry: Dict[str, np.ndarray],
    knp_x_um: np.ndarray,
    species_current_density_uA_per_cm2: Optional[Mapping[str, np.ndarray]] = None,
    fallback_species_name: str = "K",
    scale_mM_per_ms_per_uA: float = 1.0e-5,
) -> Dict[str, Any]:
    """
    Project cable membrane-current activity into a shared ECS/KNP source payload.

    Parameters
    ----------
    I_membrane_density_uA_per_cm2
        Total membrane current density [uA/cm^2] for each cable compartment.
    geometry
        Cable geometry with at least:
        - area_cm2
        - x_um
        Optional:
        - y_um, z_um
        - fiber_id
        - compartment_global_index
    knp_x_um
        1D KNP grid positions [um].
    species_current_density_uA_per_cm2
        Optional map of ion-specific membrane current densities [uA/cm^2].
        This is the physiologically preferred input path. Example keys:
        - "K"
        - "Na"
        - "Cl"
        - "Ca"
    fallback_species_name
        If species_current_density_uA_per_cm2 is omitted, total membrane current
        is deposited into this one species for backward compatibility.
    scale_mM_per_ms_per_uA
        Coupling scale from absolute current [uA] to KNP source [mM/ms].

    Returns
    -------
    dict
        A bundle-ready source payload with:
        - species_sources_mM_per_ms
        - total_membrane_source_uA
        - capacitive_source_uA
        - vc_point_sources
        - source_metadata
    """
    I_membrane_density_uA_per_cm2, area_cm2, x_cable_um = _validate_cable_projection_inputs(
        I_membrane_density_uA_per_cm2=I_membrane_density_uA_per_cm2,
        geometry=geometry,
        knp_x_um=knp_x_um,
    )
    species_density_map = _validate_species_density_map(
        species_current_density_uA_per_cm2=species_current_density_uA_per_cm2,
        n_compartments=I_membrane_density_uA_per_cm2.shape[0],
    )

    x_um, y_um, z_um = _compartment_coordinates_from_geometry(geometry)

    # Convert density [uA/cm^2] -> absolute current [uA] for each compartment.
    I_total_uA = I_membrane_density_uA_per_cm2 * area_cm2

    # Physiologically preferable route: use explicit species currents if supplied.
    if species_density_map:
        species_total_current_uA = {
            species: np.asarray(density, dtype=float) * area_cm2
            for species, density in species_density_map.items()
        }
    else:
        species_total_current_uA = {
            str(fallback_species_name): I_total_uA.copy()
        }

    species_sources_mM_per_ms: Dict[str, np.ndarray] = {
        species: np.zeros(knp_x_um.size, dtype=float)
        for species in species_total_current_uA.keys()
    }
    total_membrane_source_uA = np.zeros(knp_x_um.size, dtype=float)

    # Placeholder for future capacitive-current bookkeeping.
    capacitive_source_uA = np.zeros(knp_x_um.size, dtype=float)

    # Linear deposition into the 1D KNP domain.
    for i, x_src_um in enumerate(x_cable_um):
        total_uA_i = float(I_total_uA[i])
        _deposit_linearly_onto_1d_grid(
            x_src_um=float(x_src_um),
            value=total_uA_i,
            grid_x_um=knp_x_um,
            target=total_membrane_source_uA,
        )

        for species, I_species_uA in species_total_current_uA.items():
            _deposit_linearly_onto_1d_grid(
                x_src_um=float(x_src_um),
                value=float(I_species_uA[i]) * float(scale_mM_per_ms_per_uA),
                grid_x_um=knp_x_um,
                target=species_sources_mM_per_ms[species],
            )

    compartment_global_index = np.asarray(
        geometry.get(
            "compartment_global_index",
            np.arange(I_total_uA.size, dtype=int),
        ),
        dtype=int,
    )

    vc_point_sources: List[Dict[str, Any]] = []
    for i in range(I_total_uA.size):
        vc_point_sources.append(
            {
                "fiber_id": int(geometry.get("fiber_id", 0)),
                "compartment_local_index": int(i),
                "compartment_global_index": int(compartment_global_index[i]),
                "x_um": float(x_um[i]),
                "y_um": float(y_um[i]),
                "z_um": float(z_um[i]),
                "I_uA": float(I_total_uA[i]),
            }
        )

    return {
        "species_sources_mM_per_ms": species_sources_mM_per_ms,
        "total_membrane_source_uA": total_membrane_source_uA,
        "capacitive_source_uA": capacitive_source_uA,
        "vc_point_sources": vc_point_sources,
        "source_metadata": {
            "fiber_id": int(geometry.get("fiber_id", 0)),
            "n_compartments": int(I_total_uA.size),
            "fallback_species_name": str(fallback_species_name),
            "used_species_density_map": bool(species_density_map),
            "scale_mM_per_ms_per_uA": float(scale_mM_per_ms_per_uA),
        },
    }

# Project bundle source payloads to shared terms.
def project_bundle_source_payloads_to_shared_terms(
    payloads: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Sum multiple per-fiber source payloads into one shared ECS payload.

    This is the natural bridge from explicit multi-fiber cable stepping into
    shared KNP/VC layers.
    """
    if len(payloads) == 0:
        return {
            "species_sources_mM_per_ms": {},
            "total_membrane_source_uA": None,
            "capacitive_source_uA": None,
            "vc_point_sources": [],
            "source_metadata": {"fiber_count": 0},
        }

    first_species = payloads[0]["species_sources_mM_per_ms"]
    species_names = sorted(
        {species for payload in payloads for species in payload["species_sources_mM_per_ms"].keys()}
    )

    total_membrane_source_uA = None
    capacitive_source_uA = None
    species_sources_mM_per_ms: Dict[str, np.ndarray] = {}
    vc_point_sources: List[Dict[str, Any]] = []

    for species in species_names:
        template = None
        for payload in payloads:
            if species in payload["species_sources_mM_per_ms"]:
                template = np.asarray(payload["species_sources_mM_per_ms"][species], dtype=float)
                break
        if template is None:
            continue
        species_sources_mM_per_ms[species] = np.zeros_like(template)

    for payload in payloads:
        total_arr = np.asarray(payload["total_membrane_source_uA"], dtype=float)
        cap_arr = np.asarray(payload["capacitive_source_uA"], dtype=float)

        if total_membrane_source_uA is None:
            total_membrane_source_uA = np.zeros_like(total_arr)
        if capacitive_source_uA is None:
            capacitive_source_uA = np.zeros_like(cap_arr)

        total_membrane_source_uA += total_arr
        capacitive_source_uA += cap_arr

        for species, arr in payload["species_sources_mM_per_ms"].items():
            species_sources_mM_per_ms[species] += np.asarray(arr, dtype=float)

        vc_point_sources.extend(list(payload["vc_point_sources"]))

    return {
        "species_sources_mM_per_ms": species_sources_mM_per_ms,
        "total_membrane_source_uA": total_membrane_source_uA,
        "capacitive_source_uA": capacitive_source_uA,
        "vc_point_sources": vc_point_sources,
        "source_metadata": {
            "fiber_count": int(len(payloads)),
            "species_names": species_names,
        },
    }

# -----------------------------------------------------------------------------
# Backward-compatible legacy API
# -----------------------------------------------------------------------------
def project_cable_density_to_knp_source_terms(
    I_membrane_density_uA_per_cm2: np.ndarray,
    geometry: Dict[str, np.ndarray],
    knp_x_um: np.ndarray,
    species_name: str = "K",
    scale_mM_per_ms_per_uA: float = 1.0e-5,
) -> Dict[str, np.ndarray]:
    """
    Backward-compatible wrapper for older scheduler code.

    This now delegates to the Phase 3 payload builder and returns only the
    species-source map expected by advance_knp_state_1d(...).
    """
    payload = project_cable_density_to_shared_source_payload(
        I_membrane_density_uA_per_cm2=I_membrane_density_uA_per_cm2,
        geometry=geometry,
        knp_x_um=knp_x_um,
        species_current_density_uA_per_cm2=None,
        fallback_species_name=species_name,
        scale_mM_per_ms_per_uA=scale_mM_per_ms_per_uA,
    )
    return payload["species_sources_mM_per_ms"]

# Placeholder knp feedback to membrane.
def placeholder_knp_feedback_to_membrane(
    concentrations_mM: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Placeholder hook for future KNP -> membrane reversal-potential feedback.

    Right now this returns an empty dictionary on purpose.
    """
    return {}
