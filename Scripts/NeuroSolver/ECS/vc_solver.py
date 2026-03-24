"""
vc_solver.py

Fast quasi-static extracellular volume conductor utilities.

Purpose
-------
This is the fast extracellular forward model layer:
- point-source potential in a homogeneous medium
- superposition from many sources
- cable-to-electrode waveform sampling
- lightweight extracellular stimulation/readout support

Current approximation
---------------------
This first version uses the infinite homogeneous-medium point-source model:

    phi(r) = I / (4 * pi * sigma * |r-r0|)

This is appropriate for early validation before moving to FEM or more complex
anisotropic geometries.

Conventions
-----------
- source current: uA
- coordinates: um
- conductivity: S/m
- returned potential: mV
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np

from ..params import DEFAULT_PARAMS, SimulationParameters, um_to_m
from Scripts.NeuroSolver.architecture.vc_material_model import RadialMaterialModel

@dataclass(frozen=True)
class ElectrodeSamplePoint:
    """
    One electrode/contact sampling location.
    """
    name: str
    x_um: float
    y_um: float
    z_um: float

# Distance um.
def distance_um(
    a_xyz_um: Sequence[float],
    b_xyz_um: Sequence[float],
) -> float:
    """
    Euclidean distance between two 3D points in um.
    """
    dx = float(a_xyz_um[0]) - float(b_xyz_um[0])
    dy = float(a_xyz_um[1]) - float(b_xyz_um[1])
    dz = float(a_xyz_um[2]) - float(b_xyz_um[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)

# Compute point source phi e mv.
def compute_point_source_phi_e_mV(
    source_current_uA: float,
    source_xyz_um: Tuple[float, float, float],
    sample_xyz_um: Tuple[float, float, float],
    sigma_S_per_m: float,
    epsilon_um: float = 5.0,
) -> float:
    """
    Extracellular potential from a point source in a homogeneous medium.

    Formula
    -------
        phi = I / (4 * pi * sigma * r)

    Returns
    -------
    float
        Extracellular potential in mV.
    """
    sigma_S_per_m = float(sigma_S_per_m)
    if sigma_S_per_m <= 0.0:
        raise ValueError("sigma_S_per_m must be positive.")

    r_um = max(distance_um(source_xyz_um, sample_xyz_um), float(epsilon_um))
    r_m = um_to_m(r_um)
    I_A = float(source_current_uA) * 1.0e-6

    phi_V = I_A / (4.0 * math.pi * sigma_S_per_m * r_m)
    return 1000.0 * phi_V

# Compute superposed phi e mv.
def compute_superposed_phi_e_mV(
    source_currents_uA: Iterable[float],
    source_positions_um: Iterable[Tuple[float, float, float]],
    sample_xyz_um: Tuple[float, float, float],
    params: SimulationParameters = DEFAULT_PARAMS,
    material_model: Optional[RadialMaterialModel] = None,
) -> float:
    """
    Superpose extracellular potential from multiple point sources.

    Uses the original homogeneous-medium model when material_model is None.
    Uses the material-aware shielded model otherwise.
    """
    sigma = float(params.vc.sigma_x_S_per_m)
    epsilon = float(params.vc.singularity_epsilon_um)

    phi_total_mV = 0.0
    sx, sy, sz = float(sample_xyz_um[0]), float(sample_xyz_um[1]), float(sample_xyz_um[2])

    for I_uA, src_xyz in zip(source_currents_uA, source_positions_um):
        src_x, src_y, src_z = float(src_xyz[0]), float(src_xyz[1]), float(src_xyz[2])

        if material_model is None:
            phi_total_mV += compute_point_source_phi_e_mV(
                source_current_uA=float(I_uA),
                source_xyz_um=(src_x, src_y, src_z),
                sample_xyz_um=(sx, sy, sz),
                sigma_S_per_m=sigma,
                epsilon_um=epsilon,
            )
        else:
            phi_total_mV += compute_material_aware_point_source_potential_mV(
                source_current_uA=float(I_uA),
                source_x_um=src_x,
                source_y_um=src_y,
                source_z_um=src_z,
                sample_x_um=sx,
                sample_y_um=sy,
                sample_z_um=sz,
                material_model=material_model,
                fallback_sigma_S_per_m=sigma,
                min_distance_um=epsilon,
            )

    return phi_total_mV

# Build 1d source positions um.
def build_1d_source_positions_um(
    x_um: np.ndarray,
    y_um: float = 0.0,
    z_um: float = 0.0,
) -> np.ndarray:
    """
    Build 3D source positions from 1D cable x positions.
    """
    x_um = np.asarray(x_um, dtype=float)
    pos = np.zeros((x_um.size, 3), dtype=float)
    pos[:, 0] = x_um
    pos[:, 1] = float(y_um)
    pos[:, 2] = float(z_um)
    return pos

# Compute material aware point source potential mv.
def compute_material_aware_point_source_potential_mV(
    *,
    source_current_uA: float,
    source_x_um: float,
    source_y_um: float,
    source_z_um: float,
    sample_x_um: float,
    sample_y_um: float,
    sample_z_um: float,
    material_model: Optional[RadialMaterialModel],
    fallback_sigma_S_per_m: float = 0.30,
    min_distance_um: float = 1.0,
) -> float:
    """
    Material-aware point-source potential.

    Same overall form as the current homogeneous model, but uses a path-dependent
    effective conductivity across radial material layers.
    """
    dx_um = float(sample_x_um) - float(source_x_um)
    dy_um = float(sample_y_um) - float(source_y_um)
    dz_um = float(sample_z_um) - float(source_z_um)

    d_um = float(np.sqrt(dx_um * dx_um + dy_um * dy_um + dz_um * dz_um))
    d_um = max(d_um, float(min_distance_um))
    d_m = d_um * 1.0e-6

    if material_model is None:
        sigma_eff = float(fallback_sigma_S_per_m)
    else:
        sigma_eff = float(
            material_model.effective_conductivity_between_points(
                source_y_um=source_y_um,
                source_z_um=source_z_um,
                electrode_y_um=sample_y_um,
                electrode_z_um=sample_z_um,
            )
        )

    sigma_eff = max(sigma_eff, 1.0e-8)

    source_current_A = float(source_current_uA) * 1.0e-6
    phi_V = source_current_A / (4.0 * np.pi * sigma_eff * d_m)
    return float(phi_V * 1.0e3)

# Transmembrane current density to total current ua.
def transmembrane_current_density_to_total_current_uA(
    I_membrane_density_uA_per_cm2: np.ndarray,
    compartment_area_cm2: np.ndarray,
) -> np.ndarray:
    """
    Convert membrane current density to total source current per compartment.

    Supported input shapes
    ----------------------
    1. I_membrane_density_uA_per_cm2 shape (N,)
       compartment_area_cm2 shape (N,)

    2. I_membrane_density_uA_per_cm2 shape (T, N)
       compartment_area_cm2 shape (N,)

    Returns
    -------
    np.ndarray
        Total current in uA, with the same leading shape as the input current
        density array.
    """
    I_membrane_density_uA_per_cm2 = np.asarray(I_membrane_density_uA_per_cm2, dtype=float)
    compartment_area_cm2 = np.asarray(compartment_area_cm2, dtype=float)

    if compartment_area_cm2.ndim != 1:
        raise ValueError("compartment_area_cm2 must be a 1D array of shape (N,).")

    if I_membrane_density_uA_per_cm2.ndim == 1:
        if I_membrane_density_uA_per_cm2.shape[0] != compartment_area_cm2.shape[0]:
            raise ValueError("For 1D input, current-density and area arrays must have the same length.")
        return I_membrane_density_uA_per_cm2 * compartment_area_cm2

    if I_membrane_density_uA_per_cm2.ndim == 2:
        if I_membrane_density_uA_per_cm2.shape[1] != compartment_area_cm2.shape[0]:
            raise ValueError(
                "For 2D input, current-density array must have shape (T, N) and "
                "area array must have shape (N,)."
            )
        return I_membrane_density_uA_per_cm2 * compartment_area_cm2[np.newaxis, :]

    raise ValueError("I_membrane_density_uA_per_cm2 must be either 1D or 2D.")

# Sample phi e trace mv.
def sample_phi_e_trace_mV(
    source_current_history_uA: np.ndarray,
    source_positions_um: np.ndarray,
    sample_xyz_um: Tuple[float, float, float],
    params: SimulationParameters = DEFAULT_PARAMS,
    material_model: Optional[RadialMaterialModel] = None,
) -> np.ndarray:
    """
    Sample extracellular potential trace at one point from time-varying sources.

    Parameters
    ----------
    source_current_history_uA
        Array of shape (T, N_sources)
    source_positions_um
        Array of shape (N_sources, 3)
    sample_xyz_um
        One sampling position in um

    Returns
    -------
    np.ndarray
        Potential trace in mV, shape (T,)
    """
    source_current_history_uA = np.asarray(source_current_history_uA, dtype=float)
    source_positions_um = np.asarray(source_positions_um, dtype=float)

    if source_current_history_uA.ndim != 2:
        raise ValueError("source_current_history_uA must have shape (T, N_sources).")
    if source_positions_um.ndim != 2 or source_positions_um.shape[1] != 3:
        raise ValueError("source_positions_um must have shape (N_sources, 3).")
    if source_current_history_uA.shape[1] != source_positions_um.shape[0]:
        raise ValueError("Number of sources in current history and positions must match.")

    T = source_current_history_uA.shape[0]
    phi_trace = np.zeros(T, dtype=float)

    source_positions_list = [tuple(row) for row in source_positions_um]

    for t_idx in range(T):
        phi_trace[t_idx] = compute_superposed_phi_e_mV(
            source_currents_uA=source_current_history_uA[t_idx, :],
            source_positions_um=source_positions_list,
            sample_xyz_um=sample_xyz_um,
            params=params,
            material_model=material_model,
        )

    return phi_trace

# Sample phi e field history mv.
def sample_phi_e_field_history_mV(
    cable_result: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray],
    sample_points: Sequence[ElectrodeSamplePoint],
    params: SimulationParameters = DEFAULT_PARAMS,
    cable_y_um: float = 0.0,
    cable_z_um: float = 0.0,
    material_model: Optional[RadialMaterialModel] = None,
) -> Dict[str, np.ndarray]:
    """
    Sample extracellular potential history at an arbitrary set of points.

    This is the generic field-mapping companion to
    sample_virtual_electrodes_from_cable().

    Returns
    -------
    dict
        {
            "t_ms": shape (T,),
            "phi_e_mV": shape (T, N_points),
            "x_um": shape (N_points,),
            "y_um": shape (N_points,),
            "z_um": shape (N_points,),
            "point_names": shape (N_points,),
        }
    """
    trace_dict = sample_virtual_electrodes_from_cable(
        cable_result=cable_result,
        geometry=geometry,
        electrode_points=sample_points,
        params=params,
        cable_y_um=cable_y_um,
        cable_z_um=cable_z_um,
        material_model=material_model,
    )

    t_ms = np.asarray(trace_dict["t_ms"], dtype=float)

    names = [p.name for p in sample_points]
    x_um = np.asarray([p.x_um for p in sample_points], dtype=float)
    y_um = np.asarray([p.y_um for p in sample_points], dtype=float)
    z_um = np.asarray([p.z_um for p in sample_points], dtype=float)

    phi_cols = []
    for name in names:
        phi_cols.append(np.asarray(trace_dict[name], dtype=float))
    phi_e_mV = np.stack(phi_cols, axis=1)  # shape (T, N_points)

    return {
        "t_ms": t_ms,
        "phi_e_mV": phi_e_mV,
        "x_um": x_um,
        "y_um": y_um,
        "z_um": z_um,
        "point_names": np.asarray(names, dtype=object),
    }

# Sample virtual electrodes from cable.
def sample_virtual_electrodes_from_cable(
    cable_result: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray],
    electrode_points: Sequence[ElectrodeSamplePoint],
    params: SimulationParameters = DEFAULT_PARAMS,
    cable_y_um: float = 0.0,
    cable_z_um: float = 0.0,
    material_model: Optional[RadialMaterialModel] = None,
) -> Dict[str, np.ndarray]:
    """
    Sample extracellular waveforms from a cable simulation at multiple electrode positions.

    Expected cable_result contents
    ------------------------------
    Must contain:
    - "I_ion_uA_per_cm2": shape (T, N)
    and geometry must contain:
    - "area_cm2": shape (N,)
    - "x_um": shape (N,)

    Returns
    -------
    dict
        Mapping:
        - "t_ms" -> time axis
        - electrode name -> phi_e trace in mV
    """
    t_ms = np.asarray(cable_result["t_ms"], dtype=float)
    I_ion_density = np.asarray(cable_result["I_ion_uA_per_cm2"], dtype=float)
    area_cm2 = np.asarray(geometry["area_cm2"], dtype=float)
    x_um = np.asarray(geometry["x_um"], dtype=float)

    source_positions_um = build_1d_source_positions_um(
        x_um=x_um,
        y_um=cable_y_um,
        z_um=cable_z_um,
    )

    # Sign convention:
    # outward positive membrane current density becomes outward source current.
    I_source_history_uA = transmembrane_current_density_to_total_current_uA(
        I_membrane_density_uA_per_cm2=I_ion_density,
        compartment_area_cm2=area_cm2,
    )

    out: Dict[str, np.ndarray] = {"t_ms": t_ms.copy()}

    for electrode in electrode_points:
        out[electrode.name] = sample_phi_e_trace_mV(
            source_current_history_uA=I_source_history_uA,
            source_positions_um=source_positions_um,
            sample_xyz_um=(electrode.x_um, electrode.y_um, electrode.z_um),
            params=params,
            material_model=material_model,
        )

    return out
