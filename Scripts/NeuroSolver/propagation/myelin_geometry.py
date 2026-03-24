"""
myelin_geometry.py

Preset-driven builder for single-fiber myelinated cable geometry.

This version keeps the x-axis cable construction intact while adding y/z
placement and indexing metadata so the same geometry can later be embedded in a
bundle without changing the fast solver interfaces.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..params import (
    DEFAULT_PARAMS,
    SimulationParameters,
    axial_resistance_ohm,
    cylinder_lateral_area_cm2,
)


def _interface_axial_resistance_ohm(
    left_dx_um: float,
    left_diameter_um: float,
    right_dx_um: float,
    right_diameter_um: float,
    rho_i_ohm_cm: float,
) -> float:
    """Axial resistance across a compartment interface, modeled as two half-segments in series."""
    r_left = axial_resistance_ohm(
        length_um=0.5 * float(left_dx_um),
        diameter_um=float(left_diameter_um),
        rho_i_ohm_cm=float(rho_i_ohm_cm),
    )
    r_right = axial_resistance_ohm(
        length_um=0.5 * float(right_dx_um),
        diameter_um=float(right_diameter_um),
        rho_i_ohm_cm=float(rho_i_ohm_cm),
    )
    return r_left + r_right


def _build_label_sequence(node_count: int, internode_subsegments: int) -> List[str]:
    """
    Build the node/internode label sequence.

    Example:
    [node, internode, internode, node, internode, internode, node]
    """
    if node_count < 1:
        raise ValueError("node_count must be >= 1.")
    if internode_subsegments < 1:
        raise ValueError("internode_subsegments must be >= 1.")

    labels: List[str] = []
    for node_idx in range(node_count):
        labels.append("node")
        if node_idx < node_count - 1:
            labels.extend(["internode"] * internode_subsegments)
    return labels


def _resolve_internode_length_um(params: SimulationParameters) -> float:
    """
    Resolve internode length from either:
    - a fixed configured value, or
    - a diameter-based rule with clipping.
    """
    axon_diameter_um = float(params.geometry.axon_diameter_um)
    fiber_diameter_um = float(getattr(params.geometry, "fiber_diameter_um", axon_diameter_um))
    if bool(getattr(params.geometry, "use_diameter_based_internode_length", False)):
        return float(
            np.clip(
                float(getattr(params.geometry, "internode_length_per_fiber_diameter", 100.0)) * fiber_diameter_um,
                float(getattr(params.geometry, "min_internode_length_um", 300.0)),
                float(getattr(params.geometry, "max_internode_length_um", 1500.0)),
            )
        )
    return float(params.geometry.internode_length_um)


def _resolve_node_count(params: SimulationParameters, internode_length_um: float) -> int:
    """
    Resolve node count from either:
    - explicit topology.node_count, or
    - total cable length and spacing rules.
    """
    if not bool(getattr(params.geometry, "use_total_length_driven_topology", False)):
        return int(params.topology.node_count)

    total_length_um = 1000.0 * float(params.geometry.total_length_mm)
    node_length_um = float(params.geometry.node_length_um)
    span_per_interval_um = node_length_um + internode_length_um
    node_count = int(np.floor((total_length_um + internode_length_um) / span_per_interval_um))
    return max(2, node_count)


def build_node_internode_geometry(
    params: SimulationParameters = DEFAULT_PARAMS,
    fiber_id: int = 0,
    fascicle_id: Optional[int] = None,
    fiber_center_y_um: float = 0.0,
    fiber_center_z_um: float = 0.0,
    compartment_global_offset: int = 0,
) -> Dict[str, np.ndarray | int | float | None]:
    """
    Build the single-fiber cable geometry.

    Returns the standard fast-solver geometry plus bundle-aware metadata such as
    y/z placement, fiber/fascicle ids, and local/global compartment indices.
    """
    internode_length_um = _resolve_internode_length_um(params)
    node_count = _resolve_node_count(params, internode_length_um)

    node_length_um = float(params.geometry.node_length_um)
    axon_diameter_um = float(params.geometry.axon_diameter_um)
    fiber_diameter_um = float(getattr(params.geometry, "fiber_diameter_um", axon_diameter_um))
    rho_i = float(params.membrane.axial_resistivity_ohm_cm)

    # Match internode discretization to the target axial resolution.
    target_dx_um = max(float(params.geometry.segment_length_um), 1.0)
    internode_subsegments = max(1, int(round(internode_length_um / target_dx_um)))
    internode_dx_um = internode_length_um / internode_subsegments

    labels = _build_label_sequence(node_count=node_count, internode_subsegments=internode_subsegments)
    n_comp = len(labels)

    dx_um = np.zeros(n_comp, dtype=float)
    diameter_um = np.full(n_comp, axon_diameter_um, dtype=float)
    outer_diameter_um = np.full(n_comp, fiber_diameter_um, dtype=float)

    for i, label in enumerate(labels):
        dx_um[i] = node_length_um if label == "node" else internode_dx_um

    # Compartment centers along x.
    x_um = np.zeros(n_comp, dtype=float)
    x_cursor_um = 0.0
    for i in range(n_comp):
        x_um[i] = x_cursor_um + 0.5 * dx_um[i]
        x_cursor_um += dx_um[i]

    # y/z are constant for one fiber instance inside a bundle.
    y_um = np.full(n_comp, float(fiber_center_y_um), dtype=float)
    z_um = np.full(n_comp, float(fiber_center_z_um), dtype=float)

    total_length_um = float(np.sum(dx_um))
    area_cm2 = np.array(
        [cylinder_lateral_area_cm2(float(dx_um[i]), float(diameter_um[i])) for i in range(n_comp)],
        dtype=float,
    )

    # Interface resistances couple adjacent compartments axially.
    Ra_left_ohm = np.zeros(n_comp, dtype=float)
    Ra_right_ohm = np.zeros(n_comp, dtype=float)
    for i in range(n_comp - 1):
        r_interface = _interface_axial_resistance_ohm(
            left_dx_um=float(dx_um[i]),
            left_diameter_um=float(diameter_um[i]),
            right_dx_um=float(dx_um[i + 1]),
            right_diameter_um=float(diameter_um[i + 1]),
            rho_i_ohm_cm=rho_i,
        )
        Ra_right_ohm[i] = r_interface
        Ra_left_ohm[i + 1] = r_interface

    region_type = np.asarray(labels, dtype=object)

    # Precompute masks and index arrays for downstream bookkeeping and diagnostics.
    node_mask = region_type == "node"
    internode_mask = region_type == "internode"

    node_indices = np.flatnonzero(node_mask).astype(int)
    internode_indices = np.flatnonzero(internode_mask).astype(int)

    node_order = np.full(n_comp, -1, dtype=int)
    node_order[node_indices] = np.arange(node_indices.size, dtype=int)

    compartment_local_index = np.arange(n_comp, dtype=int)
    compartment_global_index = compartment_global_offset + compartment_local_index

    return {
        "n_compartments": int(n_comp),
        "x_um": x_um,
        "y_um": y_um,
        "z_um": z_um,
        "dx_um": dx_um,
        "diameter_um": diameter_um,
        "outer_diameter_um": outer_diameter_um,
        "area_cm2": area_cm2,
        "Ra_left_ohm": Ra_left_ohm,
        "Ra_right_ohm": Ra_right_ohm,
        "region_type": region_type,
        "node_mask": node_mask,
        "internode_mask": internode_mask,
        "node_indices": node_indices,
        "internode_indices": internode_indices,
        "node_order": node_order,
        "compartment_local_index": compartment_local_index,
        "compartment_global_index": compartment_global_index,
        "fiber_id": int(fiber_id),
        "fascicle_id": None if fascicle_id is None else int(fascicle_id),
        "fiber_center_y_um": float(fiber_center_y_um),
        "fiber_center_z_um": float(fiber_center_z_um),
        "internode_subsegments": int(internode_subsegments),
        "node_count_resolved": int(node_count),
        "internode_length_resolved_um": float(internode_length_um),
        "total_length_resolved_um": total_length_um,
    }
