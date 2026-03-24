"""
architecture_builder.py

Bridge from user-facing architecture specs into the existing NeuroSolver bundle
geometry layer.
"""

from __future__ import annotations

from typing import List

from Scripts.NeuroSolver.params import DEFAULT_PARAMS, SimulationParameters
from Scripts.NeuroSolver.propagation.bundle_geometry import (
    BundleGeometry,
    FiberPlacement,
    build_bundle_geometry,
)
from Scripts.NeuroSolver.architecture.architecture_schema import NerveArchitectureSpec

def architecture_to_placements(spec: NerveArchitectureSpec) -> List[FiberPlacement]:
    """
    Convert a NerveArchitectureSpec into the explicit placements expected by
    build_bundle_geometry(...).
    """
    spec.validate()

    placements: List[FiberPlacement] = []
    for fiber in spec.fibers:
        placements.append(
            FiberPlacement(
                fiber_id=int(fiber.fiber_id),
                center_y_um=float(fiber.center_y_um),
                center_z_um=float(fiber.center_z_um),
                fascicle_id=int(fiber.fascicle_id),
                label=fiber.label or f"fiber_{int(fiber.fiber_id)}",
                fiber_diameter_um=(
                    None if fiber.fiber_diameter_um is None else float(fiber.fiber_diameter_um)
                ),
                axon_diameter_um=(
                    None if fiber.axon_diameter_um is None else float(fiber.axon_diameter_um)
                ),
            )
        )
    return placements

def build_bundle_from_architecture(
    spec: NerveArchitectureSpec,
    params: SimulationParameters = DEFAULT_PARAMS,
) -> BundleGeometry:
    """
    Build a BundleGeometry object from a user architecture spec.

    Notes
    -----
    This deliberately reuses the existing bundle geometry engine. The new layer
    here is user-facing specification, not new propagation physics.
    """
    placements = architecture_to_placements(spec)
    if len(placements) == 0:
        raise ValueError("NerveArchitectureSpec must contain at least one FiberSpec.")

    bundle = build_bundle_geometry(
        params=params,
        n_fibers=len(placements),
        layout_name=str(spec.layout_name),
        placements=placements,
        bundle_id=str(spec.bundle_id),
    )

    # Preserve high-level architecture metadata for downstream analysis/plotting.
    bundle.metadata.setdefault("architecture_spec", spec)
    bundle.metadata.setdefault("architecture_length_um", float(spec.length_um))
    bundle.metadata.setdefault(
        "architecture_fascicles",
        [
            {
                "fascicle_id": int(f.fascicle_id),
                "center_y_um": float(f.center_y_um),
                "center_z_um": float(f.center_z_um),
                "radius_um": float(f.radius_um),
                "label": f.label,
                "metadata": dict(f.metadata),
            }
            for f in spec.fascicles
        ],
    )
    bundle.metadata.setdefault(
        "architecture_electrodes",
        [
            {
                "kind": e.kind,
                "x_um": float(e.x_um),
                "y_um": float(e.y_um),
                "z_um": float(e.z_um),
                "radius_um": None if e.radius_um is None else float(e.radius_um),
                "contact_count": int(e.contact_count),
                "spacing_um": None if e.spacing_um is None else float(e.spacing_um),
                "label": e.label,
                "metadata": dict(e.metadata),
            }
            for e in spec.electrodes
        ],
    )
    bundle.metadata.setdefault("user_metadata", dict(spec.metadata))
    return bundle